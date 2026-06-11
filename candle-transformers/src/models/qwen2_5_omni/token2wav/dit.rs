//! Token2Wav DiT — the flow-matching velocity field that turns codec
//! tokens into an 80-dim mel spectrogram.
//!
//! Mirrors `Qwen2_5OmniToken2WavDiTModel` and its sub-blocks in the
//! upstream `modeling_qwen2_5_omni.py` (DiT classes 2809–3076, model +
//! `sample()` 3429–3592, rotary embedding 2442–2505). The numerical
//! flow-matching primitives — the Kutta 3/8-rule integrator, the sway
//! time schedule, and the CFG combine — live in
//! [`super::flow_match`]; this module wires the transformer that the
//! integrator drives. The ECAPA-TDNN speaker encoder lives in
//! [`super::ecapa_tdnn`] and is consumed verbatim by
//! [`DiTInputEmbedding`].
//!
//! ## Pipeline (`sample()`)
//!
//! ```text
//! codes (B,n)  ──text_embed(×repeats=2)──►  code_embed (B, n·2, 512)
//! y0 = randn(1, n·2, 80)
//! for t in sway_schedule(num_steps, -1.0):
//!     v = forward(y, spk_emb, ref_mel, codes, t, apply_cfg)   # velocity
//!     (RK 3/8 step in flow_match.rs)
//! return solution.permute(0,2,1)  → mel (1, 80, n·2)
//! ```
//!
//! ## Critical gotchas (from `hybrie/docs/qwen2_5_omni-phases-2-6-spec.md`
//! §"Phase 4 — Token2Wav DiT" — each silently corrupts output)
//!
//! - **Only head 0 gets RoPE** (upstream line 2991:
//!   `query[:, :1], key[:, :1] = apply_rotary(...)`). The other 15 heads
//!   are positional-blind. Apply rotary to the first head's slice only.
//! - **GPT-J INTERLEAVED rotate** (`rotate_half_codec`, lines 2937-2942):
//!   `x.reshape(..., d/2, 2).unbind(-1)` → `stack(-x2, x1)` → reshape
//!   back, paired with block-layout `cos/sin = cat(freqs, freqs)`. This
//!   is NOT the split-half rotation candle's stock RoPE uses. The cos/sin
//!   layout `cat(freqs, freqs)` is duplicate-block (first half == second
//!   half), so the interleaved rotate consumes adjacent-pair channels.
//! - **AdaLayerNormZero** modulation: `linear(silu(t))` → chunk 6
//!   `[shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]`.
//!   The inner `LayerNorm` is **NON-AFFINE** (`elementwise_affine=False`,
//!   eps 1e-6) — no weight/bias key. Same for `ff_norm` and `norm_out`'s
//!   inner norm.
//! - **GELU approximate="tanh"** in `DiTMLP` (`Activation::Gelu`, the
//!   tanh approximation — candle's `Gelu` IS the tanh form; `GeluErf` is
//!   the exact one).
//! - **Time embed `scale=1000`** in `SinusPositionEmbedding`, NOT 10000
//!   (the `math.log(10000)` inside the frequency formula is a separate
//!   constant — the `scale` that multiplies the timestep is 1000).
//! - **Block-causal streaming mask** `_create_block_diff`: per-token
//!   block index `arange(seq)//block_size(24)`, `block_diff = j - i`,
//!   per-layer admit `(-look_backward ≤ block_diff ≤ look_ahead)`.
//!   `look_ahead=1` iff layer ∈ `[10]`; `look_backward=1` iff layer ∈
//!   `[0, 20]`; else own block only. Converted to additive `-inf/0`.
//! - **CFG batch-doubling happens INSIDE `input_embed`**: `cat([hs,hs])`,
//!   `cat([spk, 0])`, `cat([cond, 0])`, `cat([code, code_uncond])` where
//!   `code_uncond` is the codes embedded with the indices zeroed.
//! - **fp32 forced** throughout (config `torch_dtype="float32"`). On
//!   candle: F32 on CPU, BF16 only behind `--features cuda`.
//! - **`batch_size != 1` raises** in `sample()` (time-embed broadcast
//!   assumes b=1).
//!
//! `rotary_embed.inv_freq` IS in the checkpoint (despite `persistent=False`
//! upstream); we recompute it in `new()` and also register the key so the
//! safetensors load consumes it (no orphan).

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear, ops::softmax_last_dim, Activation, Linear, VarBuilder};

use super::ecapa_tdnn::EcapaTdnn;
use super::flow_match::{cfg_combine, sway_schedule, OdeField, RungeKutta38Solver};
use crate::models::qwen2_5_omni::config::DitConfig;

/// eps for every (non-affine) `LayerNorm` in the DiT (upstream eps=1e-6).
const DIT_LAYER_NORM_EPS: f64 = 1e-6;
/// RoPE base. `DitConfig` carries no `rope_theta`; the default from
/// `configuration_qwen2_5_omni.py` is 10000.
const DIT_ROPE_THETA: f64 = 10000.0;
/// Max sequence length guard. `DitConfig` carries no
/// `max_position_embeddings`; the default is 32768.
const DIT_MAX_POSITION_EMBEDDINGS: usize = 32768;
/// `SinusPositionEmbedding` timestep scale (NOT 10000 — line 3020).
const TIME_EMBED_SCALE: f64 = 1000.0;
/// `SinusPositionEmbedding` frequency embedding width.
const FREQ_EMBED_DIM: usize = 256;

// ---------------------------------------------------------------------------
// Helpers (non-affine LayerNorm + modulation)
// ---------------------------------------------------------------------------

/// A bias/weight-free `LayerNorm` (`elementwise_affine=False`). candle's
/// `layer_norm` always allocates affine params, so we implement the
/// normalize-only form directly. Variance pass in F32 (upstream forces
/// fp32 numerics).
fn layer_norm_non_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let orig_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    normed.to_dtype(orig_dtype)
}

/// AdaLayerNormZero — `linear(silu(emb))` → chunk-6 modulation, then
/// `norm(x) * (1 + scale_msa) + shift_msa`. Returns the modulated hidden
/// states plus the four parameters consumed later in the block
/// (`gate_msa`, `shift_mlp`, `scale_mlp`, `gate_mlp`). Inner norm is
/// non-affine (no params), so this struct holds only the projection.
#[derive(Debug, Clone)]
struct AdaLayerNormZero {
    linear: Linear,
}

impl AdaLayerNormZero {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Linear(dim → 6·dim), bias=True.
        Ok(Self {
            linear: linear(dim, dim * 6, vb.pp("linear"))?,
        })
    }

    /// `hidden_states`: `(B, T, dim)`. `emb` (time embedding): `(B, dim)`.
    /// Returns `(modulated_hidden, gate_msa, shift_mlp, scale_mlp,
    /// gate_mlp)` — each param `(B, dim)`.
    fn forward(
        &self,
        hidden_states: &Tensor,
        emb: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let emb = self.linear.forward(&emb.apply(&Activation::Silu)?)?; // (B, 6·dim)
        let dim = emb.dim(D::Minus1)? / 6;
        let shift_msa = emb.narrow(D::Minus1, 0, dim)?;
        let scale_msa = emb.narrow(D::Minus1, dim, dim)?;
        let gate_msa = emb.narrow(D::Minus1, 2 * dim, dim)?;
        let shift_mlp = emb.narrow(D::Minus1, 3 * dim, dim)?;
        let scale_mlp = emb.narrow(D::Minus1, 4 * dim, dim)?;
        let gate_mlp = emb.narrow(D::Minus1, 5 * dim, dim)?;

        // norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        let normed = layer_norm_non_affine(hidden_states, DIT_LAYER_NORM_EPS)?;
        let scale = (scale_msa.unsqueeze(1)? + 1.0)?; // (B, 1, dim)
        let shift = shift_msa.unsqueeze(1)?; // (B, 1, dim)
        let out = normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;
        Ok((out, gate_msa, shift_mlp, scale_mlp, gate_mlp))
    }
}

/// AdaLayerNormZero_Final — `linear(silu(emb))` → chunk-2 `[scale,
/// shift]`, then `norm(x) * (1 + scale) + shift`. Inner norm non-affine.
#[derive(Debug, Clone)]
struct AdaLayerNormZeroFinal {
    linear: Linear,
}

impl AdaLayerNormZeroFinal {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Linear(dim → 2·dim), bias=True.
        Ok(Self {
            linear: linear(dim, dim * 2, vb.pp("linear"))?,
        })
    }

    /// `hidden_states`: `(B, T, dim)`. `emb`: `(B, dim)`.
    fn forward(&self, hidden_states: &Tensor, emb: &Tensor) -> Result<Tensor> {
        let emb = self.linear.forward(&emb.apply(&Activation::Silu)?)?; // (B, 2·dim)
        let dim = emb.dim(D::Minus1)? / 2;
        let scale = emb.narrow(D::Minus1, 0, dim)?;
        let shift = emb.narrow(D::Minus1, dim, dim)?;

        let normed = layer_norm_non_affine(hidden_states, DIT_LAYER_NORM_EPS)?;
        let scale = (scale.unsqueeze(1)? + 1.0)?; // (B, 1, dim)
        let shift = shift.unsqueeze(1)?;
        normed.broadcast_mul(&scale)?.broadcast_add(&shift)
    }
}

// ---------------------------------------------------------------------------
// Rotary position embedding (GPT-J interleaved, head-0 only)
// ---------------------------------------------------------------------------

/// `Qwen2_5OmniDiTRotaryEmbedding` — recomputes `inv_freq` (default RoPE,
/// `theta=10000`, dim = head_dim) and produces `(cos, sin)` of shape
/// `(seq, head_dim)` via the duplicate-block layout `cat(freqs, freqs)`.
///
/// `inv_freq` is in the checkpoint (`rotary_embed.inv_freq`) despite
/// `persistent=False` upstream; we hold a `Tensor` and register the key
/// so the safetensors load is orphan-free. The held tensor is the
/// recomputed one (numerically identical to the stored buffer).
#[derive(Debug, Clone)]
struct DiTRotaryEmbedding {
    inv_freq: Tensor, // (head_dim/2,)
    head_dim: usize,
}

impl DiTRotaryEmbedding {
    fn new(head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let half = head_dim / 2;
        // inv_freq = 1 / theta^(2i/dim), i in 0..half.
        let inv: Vec<f32> = (0..half)
            .map(|i| 1f32 / (DIT_ROPE_THETA as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let recomputed = Tensor::from_vec(inv, (half,), &device)?;
        // Register the checkpoint key so its tensor is consumed on load.
        // We use the recomputed values (identical to the stored buffer)
        // for forward, but request the key from the VarBuilder so a real
        // safetensors load doesn't report an orphan. When building under a
        // fresh VarMap the get succeeds (zeros); when loading real weights
        // it reads the stored buffer. Either way `inv_freq` below is the
        // recomputed, correct tensor.
        let _stored = vb.get((half,), "inv_freq")?;
        Ok(Self {
            inv_freq: recomputed,
            head_dim,
        })
    }

    /// Build `(cos, sin)` for positions `0..seq`, each `(seq, head_dim)`,
    /// in F32. Layout is `emb = cat(freqs, freqs)` then cos/sin — the
    /// duplicate-block layout the GPT-J interleaved rotate expects.
    fn cos_sin(&self, seq: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let pos = Tensor::arange(0u32, seq as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?; // (seq, 1)
        let inv = self.inv_freq.to_dtype(DType::F32)?.unsqueeze(0)?; // (1, half)
        let freqs = pos.broadcast_mul(&inv)?; // (seq, half)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // (seq, head_dim)
        debug_assert_eq!(emb.dim(D::Minus1)?, self.head_dim);
        Ok((emb.cos()?, emb.sin()?))
    }
}

/// GPT-J interleaved `rotate_half_codec`: reshape `(..., d/2, 2)`, split
/// the trailing pair into `x1, x2`, recombine as `(-x2, x1)`, reshape
/// back. Equivalent to upstream lines 2937-2942.
fn rotate_half_codec(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let last = dims[dims.len() - 1];
    let half = last / 2;
    // (..., d/2, 2)
    let mut new_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    new_shape.push(half);
    new_shape.push(2);
    let x = x.reshape(new_shape)?;
    // unbind along the last dim → x1 = [...,0], x2 = [...,1]
    let x1 = x.narrow(D::Minus1, 0, 1)?; // (..., d/2, 1)
    let x2 = x.narrow(D::Minus1, 1, 1)?; // (..., d/2, 1)
    let neg_x2 = x2.neg()?;
    // stack((-x2, x1), dim=-1) → (..., d/2, 2)
    let stacked = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
    // reshape back to (..., d)
    let mut back_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    back_shape.push(last);
    stacked.reshape(back_shape)
}

/// Apply rotary to a single head's `q`/`k` slice. `q`,`k`: `(B, 1, seq,
/// head_dim)`; `cos`,`sin`: `(seq, head_dim)`. Computed in F32 then cast
/// back (upstream forces fp32 rotary). Returns `(q_rot, k_rot)`.
fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let orig = q.dtype();
    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    // cos/sin (seq, head_dim) → (1, 1, seq, head_dim) for broadcast over
    // (B, 1, seq, head_dim).
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half_codec(&q)?.broadcast_mul(&sin)?)?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half_codec(&k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed.to_dtype(orig)?, k_embed.to_dtype(orig)?))
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// `DiTAttention` — `to_q/to_k/to_v/to_out.0` (all bias=True). Standard
/// scaled-dot-product self-attention with an additive block-causal mask;
/// **only head 0 receives RoPE**. 16 heads, head_dim 64.
#[derive(Debug, Clone)]
struct DiTAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear, // to_out.0 (the dropout at to_out.1 is inference-noop)
    heads: usize,
    head_dim: usize,
}

impl DiTAttention {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let inner = cfg.head_dim * cfg.heads;
        Ok(Self {
            to_q: linear(dim, inner, vb.pp("to_q"))?,
            to_k: linear(dim, inner, vb.pp("to_k"))?,
            to_v: linear(dim, inner, vb.pp("to_v"))?,
            to_out: linear(inner, dim, vb.pp("to_out.0"))?,
            heads: cfg.heads,
            head_dim: cfg.head_dim,
        })
    }

    /// `hidden_states`: `(B, T, dim)`. `cos`,`sin`: `(T, head_dim)`.
    /// `attn_bias`: additive `(T, T)` mask (`-inf`/`0`). Returns `(B, T,
    /// dim)`.
    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attn_bias: &Tensor,
    ) -> Result<Tensor> {
        let (b, t, _) = hidden_states.dims3()?;
        let h = self.heads;
        let hd = self.head_dim;

        // Project & reshape to (B, H, T, head_dim).
        let q = self
            .to_q
            .forward(hidden_states)?
            .reshape((b, t, h, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .to_k
            .forward(hidden_states)?
            .reshape((b, t, h, hd))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .to_v
            .forward(hidden_states)?
            .reshape((b, t, h, hd))?
            .transpose(1, 2)?
            .contiguous()?;

        // RoPE on head 0 only. Slice (B, 1, T, head_dim), rotate, then
        // splice back ahead of the remaining (H-1) heads.
        let q0 = q.narrow(1, 0, 1)?;
        let k0 = k.narrow(1, 0, 1)?;
        let (q0, k0) = apply_rotary_pos_emb(&q0, &k0, cos, sin)?;
        let q = if h > 1 {
            let q_rest = q.narrow(1, 1, h - 1)?;
            Tensor::cat(&[&q0, &q_rest], 1)?
        } else {
            q0
        };
        let k = if h > 1 {
            let k_rest = k.narrow(1, 1, h - 1)?;
            Tensor::cat(&[&k0, &k_rest], 1)?
        } else {
            k0
        };

        // Scaled dot product. softmax in F32 (upstream fp32 numerics).
        let scaling = (hd as f64).powf(-0.5);
        let scores = (q.contiguous()?.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            * scaling)?; // (B, H, T, T)
        let scores = scores.to_dtype(DType::F32)?;
        // Add the block-causal bias (T, T), broadcast over (B, H, .., ..).
        let scores = scores.broadcast_add(&attn_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let weights = softmax_last_dim(&scores)?.to_dtype(v.dtype())?;

        // (B, H, T, head_dim) → (B, T, H·head_dim).
        let attended = weights
            .matmul(&v.contiguous()?)?
            .transpose(1, 2)?
            .reshape((b, t, h * hd))?;
        self.to_out.forward(&attended)
    }
}

// ---------------------------------------------------------------------------
// MLP
// ---------------------------------------------------------------------------

/// `DiTMLP` — `ff.ff.0: Linear(dim → dim·ff_mult)`, GELU(tanh),
/// `ff.ff.3: Linear(dim·ff_mult → dim)`. (`ff.ff.1` is the GELU,
/// `ff.ff.2` is dropout — neither carries weights.)
#[derive(Debug, Clone)]
struct DiTMlp {
    fc1: Linear,
    fc2: Linear,
}

impl DiTMlp {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.hidden_size;
        let inner = dim * cfg.ff_mult;
        // ModuleList indices 0 and 3 carry the linears.
        Ok(Self {
            fc1: linear(dim, inner, vb.pp("ff.0"))?,
            fc2: linear(inner, dim, vb.pp("ff.3"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // candle's `Activation::Gelu` is the tanh approximation
        // (`GeluErf` is the exact one); upstream uses approximate="tanh".
        let h = self.fc1.forward(x)?.apply(&Activation::Gelu)?;
        self.fc2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

/// `DiTDecoderLayer`: AdaLayerNormZero → attention (residual via
/// `gate_msa`), then non-affine `ff_norm` modulated by
/// `(scale_mlp, shift_mlp)` → MLP (residual via `gate_mlp`).
#[derive(Debug, Clone)]
struct DiTDecoderLayer {
    attn_norm: AdaLayerNormZero,
    attn: DiTAttention,
    ff: DiTMlp,
    look_ahead_block: i64,
    look_backward_block: i64,
}

impl DiTDecoderLayer {
    fn new(
        cfg: &DitConfig,
        look_ahead_block: i64,
        look_backward_block: i64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            attn_norm: AdaLayerNormZero::new(cfg.hidden_size, vb.pp("attn_norm"))?,
            attn: DiTAttention::new(cfg, vb.pp("attn"))?,
            ff: DiTMlp::new(cfg, vb.pp("ff"))?,
            look_ahead_block,
            look_backward_block,
        })
    }

    /// `hidden_states`: `(B, T, dim)`; `time_embedding`: `(B, dim)`;
    /// `cos`/`sin`: `(T, head_dim)`; `block_diff`: `(T, T)` of
    /// `block_j - block_i` integers (as F32). Builds the per-layer
    /// additive mask from `(look_backward, look_ahead)`.
    fn forward(
        &self,
        hidden_states: &Tensor,
        time_embedding: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        block_diff: &Tensor,
    ) -> Result<Tensor> {
        // Pre-norm & modulation for attention input.
        let (norm, gate_msa, shift_mlp, scale_mlp, gate_mlp) =
            self.attn_norm.forward(hidden_states, time_embedding)?;

        // Per-layer block-causal additive mask:
        //   admit (block_diff >= -look_backward) & (block_diff <= look_ahead)
        let attn_bias = block_diff_to_bias(
            block_diff,
            self.look_backward_block,
            self.look_ahead_block,
        )?;

        let attn_output = self.attn.forward(&norm, cos, sin, &attn_bias)?;
        // hidden = hidden + gate_msa[:, None] * attn_output
        let hidden_states =
            (hidden_states + gate_msa.unsqueeze(1)?.broadcast_mul(&attn_output)?)?;

        // ff_norm (non-affine) * (1 + scale_mlp) + shift_mlp
        let normed = layer_norm_non_affine(&hidden_states, DIT_LAYER_NORM_EPS)?;
        let scale = (scale_mlp.unsqueeze(1)? + 1.0)?;
        let shift = shift_mlp.unsqueeze(1)?;
        let norm2 = normed.broadcast_mul(&scale)?.broadcast_add(&shift)?;

        let ff_output = self.ff.forward(&norm2)?;
        hidden_states + gate_mlp.unsqueeze(1)?.broadcast_mul(&ff_output)?
    }
}

/// Convert an integer `block_diff` matrix `(T, T)` (as F32) into an
/// additive attention bias: `0` where `-look_backward ≤ diff ≤
/// look_ahead`, `-inf` otherwise.
fn block_diff_to_bias(
    block_diff: &Tensor,
    look_backward: i64,
    look_ahead: i64,
) -> Result<Tensor> {
    let device = block_diff.device();
    let lo = Tensor::full(-(look_backward as f32), block_diff.shape(), device)?;
    let hi = Tensor::full(look_ahead as f32, block_diff.shape(), device)?;
    // admit = (diff >= lo) & (diff <= hi)
    let ge = block_diff.ge(&lo)?; // u8 mask
    let le = block_diff.le(&hi)?;
    let admit = (ge * le)?.to_dtype(DType::F32)?; // 1.0 admit, 0.0 block
    // bias = (admit - 1) * inf  → 0 where admit, -inf where blocked.
    // Build via select to avoid 0·inf = NaN.
    let zeros = Tensor::zeros(block_diff.shape(), DType::F32, device)?;
    let neg_inf = Tensor::full(f32::NEG_INFINITY, block_diff.shape(), device)?;
    // where(admit==1, 0, -inf): admit*0 + (1-admit)*(-inf) is unsafe
    // (0·-inf=NaN); use a boolean-driven combine instead.
    let admit_bool = admit.ne(&zeros)?; // u8: 1 where admit
    admit_bool.where_cond(&zeros, &neg_inf)
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// `DiTCodecEmbedding` — `Embedding(num_embeds+1, emb_dim)`, then
/// `repeat_interleave(repeats=2)` along the sequence axis. `drop_code`
/// zeros the input indices (the unconditional code stream for CFG).
#[derive(Debug, Clone)]
struct DiTCodecEmbedding {
    weight: Tensor, // (num_embeds+1, emb_dim)
    repeats: usize,
}

impl DiTCodecEmbedding {
    fn new(num_embeds: usize, emb_dim: usize, repeats: usize, vb: VarBuilder) -> Result<Self> {
        // num_embeds + 1 rows.
        let weight = vb.get((num_embeds + 1, emb_dim), "codec_embed.weight")?;
        Ok(Self { weight, repeats })
    }

    /// `code`: `(B, n)` i64 indices. `drop_code` → zero the indices.
    /// Returns `(B, n·repeats, emb_dim)`.
    fn forward(&self, code: &Tensor, drop_code: bool) -> Result<Tensor> {
        let (b, n) = code.dims2()?;
        let code = if drop_code {
            Tensor::zeros((b, n), code.dtype(), code.device())?
        } else {
            code.clone()
        };
        // Embedding lookup: flatten indices, index_select, reshape.
        let flat = code.flatten_all()?.to_dtype(DType::U32)?;
        let emb_dim = self.weight.dim(1)?;
        let gathered = self.weight.index_select(&flat, 0)?; // (B·n, emb_dim)
        let emb = gathered.reshape((b, n, emb_dim))?;
        // repeat_interleave along dim=1 by `repeats`.
        repeat_interleave_dim1(&emb, self.repeats)
    }
}

/// `torch.repeat_interleave(x, repeats, dim=1)` for a 3-D tensor `(B, n,
/// D)` → `(B, n·repeats, D)`. Each row `i` is duplicated `repeats` times
/// contiguously (NOT tiling). Implemented via unsqueeze + broadcast +
/// reshape so it is device-agnostic.
fn repeat_interleave_dim1(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (b, n, d) = x.dims3()?;
    // (B, n, 1, D) → expand (B, n, repeats, D) → (B, n·repeats, D)
    let x = x.unsqueeze(2)?; // (B, n, 1, D)
    let x = x.broadcast_as((b, n, repeats, d))?;
    x.reshape((b, n * repeats, d))
}

/// `DiTInputEmbedding` — concat `[hidden, spk_encoder(cond), code_embed,
/// speaker_embedding]` along the feature axis → `proj` to hidden_size.
/// CFG batch-doubling happens HERE. Column order is load-bearing:
/// `[mel(80), spk_enc(128), code(512), spk_emb(192)] = 912`.
#[derive(Debug, Clone)]
struct DiTInputEmbedding {
    proj: Linear,
    spk_encoder: EcapaTdnn,
}

impl DiTInputEmbedding {
    fn new(cfg: &DitConfig, vb: VarBuilder) -> Result<Self> {
        let in_dim = cfg.mel_dim + cfg.enc_dim + cfg.enc_emb_dim + cfg.emb_dim;
        Ok(Self {
            proj: linear(in_dim, cfg.hidden_size, vb.pp("proj"))?,
            spk_encoder: EcapaTdnn::new(cfg, vb.pp("spk_encoder"))?,
        })
    }

    /// - `hidden_states`: noisy mel `(B, T, mel_dim)`
    /// - `speaker_embedding`: `(B, T, enc_emb_dim)` (already repeated over T)
    /// - `condition_vector`: reference mel `(B, T_ref, mel_dim)`
    /// - `code_embed`: `(B, T, emb_dim)` (the conditional code stream)
    /// - `code_embed_uncond`: `(B, T, emb_dim)` (the dropped code stream),
    ///   required when `apply_cfg`.
    ///
    /// When `apply_cfg`, doubles the batch: `[cond; null]`. Returns
    /// `(B', T, hidden_size)` with `B' = 2B` if CFG else `B`.
    fn forward(
        &self,
        hidden_states: &Tensor,
        speaker_embedding: &Tensor,
        condition_vector: &Tensor,
        code_embed: &Tensor,
        code_embed_uncond: Option<&Tensor>,
        apply_cfg: bool,
    ) -> Result<Tensor> {
        let (hidden_states, speaker_embedding, condition_vector, code_embed) = if apply_cfg {
            let hs = Tensor::cat(&[hidden_states, hidden_states], 0)?;
            let spk = Tensor::cat(
                &[speaker_embedding, &speaker_embedding.zeros_like()?],
                0,
            )?;
            let cond = Tensor::cat(&[condition_vector, &condition_vector.zeros_like()?], 0)?;
            let uncond = code_embed_uncond
                .ok_or_else(|| candle::Error::Msg("apply_cfg requires code_embed_uncond".into()))?;
            let code = Tensor::cat(&[code_embed, uncond], 0)?;
            (hs, spk, cond, code)
        } else {
            (
                hidden_states.clone(),
                speaker_embedding.clone(),
                condition_vector.clone(),
                code_embed.clone(),
            )
        };

        // spk_encoder(condition_vector) → (B', enc_dim), then unsqueeze +
        // repeat over T to (B', T, enc_dim).
        let t = hidden_states.dim(1)?;
        let spk_enc = self.spk_encoder.forward(&condition_vector)?; // (B', enc_dim)
        let spk_enc = spk_enc
            .unsqueeze(1)?
            .broadcast_as((spk_enc.dim(0)?, t, spk_enc.dim(1)?))?
            .contiguous()?;

        // Concat in the locked column order and project.
        let cat = Tensor::cat(
            &[&hidden_states, &spk_enc, &code_embed, &speaker_embedding],
            D::Minus1,
        )?;
        self.proj.forward(&cat)
    }
}

// ---------------------------------------------------------------------------
// Timestep embedding
// ---------------------------------------------------------------------------

/// `DiTTimestepEmbedding` — sinusoidal embedding (`scale=1000`,
/// `freq_embed_dim=256`) → `time_mlp.0: Linear(256 → hidden)` → SiLU →
/// `time_mlp.2: Linear(hidden → hidden)`.
#[derive(Debug, Clone)]
struct DiTTimestepEmbedding {
    fc1: Linear,
    fc2: Linear,
    dim: usize, // hidden_size
}

impl DiTTimestepEmbedding {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(FREQ_EMBED_DIM, hidden_size, vb.pp("time_mlp.0"))?,
            fc2: linear(hidden_size, hidden_size, vb.pp("time_mlp.2"))?,
            dim: hidden_size,
        })
    }

    /// `timestep`: `(B,)` scalar timesteps in F32. Returns `(B,
    /// hidden_size)`.
    fn forward(&self, timestep: &Tensor) -> Result<Tensor> {
        let device = timestep.device();
        let half = FREQ_EMBED_DIM / 2;
        // emb = exp(arange(half) * -(ln(10000)/(half-1)))   [the 10000 here
        // is the FIXED frequency constant — distinct from scale=1000].
        let log_inc = (10000f64).ln() / (half as f64 - 1.0);
        let inv: Vec<f32> = (0..half)
            .map(|i| (-(i as f64) * log_inc).exp() as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, half), device)?; // (1, half)
        // scale * t[:, None] * inv[None, :]   (scale = 1000)
        let t = timestep.to_dtype(DType::F32)?.unsqueeze(1)?; // (B, 1)
        let scaled = t
            .broadcast_mul(&inv)?
            .affine(TIME_EMBED_SCALE, 0.0)?; // (B, half)
        let emb = Tensor::cat(&[scaled.sin()?, scaled.cos()?], D::Minus1)?; // (B, 256)
        debug_assert_eq!(emb.dim(D::Minus1)?, FREQ_EMBED_DIM);
        // time_mlp: Linear → SiLU → Linear.
        let h = self.fc1.forward(&emb)?.apply(&Activation::Silu)?;
        let out = self.fc2.forward(&h)?;
        debug_assert_eq!(out.dim(D::Minus1)?, self.dim);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// DiT model
// ---------------------------------------------------------------------------

/// `Qwen2_5OmniToken2WavDiTModel` — the flow-matching velocity field.
/// Build with the VarBuilder scoped to
/// `token2wav.code2wav_dit_model.*`.
#[derive(Debug, Clone)]
pub struct DitModel {
    time_embed: DiTTimestepEmbedding,
    text_embed: DiTCodecEmbedding,
    input_embed: DiTInputEmbedding,
    rotary_embed: DiTRotaryEmbedding,
    transformer_blocks: Vec<DiTDecoderLayer>,
    norm_out: AdaLayerNormZeroFinal,
    proj_out: Linear,
    block_size: usize,
    mel_dim: usize,
    repeats: usize,
}

impl DitModel {
    pub fn new(cfg: &DitConfig, vb: VarBuilder) -> Result<Self> {
        let time_embed = DiTTimestepEmbedding::new(cfg.hidden_size, vb.pp("time_embed"))?;
        // text_embed.codec_embed: Embedding(num_embeds+1, emb_dim=512).
        let text_embed = DiTCodecEmbedding::new(
            cfg.num_embeds,
            cfg.emb_dim,
            cfg.repeats,
            vb.pp("text_embed"),
        )?;
        let input_embed = DiTInputEmbedding::new(cfg, vb.pp("input_embed"))?;
        let rotary_embed = DiTRotaryEmbedding::new(cfg.head_dim, vb.pp("rotary_embed"))?;

        let mut transformer_blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let look_ahead = if cfg.look_ahead_layers.contains(&i) { 1 } else { 0 };
            let look_backward = if cfg.look_backward_layers.contains(&i) {
                1
            } else {
                0
            };
            transformer_blocks.push(DiTDecoderLayer::new(
                cfg,
                look_ahead,
                look_backward,
                vb.pp(format!("transformer_blocks.{i}")),
            )?);
        }

        let norm_out = AdaLayerNormZeroFinal::new(cfg.hidden_size, vb.pp("norm_out"))?;
        let proj_out = linear(cfg.hidden_size, cfg.mel_dim, vb.pp("proj_out"))?;

        Ok(Self {
            time_embed,
            text_embed,
            input_embed,
            rotary_embed,
            transformer_blocks,
            norm_out,
            proj_out,
            block_size: cfg.block_size,
            mel_dim: cfg.mel_dim,
            repeats: cfg.repeats,
        })
    }

    /// Mel output dim (80).
    pub fn mel_dim(&self) -> usize {
        self.mel_dim
    }

    /// Build `block_diff` `(T, T)` of `block_j - block_i` as F32, where
    /// `block_idx = arange(T) // block_size`. Mirrors `_create_block_diff`
    /// (the per-layer admit window is applied later in `block_diff_to_bias`).
    fn create_block_diff(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        let blocks: Vec<f32> = (0..seq_len)
            .map(|i| (i / self.block_size) as f32)
            .collect();
        let block_i = Tensor::from_vec(blocks.clone(), (seq_len, 1), device)?; // column
        let block_j = Tensor::from_vec(blocks, (1, seq_len), device)?; // row
        // block_diff = block_j - block_i  → (T, T) via broadcast.
        block_j.broadcast_sub(&block_i)
    }

    /// The flow-matching velocity field forward used inside `sample()`.
    ///
    /// - `hidden_states`: noisy mel `(B, T, mel_dim)`
    /// - `speaker_embedding`: `(B, T, enc_emb_dim)` (the precomputed
    ///   speaker conditioning, already repeated over T)
    /// - `condition_vector`: reference mel `(B, T_ref, mel_dim)`
    /// - `quantized_code`: `(B, n_codes)` i64 codec tokens
    /// - `time_step`: scalar f32
    /// - `apply_cfg`: when true, doubles the batch internally
    ///   (`[cond; null]`) and returns velocity for both halves.
    ///
    /// Returns the velocity field `(B', T, mel_dim)` (`B' = 2B` under CFG).
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        speaker_embedding: &Tensor,
        condition_vector: &Tensor,
        quantized_code: &Tensor,
        time_step: f32,
        apply_cfg: bool,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.dim(0)?;
        let device = hidden_states.device();

        // Time embedding. After CFG batch-doubling the effective batch is
        // 2·B; the time embedding is broadcast across the doubled batch.
        let eff_batch = if apply_cfg { 2 * batch_size } else { batch_size };
        let t_vec = vec![time_step; eff_batch];
        let t_tensor = Tensor::from_vec(t_vec, (eff_batch,), device)?;
        let time_embedding = self.time_embed.forward(&t_tensor)?; // (B', hidden)

        // Code embeddings (conditional + unconditional for CFG).
        // Upstream: text_embedding = text_embed(code, drop_code=False);
        //           text_embedding_unconditioned = text_embed(code, drop_code=True) if apply_cfg.
        let code_embed = self.text_embed.forward(quantized_code, false)?;
        let code_uncond = if apply_cfg {
            Some(self.text_embed.forward(quantized_code, true)?)
        } else {
            None
        };

        // Input embedding (does the CFG batch-doubling + projection).
        let mut hidden = self.input_embed.forward(
            hidden_states,
            speaker_embedding,
            condition_vector,
            &code_embed,
            code_uncond.as_ref(),
            apply_cfg,
        )?; // (B', T, hidden)

        let seq_len = hidden.dim(1)?;
        // Positional encodings (cos/sin) — positions 0..T, head_dim wide.
        let (cos, sin) = self.rotary_embed.cos_sin(seq_len, device)?;
        let cos = cos.to_dtype(hidden.dtype())?;
        let sin = sin.to_dtype(hidden.dtype())?;
        // Block-causal difference matrix (shared across layers).
        let block_diff = self.create_block_diff(seq_len, device)?;

        // Transformer blocks.
        for block in &self.transformer_blocks {
            hidden = block.forward(&hidden, &time_embedding, &cos, &sin, &block_diff)?;
        }

        // Final modulation + projection to mel.
        let hidden = self.norm_out.forward(&hidden, &time_embedding)?;
        self.proj_out.forward(&hidden) // (B', T, mel_dim)
    }

    /// High-level: codes → mel via flow-matching RK 3/8 sampling.
    ///
    /// - `quantized_code`: `(1, n_codes)` i64 codec tokens
    /// - `conditioning_vector`: `(1, enc_emb_dim)` precomputed speaker
    ///   embedding (the per-voice `cond` from `spk_dict.pt`)
    /// - `reference_mel`: `(1, T_ref, mel_dim)` reference mel for the
    ///   ECAPA speaker encoder
    /// - `num_steps`, `guidance_scale`, `sway_coefficient`: solver knobs
    ///   (upstream defaults `10`, `0.5`, `-1.0`)
    ///
    /// Returns mel `(1, mel_dim, T)` with `T = n_codes · repeats`. Forces
    /// F32 throughout. `batch_size != 1` bails.
    pub fn sample(
        &self,
        quantized_code: &Tensor,
        conditioning_vector: &Tensor,
        reference_mel: &Tensor,
        num_steps: usize,
        guidance_scale: f32,
        sway_coefficient: f32,
    ) -> Result<Tensor> {
        let device = quantized_code.device();
        let (b, n_codes) = quantized_code.dims2()?;
        if b != 1 {
            candle::bail!(
                "Qwen2.5-Omni Token2Wav DiT sample() supports batch size = 1 only (got {b})"
            );
        }
        let maximum_duration = n_codes * self.repeats;
        if maximum_duration > DIT_MAX_POSITION_EMBEDDINGS {
            candle::bail!(
                "requested mel length ({maximum_duration}) exceeds max_position_embeddings ({DIT_MAX_POSITION_EMBEDDINGS})"
            );
        }

        // The model "forces fp32" by being LOADED at F32 (see the
        // token2wav `mod.rs` doc). On CPU the weights are F32, so all
        // tensors here are F32 already; on the CUDA BF16 smoke the weights
        // are BF16. Mirror `audio_encoder.rs`: keep activations in the
        // weight dtype and only upcast the softmax / rotary numerics
        // (already done in the attention + rotary paths). Deriving the
        // working dtype from `reference_mel` matches upstream's
        // `dtype=reference_mel_spectrogram.dtype` on `initial_state`.
        let dtype = reference_mel.dtype();
        let conditioning_vector = conditioning_vector.to_dtype(dtype)?;
        let reference_mel = reference_mel.clone();

        // y0 = randn([1, maximum_duration, mel_dim]) in the working dtype.
        let y0 = Tensor::randn(0f32, 1f32, (1, maximum_duration, self.mel_dim), device)?
            .to_dtype(dtype)?;

        // speaker_embedding = conditioning_vector.unsqueeze(1).repeat(1, T, 1)
        let enc_emb_dim = conditioning_vector.dim(D::Minus1)?;
        let speaker_embedding = conditioning_vector
            .unsqueeze(1)?
            .broadcast_as((1, maximum_duration, enc_emb_dim))?
            .contiguous()?;

        // Time grid via the sway schedule (closed-form at s=-1).
        let time_points = sway_schedule(num_steps, sway_coefficient);

        // Drive the integrator over the closure-backed velocity field.
        let field = DitOdeField {
            model: self,
            speaker_embedding,
            reference_mel,
            quantized_code: quantized_code.clone(),
            guidance_scale,
        };
        let solution = RungeKutta38Solver::integrate_final(&field, &y0, &time_points)?;

        // (1, T, mel_dim) → (1, mel_dim, T).
        solution.permute((0, 2, 1))
    }
}

/// Velocity field adapter so [`RungeKutta38Solver`] can drive the DiT.
/// Closes over the model + the conditioning tensors that are constant
/// across solver steps; `eval(t, y)` runs the DiT forward and (when
/// `guidance_scale ≥ 1e-5`) combines the doubled-batch output via CFG.
struct DitOdeField<'a> {
    model: &'a DitModel,
    speaker_embedding: Tensor,
    reference_mel: Tensor,
    quantized_code: Tensor,
    guidance_scale: f32,
}

impl OdeField for DitOdeField<'_> {
    fn eval(&self, t: f32, y: &Tensor) -> Result<Tensor> {
        if self.guidance_scale < 1e-5 {
            // No CFG: single forward, no batch doubling.
            self.model.forward(
                y,
                &self.speaker_embedding,
                &self.reference_mel,
                &self.quantized_code,
                t,
                false,
            )
        } else {
            // CFG: doubled-batch forward → chunk → combine.
            let model_output = self.model.forward(
                y,
                &self.speaker_embedding,
                &self.reference_mel,
                &self.quantized_code,
                t,
                true,
            )?; // (2, T, mel_dim)
            let b = model_output.dim(0)?;
            let half = b / 2;
            let guided = model_output.narrow(0, 0, half)?;
            let null = model_output.narrow(0, half, half)?;
            cfg_combine(&guided, &null, self.guidance_scale)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::DitConfig;
    use candle::{Device, IndexOp};
    use candle_nn::{VarBuilder, VarMap};

    /// A trivially-small DiT config for CPU shape + key-tree tests. The
    /// magnitudes are tiny but the structure is faithful: `head_dim·heads
    /// == hidden_size` (square to_q/k/v), `enc_channels` len 5 with
    /// `c_last == 3·c1` for the ECAPA sub-encoder, all channels even.
    fn tiny_cfg() -> DitConfig {
        DitConfig {
            hidden_size: 32, // = head_dim(8) · heads(4)
            num_hidden_layers: 2,
            heads: 4,
            head_dim: 8,
            ff_mult: 2,
            mel_dim: 80,
            emb_dim: 16,
            num_embeds: 31, // → Embedding(32, 16)
            repeats: 2,
            enc_dim: 12,    // ECAPA output dim
            enc_emb_dim: 6, // precomputed speaker embed width
            // ECAPA constraints (mirror ecapa_tdnn.rs tiny_cfg): all
            // SE-Res2Net channels uniform (else the Res2Net scale=2
            // residual chunks mismatch), c_last == 3·c1 for MFA.
            enc_channels: vec![16, 16, 16, 16, 48],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 2,
            enc_se_channels: 8,
            enc_lin_neurons: 6,
            dropout: 0.0,
            block_size: 4,
            look_ahead_layers: vec![1],
            look_backward_layers: vec![0],
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(
                0f32,
                0.02,
                v.as_tensor().shape().clone(),
                device,
            )?)?;
        }
        Ok(())
    }

    /// **Phase 4 acceptance — DiT shape smoke.** Tiny config: a small
    /// codes tensor → mel via `sample()`, plus a direct `forward()` in
    /// both no-CFG (B) and CFG (2B) batch modes.
    #[test]
    fn dit_forward_and_sample_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = DitModel::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let n_codes = 4usize;
        let t = n_codes * cfg.repeats; // 8

        // Direct forward — no CFG → batch stays 1.
        let noisy = Tensor::randn(0f32, 1f32, (1, t, cfg.mel_dim), &device)?;
        let spk = Tensor::randn(0f32, 1f32, (1, t, cfg.enc_emb_dim), &device)?;
        let ref_mel = Tensor::randn(0f32, 1f32, (1, 10, cfg.mel_dim), &device)?;
        let codes = Tensor::arange(0u32, n_codes as u32, &device)?
            .to_dtype(DType::I64)?
            .reshape((1, n_codes))?;

        let v = model.forward(&noisy, &spk, &ref_mel, &codes, 0.3, false)?;
        assert_eq!(v.dims(), &[1, t, cfg.mel_dim], "no-cfg velocity shape");
        assert!(v.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));

        // Direct forward — CFG → batch doubles to 2.
        let v2 = model.forward(&noisy, &spk, &ref_mel, &codes, 0.3, true)?;
        assert_eq!(v2.dims(), &[2, t, cfg.mel_dim], "cfg velocity shape");

        // sample() end to end → mel (1, mel_dim, T). Tiny num_steps.
        let mel = model.sample(&codes, &spk.i((.., 0, ..))?, &ref_mel, 2, 0.5, -1.0)?;
        assert_eq!(mel.dims(), &[1, cfg.mel_dim, t], "sample mel shape");
        assert!(mel.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));

        // guidance_scale < 1e-5 path (no CFG inside sample).
        let mel_nocfg = model.sample(&codes, &spk.i((.., 0, ..))?, &ref_mel, 2, 0.0, -1.0)?;
        assert_eq!(mel_nocfg.dims(), &[1, cfg.mel_dim, t]);
        Ok(())
    }

    /// **Phase 4 acceptance — block-diff / mask arithmetic.** Verifies
    /// the additive mask matches the upstream admit window
    /// `(block_diff >= -look_backward) & (block_diff <= look_ahead)`.
    #[test]
    fn block_diff_mask_admits_correct_window() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg(); // block_size = 4
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = DitModel::new(&cfg, vb)?;

        // seq=8 → blocks [0,0,0,0,1,1,1,1].
        let bd = model.create_block_diff(8, &device)?;
        let bd_v: Vec<Vec<f32>> = bd.to_vec2()?;
        // diff[i][j] = block_j - block_i.
        assert_eq!(bd_v[0][0], 0.0); // same block
        assert_eq!(bd_v[0][4], 1.0); // j in block 1, i in block 0
        assert_eq!(bd_v[4][0], -1.0); // j in block 0, i in block 1

        // look_backward=1, look_ahead=0 (own block + one back).
        let bias = block_diff_to_bias(&bd, 1, 0)?;
        let bias_v: Vec<Vec<f32>> = bias.to_vec2()?;
        assert_eq!(bias_v[0][0], 0.0); // diff 0 admitted
        assert_eq!(bias_v[4][0], 0.0); // diff -1 admitted (look_backward=1)
        assert!(bias_v[0][4].is_infinite() && bias_v[0][4] < 0.0); // diff +1 blocked (look_ahead=0)

        // look_ahead=1, look_backward=0 (own block + one ahead).
        let bias2 = block_diff_to_bias(&bd, 0, 1)?;
        let bias2_v: Vec<Vec<f32>> = bias2.to_vec2()?;
        assert_eq!(bias2_v[0][4], 0.0); // diff +1 admitted (look_ahead=1)
        assert!(bias2_v[4][0].is_infinite()); // diff -1 blocked (look_backward=0)
        Ok(())
    }

    /// **Phase 4 acceptance — GPT-J interleaved rotate.** `rotate_half_codec`
    /// on `[a, b, c, d]` (pairs (a,b),(c,d)) → `[-b, a, -d, c]`. This is
    /// the interleaved layout, NOT the split-half `[-c, -d, a, b]`.
    #[test]
    fn rotate_half_codec_is_interleaved() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 1, 1, 4), &device)?;
        let r = rotate_half_codec(&x)?;
        let got: Vec<f32> = r.flatten_all()?.to_vec1()?;
        assert_eq!(got, vec![-2.0, 1.0, -4.0, 3.0], "GPT-J interleaved rotate");
        Ok(())
    }

    /// **Phase 4 acceptance — safetensors key-tree audit.** The
    /// VarBuilder key tree we request must match the released
    /// `token2wav.code2wav_dit_model.*` layout exactly (excluding the
    /// ECAPA sub-encoder, whose `input_embed.spk_encoder.*` keys are
    /// owned + audited by `ecapa_tdnn.rs`). We assert: every non-ECAPA
    /// DiT key present, no extra, and that ECAPA keys all sit under the
    /// `input_embed.spk_encoder.` prefix.
    #[test]
    fn dit_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _model = DitModel::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let all: std::collections::BTreeSet<String> =
            vm.data().lock().unwrap().keys().cloned().collect();

        // Partition: ECAPA keys (owned by ecapa_tdnn.rs) vs DiT-proper.
        let spk_prefix = "input_embed.spk_encoder.";
        let dit_keys: std::collections::BTreeSet<String> = all
            .iter()
            .filter(|k| !k.starts_with(spk_prefix))
            .cloned()
            .collect();
        let ecapa_keys: Vec<&String> = all.iter().filter(|k| k.starts_with(spk_prefix)).collect();
        assert!(
            !ecapa_keys.is_empty(),
            "expected ECAPA sub-encoder keys under {spk_prefix}"
        );

        // Expected non-ECAPA DiT key set.
        let mut want: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for key in [
            "text_embed.codec_embed.weight",
            "time_embed.time_mlp.0.weight",
            "time_embed.time_mlp.0.bias",
            "time_embed.time_mlp.2.weight",
            "time_embed.time_mlp.2.bias",
            "rotary_embed.inv_freq",
            "input_embed.proj.weight",
            "input_embed.proj.bias",
            "norm_out.linear.weight",
            "norm_out.linear.bias",
            "proj_out.weight",
            "proj_out.bias",
        ] {
            want.insert(key.to_string());
        }
        for i in 0..cfg.num_hidden_layers {
            for key in [
                "attn_norm.linear.weight",
                "attn_norm.linear.bias",
                "attn.to_q.weight",
                "attn.to_q.bias",
                "attn.to_k.weight",
                "attn.to_k.bias",
                "attn.to_v.weight",
                "attn.to_v.bias",
                "attn.to_out.0.weight",
                "attn.to_out.0.bias",
                "ff.ff.0.weight",
                "ff.ff.0.bias",
                "ff.ff.3.weight",
                "ff.ff.3.bias",
            ] {
                want.insert(format!("transformer_blocks.{i}.{key}"));
            }
        }

        let extra: Vec<_> = dit_keys.difference(&want).collect();
        let missing: Vec<_> = want.difference(&dit_keys).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "DiT key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );
        Ok(())
    }

    /// **Phase 4 acceptance — real-weight CPU F32 smoke.** Loads the
    /// released `token2wav.code2wav_dit_model.*` slice, runs `sample()`
    /// with a tiny codes tensor (4 codes, num_steps=2) → asserts mel
    /// shape `(1, 80, 8)` and finite. `#[ignore]` keeps it off the
    /// default matrix; invoked from the Lambda smoke runner reading
    /// `QWEN_OMNI_3B_DIR`.
    #[test]
    #[ignore]
    fn real_weight_dit_loads() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");
        let dit_cfg = &cfg.token2wav_config.dit_config;

        let shards: Vec<_> = std::fs::read_dir(&model_dir)
            .expect("list model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        assert!(!shards.is_empty(), "no safetensors shards in {model_dir:?}");

        // CPU F32 — candle's CPU matmul has no bf16 kernel.
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let model = DitModel::new(
            dit_cfg,
            vb.pp("token2wav").pp("code2wav_dit_model"),
        )
        .expect("construct DitModel from real weights");

        // Tiny smoke: 4 codes → T = 8 frames, num_steps=2.
        let n_codes = 4usize;
        let codes = Tensor::arange(0u32, n_codes as u32, &device)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .reshape((1, n_codes))
            .unwrap();
        let cond = Tensor::randn(0f32, 1f32, (1usize, dit_cfg.enc_emb_dim), &device).unwrap();
        let ref_mel = Tensor::randn(0f32, 1f32, (1usize, 50usize, dit_cfg.mel_dim), &device).unwrap();

        let mel = model
            .sample(&codes, &cond, &ref_mel, 2, 0.5, -1.0)
            .expect("sample on real weights");
        let dims = mel.dims().to_vec();
        eprintln!("real_weight_dit_loads: mel shape = {dims:?}");
        assert_eq!(dims, vec![1, dit_cfg.mel_dim, n_codes * dit_cfg.repeats]);
        assert!(mel
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }

    /// **Phase 4 GPU acceptance — real-weight CUDA F32 smoke.**
    ///
    /// Token2Wav (DiT) is **fp32-only** in the released model (upstream
    /// forces `self.token2wav.float()`; `dit_config.torch_dtype="float32"`).
    /// candle's CUDA backend has no BF16 kernels for the DiT op chain, and
    /// the model never runs that config — so this GPU test loads + runs at
    /// **F32 on CUDA**, the actual deployment dtype. Lambda + `--features
    /// cuda` only.
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn real_weight_dit_loads_cuda_f32() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");
        let dit_cfg = &cfg.token2wav_config.dit_config;

        let shards: Vec<_> = std::fs::read_dir(&model_dir)
            .expect("list model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        assert!(!shards.is_empty(), "no safetensors shards in {model_dir:?}");

        let device = Device::new_cuda(0).expect("cuda device 0 (run on a GPU box)");
        // F32 on GPU — Token2Wav is fp32-only.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let model = DitModel::new(dit_cfg, vb.pp("token2wav").pp("code2wav_dit_model"))
            .expect("construct DitModel from real weights");

        let n_codes = 4usize;
        let codes = Tensor::arange(0u32, n_codes as u32, &device)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .reshape((1, n_codes))
            .unwrap();
        let cond = Tensor::randn(0f32, 1f32, (1usize, dit_cfg.enc_emb_dim), &device).unwrap();
        let ref_mel =
            Tensor::randn(0f32, 1f32, (1usize, 50usize, dit_cfg.mel_dim), &device).unwrap();

        let mel = model
            .sample(&codes, &cond, &ref_mel, 2, 0.5, -1.0)
            .expect("sample on real weights (cuda f32)");
        let dims = mel.dims().to_vec();
        eprintln!("real_weight_dit_loads_cuda_f32: mel shape = {dims:?}");
        assert_eq!(dims, vec![1, dit_cfg.mel_dim, n_codes * dit_cfg.repeats]);
        assert_eq!(mel.dtype(), DType::F32);
    }
}

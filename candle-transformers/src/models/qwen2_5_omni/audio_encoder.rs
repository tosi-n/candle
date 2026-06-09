//! Qwen2.5-Omni audio encoder.
//!
//! Mirrors `Qwen2_5OmniAudioEncoder` in the upstream
//! `modeling_qwen2_5_omni.py` (lines 833–929). A Whisper-style encoder:
//!
//! 1. **Mel input** `(1, num_mel_bins=128, T)` (single utterance).
//! 2. **Conv stack**: `conv1(128→1280, k=3, pad=1)` → GELU →
//!    `conv2(1280→1280, k=3, pad=1, stride=2)` → GELU. Net **÷2** so far.
//! 3. **Sinusoidal positional embedding** (Whisper formula, NOT in
//!    safetensors — computed from scratch). `max_source_positions=1500`.
//! 4. **Chunk + pack**: split into chunks of `n_window*2 = 200` mel
//!    frames (= 100 tokens after conv2), each chunk attends only to
//!    itself (block-diagonal attention via per-chunk dense softmax —
//!    matches upstream's eager forward path).
//! 5. **32 × encoder layer** (pre-norm, GELU FFN, dim 1280→5120→1280).
//! 6. **Index-gather pool** `(h[2i] + h[2i+1]) / 2` (final stride-2).
//!    Net downsample is now **÷4** (matches `position_id_per_seconds=25`
//!    for 100 fps mel input).
//! 7. **LayerNorm** + **Linear(1280 → 2048)** → output features in the
//!    Thinker's hidden-state space.
//!
//! ## Phase-1 scope
//!
//! Single-utterance forward only (`(1, 128, T)`). Multi-utterance
//! `feature_lens`-driven packed batches are a Phase-2 concern (alongside
//! the Thinker's multimodal embed merge).
//!
//! ## Critical gotchas (from upstream code audit, see
//! `hybrie/docs/qwen2_5_omni-port.md` §5)
//!
//! - `k_proj` has **no bias**. `q_proj` / `v_proj` / `out_proj` do.
//!   Asymmetric, inherited from Whisper. Get this wrong and weight load
//!   silently fails.
//! - **PRE-norm**: LayerNorm before attention and FFN, residual after.
//! - All `LayerNorm` use `eps = 1e-5` (NOT 1e-6 — that's RMSNorm
//!   elsewhere in the model).
//! - `audio_bos_eos_token` is `Embedding(2, output_dim=2048)`: present in
//!   the safetensors but unused by encoder.forward — the parent
//!   ThinkerModel uses it to wrap encoder output with BOS/EOS markers.
//!   We construct + load it so the VarBuilder pass doesn't complain
//!   about missing keys.
//! - The sinusoidal positional embedding is registered as a
//!   **non-persistent buffer** upstream → NOT in safetensors. Compute it
//!   in `new()`.

use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    conv1d, embedding, layer_norm, linear, linear_no_bias, Activation, Conv1d, Conv1dConfig,
    Embedding, LayerNorm, Linear, Module, VarBuilder,
};

use super::config::AudioEncoderConfig;

/// Default `eps` for every `LayerNorm` in the audio encoder (matches
/// PyTorch's `nn.LayerNorm` default).
const LAYER_NORM_EPS: f64 = 1e-5;

/// One Whisper-style encoder layer: pre-norm self-attention + pre-norm
/// FFN, residual connections. The MLP is `Linear(1280→5120) → GELU →
/// Linear(5120→1280)` with bias on both linears.
#[derive(Debug, Clone)]
struct AudioEncoderLayer {
    self_attn_layer_norm: LayerNorm,
    self_attn: AudioAttention,
    final_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

impl AudioEncoderLayer {
    fn new(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let ffn = cfg.encoder_ffn_dim;
        Ok(Self {
            self_attn_layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("self_attn_layer_norm"))?,
            self_attn: AudioAttention::new(cfg, vb.pp("self_attn"))?,
            final_layer_norm: layer_norm(d, LAYER_NORM_EPS, vb.pp("final_layer_norm"))?,
            fc1: linear(d, ffn, vb.pp("fc1"))?,
            fc2: linear(ffn, d, vb.pp("fc2"))?,
            activation: Activation::Gelu, // exact erf-based GELU
        })
    }

    /// `hidden_states` is a single packed chunk `(N, d_model)`. Attention
    /// is dense within the chunk (one element of the block-diagonal
    /// described in the module doc).
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm attention path with residual.
        let residual = hidden_states;
        let h = self.self_attn_layer_norm.forward(hidden_states)?;
        let h = self.self_attn.forward(&h)?;
        let h = (residual + &h)?;

        // Pre-norm FFN path with residual.
        let residual = &h;
        let n = self.final_layer_norm.forward(&h)?;
        let n = self.fc1.forward(&n)?;
        let n = n.apply(&self.activation)?;
        let n = self.fc2.forward(&n)?;
        residual + &n
    }
}

/// Whisper-style self-attention with the asymmetric bias layout
/// `Qwen2_5OmniAudioAttention` inherits: `k_proj` has no bias; the rest
/// do. Plain scaled-dot-product softmax — no RoPE, no positional bias.
#[derive(Debug, Clone)]
struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear, // NO BIAS
    v_proj: Linear,
    out_proj: Linear,
    n_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl AudioAttention {
    fn new(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        let n_heads = cfg.encoder_attention_heads;
        let head_dim = d / n_heads;
        Ok(Self {
            q_proj: linear(d, d, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?, // bias=False upstream
            v_proj: linear(d, d, vb.pp("v_proj"))?,
            out_proj: linear(d, d, vb.pp("out_proj"))?,
            n_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `(N, d_model)`. Dense self-attention within the chunk.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n = x.dim(0)?;
        // Project, reshape into (n_heads, N, head_dim).
        let q = self
            .q_proj
            .forward(x)?
            .reshape((n, self.n_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((n, self.n_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((n, self.n_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        // Scaled dot product, softmax in f32 (Whisper convention).
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let scores = (scores * self.scaling)?;
        let weights = candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;

        // (n_heads, N, head_dim) → (N, n_heads, head_dim) → (N, d_model)
        let attended = weights
            .matmul(&v)?
            .transpose(0, 1)?
            .reshape((n, self.n_heads * self.head_dim))?;

        self.out_proj.forward(&attended)
    }
}

/// Qwen2.5-Omni audio encoder. Construct with the released checkpoint's
/// `thinker.audio_tower.*` slice of the VarBuilder.
#[derive(Debug, Clone)]
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    /// Not used in `forward`, but in the safetensors and consumed by the
    /// parent ThinkerModel for BOS/EOS audio token wrapping. We construct
    /// it so VarBuilder lookups don't fail on missing keys.
    pub audio_bos_eos_token: Embedding,
    positional_embedding: Tensor,
    layers: Vec<AudioEncoderLayer>,
    ln_post: LayerNorm,
    proj: Linear,
    cfg: AudioEncoderConfig,
}

impl AudioEncoder {
    /// Build from config + a VarBuilder pre-scoped to
    /// `thinker.audio_tower` (or analogous root).
    pub fn new(cfg: &AudioEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Conv stack — both with bias=True (PyTorch Conv1d default).
        let conv1_cfg = Conv1dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let conv2_cfg = Conv1dConfig {
            padding: 1,
            stride: 2,
            ..Default::default()
        };
        let conv1 = conv1d(cfg.num_mel_bins, cfg.d_model, 3, conv1_cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(cfg.d_model, cfg.d_model, 3, conv2_cfg, vb.pp("conv2"))?;

        // Parent-owned embedding (Embedding(2, output_dim)). Present in
        // safetensors but unused by `forward` here.
        let audio_bos_eos_token = embedding(2, cfg.output_dim, vb.pp("audio_bos_eos_token"))?;

        // Whisper-style sinusoidal positional embedding (non-persistent
        // buffer upstream → NOT in safetensors). Computed in place.
        let positional_embedding = sinusoids(cfg.max_source_positions, cfg.d_model, &device)?
            .to_dtype(dtype)?;

        // Transformer layers.
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(AudioEncoderLayer::new(
                cfg,
                vb.pp(format!("layers.{i}")),
            )?);
        }

        let ln_post = layer_norm(cfg.d_model, LAYER_NORM_EPS, vb.pp("ln_post"))?;
        let proj = linear(cfg.d_model, cfg.output_dim, vb.pp("proj"))?;

        Ok(Self {
            conv1,
            conv2,
            audio_bos_eos_token,
            positional_embedding,
            layers,
            ln_post,
            proj,
            cfg: cfg.clone(),
        })
    }

    /// Encoder feature dim (= `output_dim` from config; 2048 for the 3B
    /// model).
    pub fn output_dim(&self) -> usize {
        self.cfg.output_dim
    }

    /// Single-utterance forward.
    ///
    /// Input `mel`: `(1, num_mel_bins, T)` — a single audio sample's
    /// 100 fps log-mel features. Output: `(T_enc, output_dim)` where
    /// `T_enc ≈ T / 4`. Internally chunks the mel into windows of
    /// `n_window*2 = 200` frames; each chunk is encoded independently
    /// (matches upstream's block-diagonal packed attention via the
    /// per-chunk dense path).
    ///
    /// The exact post-pool length formula is `((T - 1)/2 + 1 - 2)/2 + 1`
    /// (conv2's stride-2 then index-gather ÷2), per
    /// `_get_feat_extract_output_lengths` upstream.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let (b, c, t) = mel.dims3()?;
        if b != 1 {
            candle::bail!(
                "Qwen2.5-Omni audio encoder Phase 1 supports single-utterance forward only \
                 (got batch={b}); multi-utterance packed batches are a Phase 2 deliverable",
            );
        }
        if c != self.cfg.num_mel_bins {
            candle::bail!(
                "expected num_mel_bins={}, got {c}",
                self.cfg.num_mel_bins
            );
        }
        let n_window = self.cfg.n_window;
        let chunk_size = n_window * 2; // 200 mel frames per chunk

        // Number of chunks and per-chunk (right-padded) lengths.
        let num_chunks = (t + chunk_size - 1) / chunk_size;
        let mut chunk_lengths = vec![chunk_size; num_chunks];
        let tail = t % chunk_size;
        if tail != 0 {
            chunk_lengths[num_chunks - 1] = tail;
        }

        // Slice the single utterance into `num_chunks` (C, chunk_len)
        // pieces, pad each to chunk_size along the time axis.
        let mel = mel.squeeze(0)?; // (C, T)
        let mut padded_chunks = Vec::with_capacity(num_chunks);
        let mut offset = 0usize;
        for &len in &chunk_lengths {
            let slice = mel.narrow(1, offset, len)?; // (C, len)
            let chunk = if len < chunk_size {
                let pad = Tensor::zeros(
                    (self.cfg.num_mel_bins, chunk_size - len),
                    mel.dtype(),
                    mel.device(),
                )?;
                Tensor::cat(&[&slice, &pad], 1)?
            } else {
                slice
            };
            padded_chunks.push(chunk.unsqueeze(0)?); // (1, C, chunk_size)
            offset += len;
        }
        let padded = Tensor::cat(&padded_chunks, 0)?; // (num_chunks, C, chunk_size)

        // Conv stack. Pre-conv mask zeroes out padding so it doesn't
        // bleed into convolution outputs. Mask shape: (num_chunks, 1,
        // chunk_size).
        let mut mask_rows: Vec<Tensor> = Vec::with_capacity(num_chunks);
        let pos = Tensor::arange(0u32, chunk_size as u32, mel.device())?
            .to_dtype(padded.dtype())?;
        for &len in &chunk_lengths {
            let len_t = Tensor::new(&[len as u32], mel.device())?
                .to_dtype(padded.dtype())?;
            let m = pos.broadcast_lt(&len_t)?.to_dtype(padded.dtype())?; // (chunk_size,)
            mask_rows.push(m.unsqueeze(0)?.unsqueeze(0)?); // (1, 1, chunk_size)
        }
        let pad_mask = Tensor::cat(&mask_rows, 0)?; // (num_chunks, 1, chunk_size)

        // conv1 (stride 1) → GELU → mask out padding.
        let h = self.conv1.forward(&padded)?;
        let h = h.apply(&Activation::Gelu)?;
        let h = h.broadcast_mul(&pad_mask)?;

        // conv2 (stride 2) → GELU. Output: (num_chunks, d_model, ⌈chunk_size/2⌉).
        let h = self.conv2.forward(&h)?;
        let h = h.apply(&Activation::Gelu)?;

        // Transpose to (num_chunks, T', d_model) and add positional
        // embedding (broadcast across chunks).
        let h = h.transpose(1, 2)?; // (num_chunks, T', d_model)
        let t_prime = h.dim(1)?;
        let pos_emb = self
            .positional_embedding
            .narrow(0, 0, t_prime)?
            .unsqueeze(0)?
            .broadcast_as(h.shape())?
            .to_dtype(h.dtype())?;
        let mut h = (&h + &pos_emb)?;

        // Run all encoder layers per-chunk (block-diagonal attention
        // across the packed sequence is equivalent to per-chunk dense
        // attention when every chunk has the same uniform length).
        // Each chunk becomes (T', d_model).
        let mut chunk_outputs: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for i in 0..num_chunks {
            let mut chunk = h.i(i)?; // (T', d_model)
            for layer in &self.layers {
                chunk = layer.forward(&chunk)?;
            }
            chunk_outputs.push(chunk);
        }
        // Re-stack and drop padded positions per chunk using conv2's
        // length formula `after_conv2 = (chunk_len - 1)/2 + 1`.
        let mut valid_pieces: Vec<Tensor> = Vec::with_capacity(num_chunks);
        for (i, chunk) in chunk_outputs.iter().enumerate() {
            let valid = (chunk_lengths[i] - 1) / 2 + 1; // post-conv2 length
            valid_pieces.push(chunk.i(0..valid)?);
        }
        h = Tensor::cat(&valid_pieces, 0)?; // (S, d_model)

        // Final stride-2 average pool via index gather:
        // `(h[2i] + h[2i+1]) / 2`. Handle odd S by truncating.
        let s = h.dim(0)?;
        let pooled = pool_pairs_avg(&h, s)?;

        // ln_post + projection 1280 → 2048.
        let h = self.ln_post.forward(&pooled)?;
        self.proj.forward(&h)
    }
}

/// Whisper sinusoidal positional embedding (length × channels), exactly
/// `SinusoidsPositionEmbedding` from upstream. `max_timescale=10000`.
/// Returns f32 — caller casts to the encoder's dtype.
fn sinusoids(length: usize, channels: usize, device: &Device) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let half = channels / 2;
    let log_inc = max_timescale.ln() / (half as f32 - 1.0);
    let inv: Vec<f32> = (0..half).map(|i| (-(i as f32) * log_inc).exp()).collect();
    let inv = Tensor::from_vec(inv, (1, half), device)?;
    let arange =
        Tensor::arange(0u32, length as u32, device)?.to_dtype(DType::F32)?.unsqueeze(1)?;
    let scaled = arange.broadcast_mul(&inv)?; // (length, half)
    Tensor::cat(&[scaled.sin()?, scaled.cos()?], 1)
}

/// Stride-2 average pool implemented as `(h[2i] + h[2i+1]) / 2`. Mirrors
/// the upstream `get_pool_indices` gather — clean and length-formula
/// compatible. Drops a final unpaired row when `s` is odd.
fn pool_pairs_avg(h: &Tensor, s: usize) -> Result<Tensor> {
    let paired = s & !1; // largest even ≤ s
    let d = h.dim(1)?;
    let even = h.i(0..paired)?.reshape((paired / 2, 2, d))?; // (S/2, 2, d)
    even.mean(1) // (S/2, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::AudioEncoderConfig;
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// A trivially-small encoder config used by all CPU shape tests.
    /// Picks dims that are tiny but structurally faithful: stride/pad
    /// match the real model, only the magnitudes change.
    fn tiny_cfg() -> AudioEncoderConfig {
        AudioEncoderConfig {
            d_model: 32,
            num_hidden_layers: 2,
            encoder_attention_heads: 4,
            encoder_ffn_dim: 64,
            num_mel_bins: 16,
            max_source_positions: 50,
            n_window: 5, // chunk_size = 10
            output_dim: 24,
            scale_embedding: false,
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

    /// **Phase 1 acceptance #1** — single-utterance forward produces the
    /// expected ÷4-downsampled output dim. Tests both an aligned input
    /// (T divisible by chunk_size) and a tail-padded one.
    #[test]
    fn audio_encoder_single_utterance_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg(); // chunk_size = 10
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let enc = AudioEncoder::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        // Aligned: T=20 → 2 chunks of 10 → post-conv2 5 each → packed 10
        // → pooled 5. Output (5, output_dim=24).
        let mel = Tensor::randn(0f32, 1f32, (1, cfg.num_mel_bins, 20), &device)?;
        let out = enc.forward(&mel)?;
        assert_eq!(out.dims(), &[5, cfg.output_dim]);
        assert!(out.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));

        // Tail-padded: T=15 → 2 chunks (10+5) → post-conv2 lengths
        // {(10-1)/2+1=5, (5-1)/2+1=3} → packed 8 → pooled 4.
        let mel = Tensor::randn(0f32, 1f32, (1, cfg.num_mel_bins, 15), &device)?;
        let out = enc.forward(&mel)?;
        assert_eq!(out.dims(), &[4, cfg.output_dim]);
        Ok(())
    }

    /// **Phase 1 acceptance #2** — the VarBuilder key tree we request
    /// matches the released `thinker.audio_tower.*` checkpoint layout
    /// exactly. Building under a VarMap collects every requested name;
    /// we assert every expected key is present + no extra. This is the
    /// test that would have caught the duplicate-alias trap in config.rs
    /// (advisor #1, prior session).
    #[test]
    fn audio_encoder_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _enc = AudioEncoder::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let names: std::collections::BTreeSet<String> = vm
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        // The full set we expect, modelled on the 3B safetensors index.
        let mut want: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for key in [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "audio_bos_eos_token.weight",
            "ln_post.weight",
            "ln_post.bias",
            "proj.weight",
            "proj.bias",
        ] {
            want.insert(key.to_string());
        }
        for i in 0..cfg.num_hidden_layers {
            for key in [
                "self_attn_layer_norm.weight",
                "self_attn_layer_norm.bias",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                // CRITICAL: no `self_attn.k_proj.bias` — k_proj is
                // bias-less to match upstream's Whisper inheritance.
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "fc1.weight",
                "fc1.bias",
                "fc2.weight",
                "fc2.bias",
                "final_layer_norm.weight",
                "final_layer_norm.bias",
            ] {
                want.insert(format!("layers.{i}.{key}"));
            }
        }

        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );
        Ok(())
    }

    /// **Phase 1 acceptance #3** — real-weight smoke against
    /// `Qwen/Qwen2.5-Omni-3B` audio_tower shards. Lambda-only;
    /// `#[ignore]` keeps it out of the default `cargo test` matrix.
    /// Invoked from the Lambda smoke runner (see
    /// `hybrie/scripts/lambda_omni_phase1.sh`).
    #[test]
    #[ignore]
    fn real_weight_audio_encoder_loads() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
        use std::path::PathBuf;

        let model_dir =
            std::env::var("QWEN_OMNI_3B_DIR").expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text = std::fs::read_to_string(model_dir.join("config.json"))
            .expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");

        // Discover the safetensors shards.
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

        let device = Device::Cpu; // weight-load smoke — no inference, just construction
        // Released weights are bf16 on disk; cast to F32 on read so the
        // smoke runs on CPU (candle's CPU matmul has no bf16 kernel —
        // GPU bf16 lives in Phase 2 alongside the Thinker integration).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let enc = AudioEncoder::new(
            &cfg.thinker_config.audio_config,
            vb.pp("thinker").pp("audio_tower"),
        )
        .expect("construct AudioEncoder from real weights");

        // Tiny forward at the real dims (128 mel bins, 400 frames = 4 s
        // of 100 fps mel) to prove the conv stack + a couple of layers
        // execute without panic.
        let mel = Tensor::randn(0f32, 1f32, (1usize, enc.cfg.num_mel_bins, 400), &device)
            .unwrap();
        let out = enc.forward(&mel).expect("forward on real weights");
        // ÷4 downsample on 400 mel frames → ~100 tokens. With chunking
        // of 200, two chunks of 100 post-conv2 → pooled to 50 each →
        // packed 100. Output dim = 2048.
        let (n, d) = (out.dim(0).unwrap(), out.dim(1).unwrap());
        eprintln!("real_weight_audio_encoder_loads: output shape = ({n}, {d})");
        assert_eq!(d, enc.output_dim());
        assert!(n > 0 && n <= 400);
    }
}

//! Qwen2.5-Omni Thinker text decoder.
//!
//! Mirrors `Qwen2_5OmniThinkerTextModel` (modeling_qwen2_5_omni.py:1602-
//! 1714) + the wrapper `Qwen2_5OmniThinkerForConditionalGeneration`
//! (lines 1722-2077). The decoder is **standard Qwen2.5** with M-RoPE
//! substituted for 1-D RoPE:
//!
//! - hidden 2048, 36 layers, 16 heads, 2 kv-heads (GQA group=8),
//!   head_dim 128, intermediate 11008, vocab 151936, rope_theta 1e6,
//!   rms_eps 1e-6, max_pos 32768
//! - q/k/v `bias=True`, o `bias=False` (matches candle's `qwen2.rs`)
//! - SwiGLU MLP with **no bias** (matches qwen2.rs)
//! - RMSNorm eps 1e-6, variance computed in F32 then cast back
//! - **No q-norm / k-norm** — Qwen3-only; zero such tensors in the
//!   released safetensors
//! - **No sliding window** — `use_sliding_window=false` collapses every
//!   layer to full attention; `max_window_layers=70` is dead config
//! - M-RoPE section `[16, 24, 24]` (`sum = 64 = head_dim/2`)
//!
//! ## Phase 2 scope
//!
//! Decoder + lm_head + a text-only forward path proving the safetensors
//! key tree loads against the released `thinker.model.*` shards. The
//! multimodal embed splice (`masked_scatter` on placeholder positions)
//! and the full `get_rope_index` multimodal position builder are
//! follow-ups — this PR lands the LM backbone so Phase 3 (Talker) can
//! consume it.

use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Activation, Embedding,
    Linear, RmsNorm, VarBuilder,
};

use super::config::ThinkerTextConfig;
use super::mrope::{text_only_position_ids, MRopeTable};

/// RMSNorm eps for every norm in the decoder (PyTorch default for the
/// Qwen2.5 family; config field `rms_norm_eps`).
const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;

/// Standard Qwen2.5 self-attention with M-RoPE. The bias layout matches
/// upstream: `q/k/v_proj` have biases, `o_proj` does not.
#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rotary: std::sync::Arc<MRopeTable>,
    /// Per-layer KV cache for incremental decode: post-RoPE,
    /// pre-`repeat_kv` (k, v), each `(B, n_kv_heads, L, head_dim)`.
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        cfg: &ThinkerTextConfig,
        rotary: std::sync::Arc<MRopeTable>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let kv_size = n_kv_heads * head_dim;
        Ok(Self {
            q_proj: linear(hidden, n_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear(hidden, kv_size, vb.pp("k_proj"))?,
            v_proj: linear(hidden, kv_size, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(n_heads * head_dim, hidden, vb.pp("o_proj"))?,
            n_heads,
            n_kv_heads,
            head_dim,
            rotary,
            kv_cache: None,
        })
    }

    /// `x: (B, S, hidden)`, `position_ids: (3, B, S)` carrying ABSOLUTE
    /// positions. Appends this chunk's (k, v) to the per-layer cache so
    /// decode steps pass one new token. Returns `(B, S, hidden)`.
    fn forward(&mut self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        // Project + reshape to (B, H, S, head_dim).
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, s, self.n_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, s, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // M-RoPE applied to q + k.
        let (q, k) = self.rotary.apply(&q, &k, position_ids)?;

        // KV cache append (decode steps carry one token; prefill carries
        // the whole prompt with an empty cache).
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2)?.contiguous()?,
                Tensor::cat(&[pv, &v], 2)?.contiguous()?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // GQA: repeat kv heads to match q heads.
        let groups = self.n_heads / self.n_kv_heads;
        let k = if groups > 1 { repeat_kv(&k, groups)? } else { k };
        let v = if groups > 1 { repeat_kv(&v, groups)? } else { v };

        // Causal scaled-dot-product attention. We use a simple additive
        // mask of `-inf` above the diagonal; the F32 softmax avoids the
        // BF16 numerical artifacts that bit the audio encoder.
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let scores_f32 = scores.to_dtype(DType::F32)?;
        let scores_masked = apply_causal_mask(&scores_f32)?;
        let weights = softmax_last_dim(&scores_masked)?.to_dtype(q.dtype())?;
        let attn = weights.matmul(&v)?;

        // (B, H, S, head_dim) → (B, S, hidden)
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Repeat each KV head `groups` times along the head axis to match q
/// heads under GQA. `x: (B, kv, S, head_dim)` → `(B, kv·groups, S, head_dim)`.
fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (b, kv, s, d) = x.dims4()?;
    let x = x.unsqueeze(2)?; // (B, kv, 1, S, d)
    let x = x.broadcast_as((b, kv, groups, s, d))?;
    x.reshape((b, kv * groups, s, d))
}

/// Build an `(S, S)` causal mask in F32 with `0` on/below diagonal and
/// `-inf` above. Broadcasts over batch+head when added to scores
/// `(B, H, S, S)`.
fn apply_causal_mask(scores: &Tensor) -> Result<Tensor> {
    let q = scores.dim(D::Minus2)?;
    let kv = scores.dim(D::Minus1)?;
    if q == 1 {
        // A single decode step attends the whole cache — no mask.
        return Ok(scores.clone());
    }
    let offset = kv - q;
    let device = scores.device();
    let mut row = Vec::with_capacity(q * kv);
    for i in 0..q {
        for j in 0..kv {
            row.push(if j > i + offset { f32::NEG_INFINITY } else { 0f32 });
        }
    }
    let mask = Tensor::from_vec(row, (q, kv), device)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    scores.broadcast_add(&mask)
}

/// SwiGLU MLP, no bias on any linear — matches Qwen2 and qwen2.rs idiom.
#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    fn new(cfg: &ThinkerTextConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            activation: Activation::Silu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.activation)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(&gate * &up)?)
    }
}

/// One decoder layer: pre-norm attention + pre-norm MLP, both with
/// residual connections.
#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(
        cfg: &ThinkerTextConfig,
        rotary: std::sync::Arc<MRopeTable>,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: rms_norm(
                cfg.hidden_size,
                DEFAULT_RMS_NORM_EPS,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::new(cfg, rotary, vb.pp("self_attn"))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                DEFAULT_RMS_NORM_EPS,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&mut self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.self_attn.forward(&h, position_ids)?;
        let x = (residual + &h)?;
        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        residual + &h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Qwen2.5-Omni Thinker text-only LLM backbone.
///
/// Construct from a VarBuilder pre-scoped to `thinker.model` for the
/// inner decoder, and a separate `lm_head` projection from
/// `thinker.lm_head`.
#[derive(Debug, Clone)]
pub struct Thinker {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    cfg: ThinkerTextConfig,
    rotary: std::sync::Arc<MRopeTable>,
}

impl Thinker {
    /// `vb_root` should be scoped at `thinker.` — i.e. the VarBuilder
    /// looking at the released checkpoint at `thinker.model.*` and
    /// `thinker.lm_head.*`. Internally this calls `vb_root.pp("model")`
    /// and `vb_root.pp("lm_head")`.
    pub fn new(cfg: &ThinkerTextConfig, vb_root: VarBuilder) -> Result<Self> {
        let device = vb_root.device().clone();
        let dtype = vb_root.dtype();

        let rotary = std::sync::Arc::new(MRopeTable::new(
            cfg.head_dim(),
            cfg.rope_theta,
            cfg.max_position_embeddings,
            &cfg.rope_scaling.mrope_section,
            &device,
            dtype,
        )?);

        let vb_model = vb_root.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_model.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                vb_model.pp(format!("layers.{i}")),
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, DEFAULT_RMS_NORM_EPS, vb_model.pp("norm"))?;

        // tie_word_embeddings=false for Qwen2.5-Omni-3B Thinker — load
        // an explicit `thinker.lm_head.weight`.
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_root.pp("lm_head"))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cfg: cfg.clone(),
            rotary,
        })
    }

    pub fn embed_tokens(&self) -> &Embedding {
        &self.embed_tokens
    }

    pub fn config(&self) -> &ThinkerTextConfig {
        &self.cfg
    }

    /// Text-only forward — embeds `input_ids: (B, S)` and runs the full
    /// decoder + lm_head, returning logits `(B, S, vocab_size)`.
    /// `offset` is the prefill offset for position ids (0 on the first
    /// pass; non-zero for subsequent decode steps once KV cache is wired).
    ///
    /// Phase 2 has **no KV cache** wired yet — every call computes the
    /// full causal forward. KV-cached generation lands in Phase 3
    /// alongside the Talker decode loop (the two share the same Qwen2
    /// decoder pattern + cache infra).
    pub fn forward_text_only(&mut self, input_ids: &Tensor, offset: usize) -> Result<Tensor> {
        // Stateless semantics preserved: every call starts from an empty
        // cache (callers that want incremental decode use
        // `forward_text_only_cached` and manage clearing themselves).
        self.clear_kv_cache();
        self.forward_text_only_cached(input_ids, offset)
    }

    /// KV-cached forward chunk: appends to the per-layer cache, so a
    /// decode step passes ONE token with its absolute `offset`. Callers
    /// MUST `clear_kv_cache` before a new sequence.
    pub fn forward_text_only_cached(
        &mut self,
        input_ids: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let mut h = self.embed_tokens.forward(input_ids)?;
        let position_ids = text_only_position_ids(b, s, offset, h.device())?;
        for layer in &mut self.layers {
            h = layer.forward(&h, &position_ids)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Reset all per-layer KV caches — MUST run before every new sequence.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Variant that accepts pre-spliced `inputs_embeds: (B, S, hidden)`
    /// — the entry point used by the multimodal splice path (where
    /// audio/vision encoder outputs have already been scattered into
    /// the LM's embedding stream at placeholder positions). Multimodal
    /// position-id construction is the responsibility of the caller;
    /// for now we accept arbitrary `position_ids: (3, B, S)`.
    pub fn forward_with_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        self.clear_kv_cache();
        let mut h = inputs_embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, position_ids)?;
        }
        let h = self.norm.forward(&h)?;
        self.lm_head.forward(&h)
    }

    /// Embed `ids: (B, S)` through the Thinker's input embedding table,
    /// returning `(B, S, hidden)`. Thin wrapper so the end-to-end
    /// orchestrator (inference.rs) can place text-special tokens (BOS /
    /// EOS / PAD) into the Talker prefill in Thinker-hidden space.
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(ids)
    }

    /// Greedy text generation (`do_sample=false`). Starting from
    /// `input_ids: (1, S)`, repeatedly runs the full causal forward
    /// (no KV cache — full recompute each step), argmaxes the last
    /// position, appends, and stops at `eos` or after `max_new` tokens.
    ///
    /// Returns ONLY the newly-generated ids `(1, G)` (G may be 0 if the
    /// very first sampled token is `eos`). Batch size 1.
    pub fn generate_greedy(
        &mut self,
        input_ids: &Tensor,
        max_new: usize,
        eos: i64,
    ) -> Result<Tensor> {
        let (b, prompt_len) = input_ids.dims2()?;
        if b != 1 {
            candle::bail!("generate_greedy: batch size must be 1, got {b}");
        }
        let device = input_ids.device().clone();
        // KV-cached: prefill once, then ONE token per step with its
        // absolute position — O(N) forwards instead of the previous
        // full-recompute O(N²).
        self.clear_kv_cache();
        let mut gen: Vec<i64> = Vec::with_capacity(max_new);
        let mut logits = self.forward_text_only_cached(input_ids, 0)?;
        for step in 0..max_new {
            let s = logits.dim(1)?;
            let last = logits.i((0, s - 1, ..))?.to_dtype(DType::F32)?;
            let next = argmax_i64(&last)?;
            gen.push(next);
            if next == eos || step + 1 == max_new {
                break;
            }
            let tok = Tensor::from_vec(vec![next], (1, 1), &device)?;
            logits = self.forward_text_only_cached(&tok, prompt_len + step)?;
        }
        self.clear_kv_cache();
        let g = gen.len();
        Tensor::from_vec(gen, (1, g), &device)
    }

    /// Run a single full causal forward over `input_ids: (1, S)` and
    /// capture two per-position tensors needed by the Talker fusion:
    ///
    /// - `last_hidden`: the decoder output AFTER the final RMSNorm but
    ///   BEFORE `lm_head` — `(1, S, hidden=2048)`.
    /// - `tok_embeds`: the layer-0 input embedding (`embed_tokens`) —
    ///   `(1, S, hidden=2048)`.
    ///
    /// Because the decoder is causal, capturing both for ALL positions
    /// in one pass over `[prompt + generated]` is identical to
    /// capturing them step-by-step during generation (upstream's
    /// approach), which is the simplification the orchestrator relies on.
    pub fn forward_collect(&mut self, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        self.clear_kv_cache();
        let (b, s) = input_ids.dims2()?;
        let tok_embeds = self.embed_tokens.forward(input_ids)?;
        let position_ids = text_only_position_ids(b, s, 0, tok_embeds.device())?;
        let mut h = tok_embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, &position_ids)?;
        }
        let last_hidden = self.norm.forward(&h)?;
        Ok((last_hidden, tok_embeds))
    }
}

/// Argmax over a 1-D F32 tensor, returning the index as i64. Used for
/// greedy decode on CPU where a `to_vec1` + manual scan is simpler and
/// dtype-safe than tensor `argmax` + cast.
fn argmax_i64(v: &Tensor) -> Result<i64> {
    let xs = v.to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in xs.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    Ok(best as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::{RopeScaling, ThinkerTextConfig};
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// A trivially small Thinker config for CPU shape tests. Picks the
    /// same architectural pattern as 3B (q/k/v bias, no q_norm, M-RoPE)
    /// at tiny dims so unit tests run in milliseconds.
    ///
    /// hidden=64, heads=4 → head_dim=16 → M-RoPE invariant requires
    /// `sum(mrope_section) == 8`, so `[2, 3, 3]` (sum=8) is the
    /// structurally-faithful tiny analog of the 3B `[16, 24, 24]`.
    fn tiny_cfg() -> ThinkerTextConfig {
        ThinkerTextConfig {
            vocab_size: 64,
            hidden_size: 64,
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2, // GQA group=2
            max_position_embeddings: 32,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            rope_scaling: RopeScaling {
                mrope_section: vec![2, 3, 3], // sum 8 = head_dim/2 = 16/2 ✓
            },
            head_dim: None, // derived 64/4 = 16
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.02, v.as_tensor().shape().clone(), device)?)?;
        }
        Ok(())
    }

    /// **Phase 2 acceptance #1** — text-only forward produces the
    /// expected `(B, S, vocab_size)` logits and all values are finite.
    #[test]
    fn thinker_text_only_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let mut thinker = Thinker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let input_ids = Tensor::from_vec(vec![1i64, 2, 3, 4, 5], (1, 5), &device)?;
        let logits = thinker.forward_text_only(&input_ids, 0)?;
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        assert!(logits.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite()));
        Ok(())
    }

    /// **KV-cache equivalence** — the cached greedy loop must emit the
    /// SAME token sequence as a manual full-recompute loop (causal
    /// attention with absolute positions makes them mathematically
    /// identical; this guards the cache/mask/offset arithmetic).
    #[test]
    fn cached_greedy_matches_full_recompute() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let mut thinker = Thinker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let input_ids = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
        let max_new = 6;
        let eos = -1; // never fires — force max_new tokens both ways

        let cached = thinker
            .generate_greedy(&input_ids, max_new, eos)?
            .flatten_all()?
            .to_vec1::<i64>()?;

        // Manual full-recompute reference.
        let mut all: Vec<i64> = input_ids.flatten_all()?.to_vec1::<i64>()?;
        let mut reference: Vec<i64> = Vec::new();
        for _ in 0..max_new {
            let cur = Tensor::from_vec(all.clone(), (1, all.len()), &device)?;
            let logits = thinker.forward_text_only(&cur, 0)?;
            let s = logits.dim(1)?;
            let last = logits.i((0, s - 1, ..))?.to_dtype(DType::F32)?;
            let next = argmax_i64(&last)?;
            reference.push(next);
            all.push(next);
        }
        assert_eq!(cached, reference, "cached greedy diverged from full recompute");
        Ok(())
    }

    /// **Phase 2 acceptance #2** — the VarBuilder key tree we request
    /// matches the released `thinker.{model.*, lm_head.weight}` layout
    /// exactly. Catches:
    ///   - q/k/v biases on attention; NO o_proj.bias
    ///   - SwiGLU MLP with NO biases
    ///   - RMSNorm weight-only at the right places
    ///   - NO `q_norm`/`k_norm` keys
    ///   - explicit `lm_head.weight` (no embed tie)
    #[test]
    fn thinker_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _thinker = Thinker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let names: std::collections::BTreeSet<String> = vm
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        let mut want = std::collections::BTreeSet::<String>::new();
        want.insert("model.embed_tokens.weight".to_string());
        want.insert("model.norm.weight".to_string());
        want.insert("lm_head.weight".to_string());
        for i in 0..cfg.num_hidden_layers {
            for key in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.q_proj.bias",
                "self_attn.k_proj.weight",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.weight",
                "self_attn.v_proj.bias",
                "self_attn.o_proj.weight",
                // NO self_attn.o_proj.bias (linear_no_bias)
                // NO self_attn.q_norm / k_norm (Qwen3-only feature)
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
                // NO mlp.*_proj.bias (linear_no_bias)
            ] {
                want.insert(format!("model.layers.{i}.{key}"));
            }
        }

        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );
        // Specifically ensure no q_norm/k_norm got requested.
        assert!(!names.iter().any(|k| k.contains("q_norm") || k.contains("k_norm")));
        Ok(())
    }

    /// **Phase 2 acceptance #3** — real-weight Lambda smoke. Loads the
    /// released `thinker.{model.*, lm_head.weight}` from
    /// `Qwen/Qwen2.5-Omni-3B` shards and runs a tiny text-only forward.
    /// Asserts shape + that no key was missing. CPU+F32 for portability
    /// (matches the Phase 1 smoke convention).
    #[test]
    #[ignore]
    fn real_weight_thinker_loads() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");

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

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let mut thinker = Thinker::new(&cfg.thinker_config.text_config, vb.pp("thinker"))
            .expect("construct Thinker from real weights");

        // Tiny forward: a 4-token prompt.
        let input_ids = Tensor::from_vec(vec![151644i64, 151645, 100, 200], (1, 4), &device)
            .unwrap();
        let logits = thinker
            .forward_text_only(&input_ids, 0)
            .expect("forward on real weights");
        let (b, s, v) = (
            logits.dim(0).unwrap(),
            logits.dim(1).unwrap(),
            logits.dim(2).unwrap(),
        );
        eprintln!("real_weight_thinker_loads: logits = ({b}, {s}, {v})");
        assert_eq!(b, 1);
        assert_eq!(s, 4);
        assert_eq!(v, thinker.config().vocab_size);
    }

    /// **Phase 2 GPU acceptance** — same as the CPU smoke but on CUDA
    /// at native BF16.
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn real_weight_thinker_loads_cuda_bf16() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");

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
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::BF16, &device)
                .expect("mmap safetensors")
        };
        let mut thinker = Thinker::new(&cfg.thinker_config.text_config, vb.pp("thinker"))
            .expect("construct Thinker from real weights");

        let input_ids = Tensor::from_vec(vec![151644i64, 151645, 100, 200], (1, 4), &device)
            .unwrap();
        let logits = thinker
            .forward_text_only(&input_ids, 0)
            .expect("forward on real weights (cuda bf16)");
        let (b, s, v) = (
            logits.dim(0).unwrap(),
            logits.dim(1).unwrap(),
            logits.dim(2).unwrap(),
        );
        eprintln!("real_weight_thinker_loads_cuda_bf16: logits = ({b}, {s}, {v})");
        assert_eq!(b, 1);
        assert_eq!(s, 4);
        assert_eq!(v, thinker.config().vocab_size);
        assert_eq!(logits.dtype(), DType::BF16);
    }
}

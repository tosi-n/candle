//! Qwen2.5-Omni Talker — autoregressive speech-token (codec) generator.
//!
//! Mirrors `Qwen2_5OmniTalkerForConditionalGeneration` (upstream
//! `modeling_qwen2_5_omni.py:2218-2244` + forward 2255-2381). The Talker
//! is a smaller Qwen2 decoder whose lm_head is replaced by a `codec_head`,
//! producing **discrete codec tokens autoregressively** (single codebook,
//! NOT diffusion, NOT RVQ). Generation drives the upstream
//! `GenerationMixin.generate()` with these defaults (lines 3765-3771):
//! `do_sample=True, top_k=40, top_p=0.8, temperature=0.9,
//! repetition_penalty=1.05, max_new_tokens=4096`. EOS = `{8292, 8294}`,
//! suppress = `[8293]` (codec BOS).
//!
//! ## What's loaded vs. what's computed
//!
//! - `thinker_to_talker_proj`: `Linear(2048→896, bias=True)` — projects
//!   Thinker hidden states down to the Talker decoder's hidden size.
//! - `model.embed_tokens`: `Embedding(8448, 2048)` — embeddings live in
//!   the **Thinker** hidden space (2048), NOT the Talker's 896. The
//!   forward path: `codec_embed (2048) → thinker_to_talker_proj → 896
//!   → decoder → codec_head → logits (8448)`.
//! - 24 Qwen2 decoder layers (`hidden=896, heads=14, kv=2, head_dim=64`,
//!   `intermediate=4864`). Architecturally **identical** to the Thinker
//!   layers (q/k/v bias, no q_norm, no sliding window) — just smaller +
//!   M-RoPE section `[16, 16, 0]` (W-axis 0-width).
//! - `codec_head`: `Linear(896→8448, bias=False)`.
//!
//! ## Phase 3 scope
//!
//! Module construction + per-step forward producing codec-token logits.
//! KV-cached generation, sampling (top-k / top-p / suppress /
//! repetition_penalty), and the rolling `thinker_reply_part` buffer
//! that's added to per-step codec embeddings are all wired into the
//! decode loop in a Phase 3.5 follow-up. This PR lands the module so
//! Phase 6's end-to-end orchestration can compose against a real Talker.

use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, ops::softmax_last_dim, rms_norm, Activation, Embedding,
    Linear, RmsNorm, VarBuilder,
};

use super::config::TalkerConfig;
use super::mrope::MRopeTable;

const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;

/// Self-attention identical in shape contract to the Thinker's, just at
/// the Talker's smaller dims. Kept in this file (rather than reused
/// directly from thinker.rs) so the structs stay independent + the
/// load-bearing GQA/bias asymmetry is explicit at the call site.
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
    /// Per-layer KV cache for incremental AR decode: pre-`repeat_kv`
    /// post-RoPE (k, v), each `(B, n_kv_heads, L, head_dim)`. `None`
    /// between generations; `Talker::clear_kv_cache` resets it.
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(cfg: &TalkerConfig, rotary: std::sync::Arc<MRopeTable>, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
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

    fn forward(&mut self, x: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
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

        let (q, k) = self.rotary.apply(&q, &k, position_ids)?;

        // KV cache: append this chunk's (k, v) to the running cache so AR
        // decode steps only compute attention for the NEW positions.
        // `position_ids` must carry absolute positions for RoPE to match
        // the cached entries.
        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((pk, pv)) => (
                Tensor::cat(&[pk, &k], 2)?.contiguous()?,
                Tensor::cat(&[pv, &v], 2)?.contiguous()?,
            ),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let groups = self.n_heads / self.n_kv_heads;
        let k = if groups > 1 { repeat_kv(&k, groups)? } else { k };
        let v = if groups > 1 { repeat_kv(&v, groups)? } else { v };

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let scores_f32 = scores.to_dtype(DType::F32)?;
        let scores_masked = apply_causal_mask(&scores_f32)?;
        let weights = softmax_last_dim(&scores_masked)?.to_dtype(q.dtype())?;
        let attn = weights.matmul(&v)?;
        let attn = attn
            .transpose(1, 2)?
            .reshape((b, s, self.n_heads * self.head_dim))?;
        self.o_proj.forward(&attn)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

fn repeat_kv(x: &Tensor, groups: usize) -> Result<Tensor> {
    let (b, kv, s, d) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.broadcast_as((b, kv, groups, s, d))?;
    x.reshape((b, kv * groups, s, d))
}

/// Causal mask over `(B, H, q_len, kv_len)` scores. With a KV cache the
/// query rows sit at the END of the key axis: query row `i` may attend
/// keys `j <= i + (kv_len − q_len)`. Decode steps (`q_len == 1`) attend
/// everything, so the mask is skipped entirely.
fn apply_causal_mask(scores: &Tensor) -> Result<Tensor> {
    let q = scores.dim(D::Minus2)?;
    let kv = scores.dim(D::Minus1)?;
    if q == 1 {
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

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: Activation,
}

impl Mlp {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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

#[derive(Debug, Clone)]
struct DecoderLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
}

impl DecoderLayer {
    fn new(
        cfg: &TalkerConfig,
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

/// Qwen2.5-Omni Talker.
///
/// Construct from a VarBuilder pre-scoped at `talker.` (so
/// `vb.pp("model")` resolves to `talker.model.*` and `vb.pp("codec_head")`
/// to `talker.codec_head.*`).
#[derive(Debug, Clone)]
pub struct Talker {
    thinker_to_talker_proj: Linear,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codec_head: Linear,
    cfg: TalkerConfig,
    rotary: std::sync::Arc<MRopeTable>,
}

impl Talker {
    /// Build from config + a VarBuilder pre-scoped to `talker.`.
    pub fn new(cfg: &TalkerConfig, vb_root: VarBuilder) -> Result<Self> {
        let device = vb_root.device().clone();
        let dtype = vb_root.dtype();

        let rotary = std::sync::Arc::new(MRopeTable::new(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_position_embeddings,
            &cfg.rope_scaling.mrope_section,
            &device,
            dtype,
        )?);

        // `thinker_to_talker_proj`: bias=True (default `nn.Linear`).
        let thinker_to_talker_proj = linear(
            cfg.embedding_size,
            cfg.hidden_size,
            vb_root.pp("thinker_to_talker_proj"),
        )?;

        let vb_model = vb_root.pp("model");
        // CRITICAL: `embed_tokens` width is the Thinker hidden size
        // (`embedding_size=2048`), NOT the Talker's `hidden_size=896`.
        // The codec_embed lookup happens in Thinker-space, then is
        // projected down to Talker-space.
        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.embedding_size,
            vb_model.pp("embed_tokens"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                vb_model.pp(format!("layers.{i}")),
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, DEFAULT_RMS_NORM_EPS, vb_model.pp("norm"))?;
        let codec_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_root.pp("codec_head"))?;

        Ok(Self {
            thinker_to_talker_proj,
            embed_tokens,
            layers,
            norm,
            codec_head,
            cfg: cfg.clone(),
            rotary,
        })
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.cfg
    }

    /// Project Thinker hidden states `(B, S, embedding_size=2048)` down
    /// to the Talker decoder's input dim `(B, S, hidden_size=896)`.
    /// Exposed so the orchestrator can build the `talker_inputs_embeds`
    /// using the upstream fusion (Phase 6 spec §6).
    pub fn project_thinker(&self, hidden: &Tensor) -> Result<Tensor> {
        self.thinker_to_talker_proj.forward(hidden)
    }

    pub fn embed_codec(&self, codec_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(codec_ids)
    }

    /// Run the Talker decoder over pre-projected `inputs_embeds:
    /// (B, S, hidden_size=896)`, returning codec-token logits
    /// `(B, S, vocab_size=8448)`. Caller is responsible for the
    /// per-step `codec_embed + thinker_reply_part[:, :1]` rolling-buffer
    /// addition that the upstream forward (line 2348) does (see Phase 3
    /// spec for the exact fusion).
    ///
    /// **Critical fp32 cast on logits** (upstream line 2369): top-k /
    /// top-p / multinomial sampling in BF16 silently mis-samples around
    /// the threshold. The caller must sample in F32.
    ///
    /// **KV-cached**: each call appends this chunk's keys/values to the
    /// per-layer cache, so AR decode passes ONE new token per step with
    /// its absolute position id. Call [`Self::clear_kv_cache`] before a
    /// new sequence (prefill).
    pub fn forward_from_projected_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let mut h = inputs_embeds.clone();
        for layer in &mut self.layers {
            h = layer.forward(&h, position_ids)?;
        }
        let h = self.norm.forward(&h)?;
        // codec_head + force f32 for downstream sampling.
        self.codec_head.forward(&h)?.to_dtype(DType::F32)
    }

    /// Reset all per-layer KV caches. MUST be called before every new
    /// generation — stale entries poison attention for the next sequence.
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::{RopeScaling, TalkerConfig};
    use crate::models::qwen2_5_omni::mrope::text_only_position_ids;
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// Tiny Talker config. Talker's `mrope_section` is `[16, 16, 0]` →
    /// head_dim/2 = 32, so head_dim = 64. We mirror the structure at
    /// tiny dims: head_dim=8 → mrope_section=[2, 2, 0] (sum=4=8/2).
    fn tiny_cfg() -> TalkerConfig {
        TalkerConfig {
            hidden_size: 32,
            embedding_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 48,
            vocab_size: 24, // codec head width
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 32,
            rope_scaling: RopeScaling {
                mrope_section: vec![2, 2, 0], // sum 4 = head_dim/2 = 8/2 ✓
            },
            position_id_per_seconds: 25,
            seconds_per_chunk: 2,
            tts_codec_start_token_id: 19,
            tts_codec_end_token_id: 20,
            tts_codec_pad_token_id: 18,
            tts_codec_mask_token_id: 22,
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.02, v.as_tensor().shape().clone(), device)?)?;
        }
        Ok(())
    }

    /// **Phase 3 acceptance #1** — projection + decoder + codec_head
    /// produce the expected `(B, S, vocab_size)` codec logits and the
    /// output is forced fp32 for safe sampling.
    #[test]
    fn talker_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let mut talker = Talker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        // Simulate the upstream fusion: a (1, 5, embedding_size=64)
        // Thinker hidden, projected down to (1, 5, hidden_size=32) for
        // the decoder.
        let thinker_hidden = Tensor::randn(0f32, 1f32, (1usize, 5, cfg.embedding_size), &device)?;
        let projected = talker.project_thinker(&thinker_hidden)?;
        assert_eq!(projected.dims(), &[1, 5, cfg.hidden_size]);

        let pids = text_only_position_ids(1, 5, 0, &device)?;
        let logits = talker.forward_from_projected_embeds(&projected, &pids)?;
        assert_eq!(logits.dims(), &[1, 5, cfg.vocab_size]);
        // Sampling correctness gate (upstream line 2369).
        assert_eq!(logits.dtype(), DType::F32);
        Ok(())
    }

    /// **Phase 3 acceptance #2** — the VarBuilder key tree matches the
    /// released `talker.*` checkpoint layout exactly. Catches:
    ///   - `embed_tokens` width is `embedding_size` (2048), NOT `hidden`
    ///   - `thinker_to_talker_proj` bias=True
    ///   - `codec_head` bias=False
    ///   - q/k/v bias on attention; no o_proj.bias
    ///   - MLP all-no-bias
    #[test]
    fn talker_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _talker = Talker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let names: std::collections::BTreeSet<String> = vm
            .data()
            .lock()
            .unwrap()
            .keys()
            .cloned()
            .collect();

        let mut want = std::collections::BTreeSet::<String>::new();
        want.insert("thinker_to_talker_proj.weight".to_string());
        want.insert("thinker_to_talker_proj.bias".to_string());
        want.insert("codec_head.weight".to_string()); // NO bias
        want.insert("model.embed_tokens.weight".to_string());
        want.insert("model.norm.weight".to_string());
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
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                want.insert(format!("model.layers.{i}.{key}"));
            }
        }

        // Sanity: ensure no codec_head.bias / no thinker.lm_head /
        // no q_norm/k_norm got requested.
        for forbidden in ["codec_head.bias", "thinker.lm_head", "q_norm", "k_norm"] {
            assert!(
                !names.iter().any(|n| n.contains(forbidden)),
                "Talker requested forbidden key containing {forbidden:?}"
            );
        }

        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );
        Ok(())
    }

    /// **Phase 3 acceptance #3** — embed_tokens width is `embedding_size`
    /// (= Thinker hidden), NOT `hidden_size`. The codec embedding shape
    /// must round-trip through the projection.
    #[test]
    fn talker_embed_tokens_in_thinker_space() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let talker = Talker::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let ids = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
        let codec_embed = talker.embed_codec(&ids)?;
        // Embedding width must be `embedding_size`, not `hidden_size`.
        assert_eq!(codec_embed.dims(), &[1, 3, cfg.embedding_size]);
        let projected = talker.project_thinker(&codec_embed)?;
        assert_eq!(projected.dims(), &[1, 3, cfg.hidden_size]);
        Ok(())
    }

    /// **Phase 3 acceptance #4** — real-weight Lambda smoke (CPU+F32).
    /// Loads `talker.*` from the released 3B and runs one forward.
    #[test]
    #[ignore]
    fn real_weight_talker_loads() {
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
        let mut talker = Talker::new(&cfg.talker_config, vb.pp("talker"))
            .expect("construct Talker from real weights");

        // Tiny forward: project a (1, 4, 2048) Thinker-like hidden + run
        // one decoder pass. Outputs (1, 4, 8448).
        let cfg = talker.config().clone();
        let h = Tensor::randn(0f32, 1f32, (1, 4, cfg.embedding_size), &device).unwrap();
        let projected = talker.project_thinker(&h).expect("projection");
        let pids = text_only_position_ids(1, 4, 0, &device).expect("pids");
        let logits = talker
            .forward_from_projected_embeds(&projected, &pids)
            .expect("forward");
        eprintln!(
            "real_weight_talker_loads: logits = ({}, {}, {})",
            logits.dim(0).unwrap(),
            logits.dim(1).unwrap(),
            logits.dim(2).unwrap()
        );
        assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size); // 8448
    }

    /// **Phase 3 GPU acceptance** — same as CPU smoke but native BF16.
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn real_weight_talker_loads_cuda_bf16() {
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

        let device = Device::new_cuda(0).expect("cuda device 0");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::BF16, &device)
                .expect("mmap safetensors")
        };
        let mut talker = Talker::new(&cfg.talker_config, vb.pp("talker"))
            .expect("construct Talker from real weights");
        let cfg = talker.config().clone();
        let h = Tensor::randn(0f32, 1f32, (1, 4, cfg.embedding_size), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let projected = talker.project_thinker(&h).expect("projection");
        let pids = text_only_position_ids(1, 4, 0, &device).expect("pids");
        let logits = talker
            .forward_from_projected_embeds(&projected, &pids)
            .expect("forward");
        assert_eq!(logits.dim(2).unwrap(), cfg.vocab_size);
        // codec_head forces F32 (upstream line 2369).
        assert_eq!(logits.dtype(), DType::F32);
    }
}

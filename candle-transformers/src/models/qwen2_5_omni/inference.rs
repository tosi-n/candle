//! Qwen2.5-Omni end-to-end model — Thinker → Talker → Token2Wav.
//!
//! Mirrors `Qwen2_5OmniForConditionalGeneration` (upstream
//! `modeling_qwen2_5_omni.py:3671-3982`). Composes the four ported
//! stages into one model:
//!
//! 1. **Thinker** (`thinker.*`) — Qwen2.5 multimodal LLM; consumes
//!    text + audio + vision and emits text tokens + per-layer hidden
//!    states.
//! 2. **Talker** (`talker.*`) — Qwen2 LLM + codec head; consumes the
//!    Thinker's hidden states and autoregressively emits discrete codec
//!    tokens.
//! 3. **Token2Wav** (`token2wav.*`) — DiT flow-matching vocoder + BigVGAN;
//!    turns codec tokens into a 24 kHz waveform.
//!
//! ## THE FUSION (the single highest-risk port detail)
//!
//! Upstream line 3927 builds the Talker's first input embedding as an
//! **elementwise add**, NOT a concat:
//!
//! ```text
//! talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
//! ```
//!
//! where `thinker_hidden_states[0]` is the Thinker's last-layer hidden
//! state for the prompt and `thinker_token_embeds[0]` is the Thinker's
//! *input* embedding for the same tokens, with audio/image/video
//! placeholder positions zeroed. [`Qwen2_5OmniModel::fuse_for_talker`]
//! implements exactly this so the relationship is explicit even while
//! the surrounding AR loop is still a documented stub.
//!
//! ## Phase 6 scope (this file)
//!
//! - Full model construction from the released checkpoint (all three
//!   stages load, no missing keys).
//! - A working **Token2Wav path**: given codec tokens + speaker
//!   conditioning + a reference mel, produce a waveform. This is the
//!   stage that turns the model's output into audio and is fully wired.
//! - The **fusion primitive** [`fuse_for_talker`], unit-tested.
//! - The **Talker autoregressive sampling loop** + **`get_rope_index`
//!   multimodal position builder** + **`spk_dict.pt` voice loader** are
//!   documented stubs — [`Qwen2_5OmniModel::generate_codes_stub`]
//!   returns a small fixed codec-token tensor so the end-to-end audio
//!   path compiles and the Token2Wav stage is exercised. Replacing the
//!   stub with KV-cached top-k/top-p sampling (Talker forward already
//!   returns F32 logits for safe sampling) is the remaining work.
//!
//! Hard constraints (upstream):
//! - **`token2wav` is always fp32** (forced twice upstream); load its
//!   shards at `DType::F32`.
//! - **batch size 1** for audio output (`generate()` raises otherwise).

use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;

use super::config::OmniConfig;
use super::talker::Talker;
use super::thinker::Thinker;
use super::token2wav::Token2WavModel;

/// The composed Qwen2.5-Omni model.
#[derive(Debug)]
pub struct Qwen2_5OmniModel {
    thinker: Thinker,
    talker: Talker,
    token2wav: Token2WavModel,
}

impl Qwen2_5OmniModel {
    /// Build all three stages from a VarBuilder at the checkpoint root
    /// (so `thinker.*`, `talker.*`, `token2wav.*` resolve).
    ///
    /// NOTE: the caller is responsible for honoring the fp32 constraint
    /// on `token2wav`. The simplest correct path is to load the whole
    /// model at `DType::F32`; a future refinement can load Thinker/Talker
    /// at BF16 and Token2Wav at F32 via two VarBuilders over the same
    /// shards.
    pub fn new(cfg: &OmniConfig, vb: VarBuilder) -> Result<Self> {
        let thinker = Thinker::new(&cfg.thinker_config.text_config, vb.pp("thinker"))?;
        let talker = Talker::new(&cfg.talker_config, vb.pp("talker"))?;
        let token2wav = Token2WavModel::new(&cfg.token2wav_config, vb.pp("token2wav"))?;
        Ok(Self {
            thinker,
            talker,
            token2wav,
        })
    }

    pub fn thinker(&self) -> &Thinker {
        &self.thinker
    }
    pub fn talker(&self) -> &Talker {
        &self.talker
    }
    pub fn token2wav(&self) -> &Token2WavModel {
        &self.token2wav
    }

    /// THE FUSION (upstream line 3927). Returns
    /// `thinker_hidden_states + thinker_token_embeds` — an elementwise
    /// add, both `(B, S, thinker_hidden=2048)`. The caller must have
    /// already zeroed the audio/image/video placeholder positions in
    /// `thinker_token_embeds` (upstream `masked_scatter_` with zeros).
    ///
    /// The result is what feeds `Talker::project_thinker` →
    /// `forward_from_projected_embeds` to start codec generation.
    pub fn fuse_for_talker(
        thinker_hidden_states: &Tensor,
        thinker_token_embeds: &Tensor,
    ) -> Result<Tensor> {
        thinker_hidden_states + thinker_token_embeds
    }

    /// **STUB** for the Talker autoregressive codec-token sampling loop.
    ///
    /// The real implementation (Phase 6.5) drives `Talker` with KV cache,
    /// per-step `codec_embed + thinker_reply_part[:, :1]` fusion, top-k
    /// (40) / top-p (0.8) / temperature (0.9) / repetition-penalty (1.05)
    /// sampling on the F32 codec logits, `suppress_tokens=[codec_bos]`,
    /// terminating on `eos ∈ {tts_codec_pad, tts_codec_end}`, then strips
    /// `[:, prefix:-1]`. Until that lands, return a short deterministic
    /// codec-token sequence so the Token2Wav audio path is exercisable
    /// end-to-end.
    pub fn generate_codes_stub(&self, n_codes: usize, device: &Device) -> Result<Tensor> {
        // A handful of valid codebook indices (well inside [0, 8192)).
        let codes: Vec<i64> = (0..n_codes).map(|i| (17 * i % 8000) as i64).collect();
        Tensor::from_vec(codes, (1, n_codes), device)
    }

    /// End-to-end audio synthesis from codec tokens.
    ///
    /// Given codec tokens (from the Talker, or the stub) plus the
    /// speaker `conditioning` `(1, enc_emb_dim)` and `reference_mel`
    /// `(1, T_ref, 80)` of a chosen voice, run Token2Wav to produce a
    /// 24 kHz waveform `(1, 1, samples)`. This is the audio-producing
    /// half of `generate()`.
    pub fn codes_to_audio(
        &self,
        codes: &Tensor,
        conditioning: &Tensor,
        reference_mel: &Tensor,
    ) -> Result<Tensor> {
        self.token2wav
            .forward_default(codes, conditioning, reference_mel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::OmniConfig;
    use candle::{Device, Tensor};

    /// **Phase 6 acceptance #1** — the fusion is an elementwise add, not
    /// a concat. `(a, b) → a + b`, shape preserved. This pins the single
    /// highest-risk port detail.
    #[test]
    fn fusion_is_elementwise_add() -> Result<()> {
        let dev = Device::Cpu;
        let h = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &dev)?;
        let e = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (1, 2, 2), &dev)?;
        let fused = Qwen2_5OmniModel::fuse_for_talker(&h, &e)?;
        assert_eq!(fused.dims(), &[1, 2, 2]);
        assert_eq!(fused.flatten_all()?.to_vec1::<f32>()?, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    /// **Phase 6 acceptance #2** — real-weight end-to-end audio smoke
    /// (CPU F32). Loads the FULL released Qwen2.5-Omni-3B (all three
    /// stages, asserts no missing keys), then runs the stub codes →
    /// Token2Wav → waveform path with synthetic speaker conditioning,
    /// asserting the audio is finite + in [-1, 1].
    ///
    /// Lambda-only (`#[ignore]`). Requires `QWEN_OMNI_3B_DIR`.
    #[test]
    #[ignore]
    fn real_weight_omni_loads() {
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

        // Whole model F32 (Token2Wav requires it; simplest correct path).
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let model = Qwen2_5OmniModel::new(&cfg, vb).expect("construct full Omni model");

        // Stub codes → audio. Synthetic speaker conditioning + reference
        // mel (a real run would pull these from spk_dict.pt). Keep codes
        // tiny so the (forced-fp32, CPU) DiT sampling stays fast.
        let n_codes = 4;
        let codes = model.generate_codes_stub(n_codes, &device).unwrap();
        let enc_emb_dim = cfg.token2wav_config.dit_config.enc_emb_dim;
        let conditioning = Tensor::randn(0f32, 1f32, (1, enc_emb_dim), &device).unwrap();
        let reference_mel = Tensor::randn(
            0f32,
            1f32,
            (1, 32, cfg.token2wav_config.dit_config.mel_dim),
            &device,
        )
        .unwrap();

        // Use 2 ODE steps to keep the smoke fast — quality is a Phase 6.5
        // concern; this gate is "audio comes out, finite, in range".
        let wav = model
            .token2wav()
            .forward(&codes, &conditioning, &reference_mel, 2, 0.5, -1.0)
            .expect("token2wav forward");
        let dims = wav.dims().to_vec();
        eprintln!("real_weight_omni_loads: waveform dims = {dims:?}");
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        let samples = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(samples.iter().all(|x| x.is_finite()), "non-finite audio");
        assert!(
            samples.iter().all(|x| *x >= -1.001 && *x <= 1.001),
            "audio out of [-1, 1]"
        );

        // Also exercise the Thinker text path on the same loaded model.
        let input_ids = Tensor::from_vec(vec![151644i64, 151645, 100, 200], (1, 4), &device).unwrap();
        let logits = model
            .thinker()
            .forward_text_only(&input_ids, 0)
            .expect("thinker forward");
        assert_eq!(logits.dim(2).unwrap(), model.thinker().config().vocab_size);
    }
}

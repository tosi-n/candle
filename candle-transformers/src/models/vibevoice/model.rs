//! Top-level VibeVoice model: wires the diffusion head, the two
//! audio tokenizers, the speech connectors, and the
//! σ-VAE scaling buffers into a single bundle that can be driven by
//! [`crate::models::vibevoice::inference::generate_audio`].
//!
//! Pure-candle port of `vibevoice/modular/modeling_vibevoice.py`. The
//! Qwen2 LLM backbone is **not** owned by this struct — callers
//! construct it via [`crate::models::qwen2::Model`] and pass it as a
//! mutable reference to `generate_audio`. Keeping the LLM out of the
//! struct lets the same VibeVoice port slot into different LLM
//! variants (1.5B / 7B) and decouples the KV-cache lifecycle from the
//! audio side of the model.

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::{
    VibeVoiceAcousticTokenizerConfig, VibeVoiceConfig, VibeVoiceSemanticTokenizerConfig,
    VibeVoiceTokenizerConfig,
};
use super::connectors::SpeechConnector;
use super::diffusion_head::VibeVoiceDiffusionHead;
use super::dpm_solver::{DpmSolver, DpmSolverConfig};
use super::tokenizer::{AcousticTokenizer, SemanticTokenizer};

pub use super::inference::{generate_audio, GenerateOutput};

/// Inference-time options passed per `generate` call.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    /// Cap the synthesized audio at this many seconds. Used when
    /// `max_new_tokens` is `None` to derive a token budget.
    pub max_audio_seconds: f64,
    /// Number of DPM-Solver inference steps. VibeVoice-1.5B ships 20.
    pub diffusion_steps: usize,
    /// Classifier-free-guidance scale. Reasonable default: 1.3.
    pub cfg_scale: f64,
    /// Optional reference audio for speaker conditioning. Shape
    /// `(1, channels, T_samples)` at the model's `sample_rate`.
    pub reference_audio: Option<Tensor>,
    /// Hard cap on the number of acoustic latents produced. When `None`
    /// we derive the cap from `max_audio_seconds` and the tokenizer's
    /// downsample factor.
    pub max_new_tokens: Option<usize>,
    /// Output sample rate. Used only when deriving `max_new_tokens`
    /// from `max_audio_seconds`. VibeVoice's released checkpoint runs
    /// at 24 kHz.
    pub sample_rate: usize,
    /// Optional vocab token id that signals end-of-audio. When set
    /// together with a Qwen2 lm_head, generate exits early on detection.
    /// Upstream's streaming variant uses a separate binary classifier
    /// and ignores this field — see [`crate::models::vibevoice::inference`]
    /// module docs for the divergence.
    pub eos_audio_token_id: Option<u32>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_audio_seconds: 30.0,
            diffusion_steps: 20,
            cfg_scale: 1.3,
            reference_audio: None,
            max_new_tokens: None,
            sample_rate: 24_000,
            eos_audio_token_id: None,
        }
    }
}

/// Load a 0-d / 1-d scalar buffer (e.g. `speech_scaling_factor`) from
/// the VarBuilder, normalising the result to a rank-1 single-element
/// tensor. Falls back to `default_value` when the buffer is missing —
/// useful for synthetic-init test paths that build the model from a
/// VarMap rather than safetensors.
///
/// We try the rank-0 layout first (the released checkpoint's actual
/// shape) and fall back to rank-1 to accommodate forks that store the
/// buffer as `torch.tensor([0.0])`. Both backends accept `contains_tensor`
/// for presence checks before the typed `get`.
fn load_scalar_buffer(vb: &VarBuilder, name: &str, default_value: f32) -> Result<Tensor> {
    if !vb.contains_tensor(name) {
        return Tensor::new(&[default_value], vb.device())?.to_dtype(vb.dtype());
    }
    let raw = vb
        .get((), name)
        .or_else(|_| vb.get(1, name))
        .or_else(|_| vb.get((1,), name))?;
    raw.to_dtype(vb.dtype())?.flatten_all()?.reshape(1)
}

/// Owned VibeVoice model. Holds everything except the Qwen2 LLM backbone.
#[derive(Debug, Clone)]
pub struct VibeVoiceModel {
    config: VibeVoiceConfig,
    /// Hidden size of the LLM the connectors were sized against.
    /// Required for the unconditional-condition zero vector and for
    /// validating that the caller's [`crate::models::qwen2::Model`]
    /// was built with matching dimensions.
    lm_hidden_size: usize,
    diffusion_head: VibeVoiceDiffusionHead,
    acoustic_tokenizer: AcousticTokenizer,
    semantic_tokenizer: SemanticTokenizer,
    acoustic_connector: SpeechConnector,
    semantic_connector: SpeechConnector,
    /// `speech_scaling_factor` and `speech_bias_factor` are 0-d scalar
    /// buffers in the released safetensors. We store them as 1-element
    /// rank-1 tensors so they broadcast cleanly against `(B, T, vae_dim)`
    /// latents in the inference pipeline.
    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,
}

impl VibeVoiceModel {
    /// Construct from a [`VarBuilder`] pointing at the `model.*` prefix
    /// of the safetensors. Wires the diffusion head, the two
    /// tokenizers, the two connectors, and the σ-VAE scaling buffers.
    ///
    /// `lm_hidden_size` must match the hidden size of the LLM the
    /// caller plans to drive `generate` with — the connectors project
    /// into that space and the unconditional condition vector is built
    /// at that width.
    pub fn new(config: VibeVoiceConfig, lm_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let diffusion_head =
            VibeVoiceDiffusionHead::new(&config.diffusion_head_config, vb.pp("prediction_head"))?;
        let acoustic_cfg: VibeVoiceTokenizerConfig = (&config.acoustic_tokenizer_config).into();
        let acoustic_tokenizer = AcousticTokenizer::new(acoustic_cfg, vb.pp("acoustic_tokenizer"))?;
        let semantic_cfg: VibeVoiceTokenizerConfig = (&config.semantic_tokenizer_config).into();
        let semantic_tokenizer = SemanticTokenizer::new(semantic_cfg, vb.pp("semantic_tokenizer"))?;

        let acoustic_vae_dim = config.acoustic_tokenizer_config.vae_dim;
        let semantic_vae_dim = config.semantic_tokenizer_config.vae_dim;
        let acoustic_connector = SpeechConnector::new(
            acoustic_vae_dim,
            lm_hidden_size,
            vb.pp("acoustic_connector"),
        )?;
        let semantic_connector = SpeechConnector::new(
            semantic_vae_dim,
            lm_hidden_size,
            vb.pp("semantic_connector"),
        )?;

        // The released safetensors store the buffers as 0-d scalars
        // (`register_buffer('speech_scaling_factor', torch.tensor(NaN))`
        // at model.py L132). We pull them via `get_unchecked` to accept
        // any rank — 0-d, 1-d, or otherwise — then reshape to a single
        // 1-d element so `broadcast_div` / `broadcast_sub` work against
        // `(B, T, vae_dim)` latents. When no checkpoint is available
        // (the smoke-test path through a VarMap) we fall back to the
        // identity transform `(scaling=1, bias=0)`.
        let scaling = load_scalar_buffer(&vb, "speech_scaling_factor", 1.0_f32)?;
        let bias = load_scalar_buffer(&vb, "speech_bias_factor", 0.0_f32)?;

        Ok(Self {
            config,
            lm_hidden_size,
            diffusion_head,
            acoustic_tokenizer,
            semantic_tokenizer,
            acoustic_connector,
            semantic_connector,
            speech_scaling_factor: scaling,
            speech_bias_factor: bias,
        })
    }

    pub fn config(&self) -> &VibeVoiceConfig {
        &self.config
    }

    pub fn acoustic_config(&self) -> &VibeVoiceAcousticTokenizerConfig {
        &self.config.acoustic_tokenizer_config
    }

    pub fn semantic_config(&self) -> &VibeVoiceSemanticTokenizerConfig {
        &self.config.semantic_tokenizer_config
    }

    pub fn lm_hidden_size(&self) -> usize {
        self.lm_hidden_size
    }

    pub fn diffusion_head(&self) -> &VibeVoiceDiffusionHead {
        &self.diffusion_head
    }

    pub fn acoustic_tokenizer(&self) -> &AcousticTokenizer {
        &self.acoustic_tokenizer
    }

    pub fn semantic_tokenizer(&self) -> &SemanticTokenizer {
        &self.semantic_tokenizer
    }

    pub fn acoustic_connector(&self) -> &SpeechConnector {
        &self.acoustic_connector
    }

    pub fn semantic_connector(&self) -> &SpeechConnector {
        &self.semantic_connector
    }

    pub fn speech_scaling_factor(&self) -> &Tensor {
        &self.speech_scaling_factor
    }

    pub fn speech_bias_factor(&self) -> &Tensor {
        &self.speech_bias_factor
    }

    /// Total downsample factor of the acoustic tokenizer — equivalent
    /// to the product of `encoder_ratios`. For VibeVoice-1.5B this is
    /// `8 * 5 * 5 * 4 * 2 * 2 = 3200` samples per latent at 24 kHz.
    pub fn acoustic_downsample(&self) -> usize {
        self.config.acoustic_tokenizer_config.total_downsample()
    }

    /// Build a DPM-Solver pre-configured for VibeVoice's noise schedule.
    /// Useful for ahead-of-time inspection of the timestep schedule.
    pub fn make_solver(&self, num_inference_steps: usize) -> DpmSolver {
        DpmSolver::new(DpmSolverConfig::default(), num_inference_steps)
    }

    /// One-shot text-to-audio entry point that wraps
    /// [`generate_audio`] for callers that already own a Qwen2 model
    /// and just want a Tensor back.
    ///
    /// Provided as a convenience over the lower-level
    /// [`generate_audio`] free function — the latter is preferred when
    /// the caller wants to inspect the intermediate latents
    /// ([`GenerateOutput::latents`]) or wire in a custom EOS detector
    /// against the lm_head output.
    pub fn generate(
        &self,
        language_model: &mut crate::models::qwen2::Model,
        input_ids: &Tensor,
        options: &GenerateOptions,
    ) -> Result<Tensor> {
        let out = generate_audio(self, language_model, None, input_ids, None, options)?;
        Ok(out.audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    /// `load_scalar_buffer` accepts both 0-d and 1-d buffer layouts and
    /// always returns a rank-1 single-element tensor. This is the
    /// shape the released safetensors actually ships
    /// (`register_buffer(..., torch.tensor(NaN))` produces a 0-d).
    #[test]
    fn scalar_buffer_reshapes_to_rank_one() -> candle::Result<()> {
        use candle::DType;
        use candle_nn::{VarBuilder, VarMap};

        let device = Device::Cpu;
        let vm = VarMap::new();
        // Insert a 0-d tensor under the canonical name. VarMap normally
        // initialises tensors lazily on first `get`, but `data().insert`
        // lets us seed an arbitrary shape directly — including the rank-0
        // case the upstream checkpoint uses.
        vm.data().lock().unwrap().insert(
            "speech_scaling_factor".to_string(),
            candle::Var::from_tensor(&Tensor::new(2.5_f32, &device)?)?,
        );
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let scaling = load_scalar_buffer(&vb, "speech_scaling_factor", 1.0)?;
        assert_eq!(scaling.dims(), &[1]);
        let got = scaling.to_vec1::<f32>()?[0];
        assert!((got - 2.5).abs() < 1e-6, "expected 2.5, got {got}");

        // Missing buffer falls back to the default.
        let bias = load_scalar_buffer(&vb, "speech_bias_factor", -0.7)?;
        assert_eq!(bias.dims(), &[1]);
        assert!((bias.to_vec1::<f32>()?[0] - (-0.7)).abs() < 1e-6);
        Ok(())
    }
}

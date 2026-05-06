//! Top-level VibeVoice model: wires Qwen2.5 + diffusion head + tokenizers
//! + DPM-Solver into a single `generate` entry point.
//!
//! Pure-candle port of `vibevoice/modular/modeling_vibevoice.py`.
//!
//! Status: skeleton. The full inference loop lives in
//! `vibevoice/modular/modeling_vibevoice_streaming_inference.py` and
//! lands as a follow-up commit.

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::{
    VibeVoiceAcousticTokenizerConfig, VibeVoiceConfig, VibeVoiceSemanticTokenizerConfig,
    VibeVoiceTokenizerConfig,
};
use super::diffusion_head::VibeVoiceDiffusionHead;
use super::dpm_solver::{DpmSolver, DpmSolverConfig};
use super::tokenizer::{AcousticTokenizer, SemanticTokenizer};

/// Inference-time options passed per `generate` call.
#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_audio_seconds: f64,
    pub diffusion_steps: usize,
    pub cfg_scale: f64,
    /// Optional reference audio for speaker conditioning. Shape
    /// `(1, 1, T_samples)` at the model's `sample_rate`.
    pub reference_audio: Option<Tensor>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_audio_seconds: 30.0,
            diffusion_steps: 20,
            cfg_scale: 1.3,
            reference_audio: None,
        }
    }
}

/// Owned VibeVoice model.
#[derive(Debug, Clone)]
pub struct VibeVoiceModel {
    config: VibeVoiceConfig,
    pub diffusion_head: VibeVoiceDiffusionHead,
    pub acoustic_tokenizer: AcousticTokenizer,
    pub semantic_tokenizer: SemanticTokenizer,
    // Note: the LLM backbone is constructed by the caller via
    // [`crate::models::qwen2::Model`] and passed in at generate time.
    // Keeping that out of this struct lets the same VibeVoice port slot
    // into different LLM variants (1.5B / 7B) without changing the
    // model-side code path.
}

impl VibeVoiceModel {
    pub fn new(config: VibeVoiceConfig, vb: VarBuilder) -> Result<Self> {
        let diffusion_head =
            VibeVoiceDiffusionHead::new(&config.diffusion_head_config, vb.pp("diffusion_head"))?;
        let acoustic_cfg: VibeVoiceTokenizerConfig = (&config.acoustic_tokenizer_config).into();
        let acoustic_tokenizer = AcousticTokenizer::new(acoustic_cfg, vb.pp("acoustic_tokenizer"))?;
        let semantic_cfg: VibeVoiceTokenizerConfig = (&config.semantic_tokenizer_config).into();
        let semantic_tokenizer = SemanticTokenizer::new(semantic_cfg, vb.pp("semantic_tokenizer"))?;
        Ok(Self {
            config,
            diffusion_head,
            acoustic_tokenizer,
            semantic_tokenizer,
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

    /// Build a DPM-Solver pre-configured for VibeVoice's noise schedule.
    pub fn make_solver(&self, num_inference_steps: usize) -> DpmSolver {
        DpmSolver::new(DpmSolverConfig::default(), num_inference_steps)
    }

    /// Synthesize speech from a prompt. Skeleton — full inference loop
    /// arrives with the modeling_vibevoice spec.
    pub fn generate(&self, _text: &str, _options: &GenerateOptions) -> Result<Tensor> {
        candle::bail!(
            "VibeVoiceModel::generate skeleton; full inference loop arrives in a follow-up commit"
        )
    }
}

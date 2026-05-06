//! Acoustic and semantic audio tokenizers for VibeVoice.
//!
//! Pure-candle port of `vibevoice/modular/modular_vibevoice_tokenizer.py`.
//! σ-VAE-style continuous tokenizer operating at 7.5 Hz against a 24 kHz
//! waveform (3200× downsample).
//!
//! Status: skeleton. Encoder / decoder bodies fill in once the
//! discovery subagent returns its port spec for the upstream file.

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::config::VibeVoiceTokenizerConfig;

/// σ-VAE acoustic tokenizer: encodes 24 kHz waveform → 7.5 Hz continuous
/// tokens; decodes back to 24 kHz.
#[derive(Debug, Clone)]
pub struct AcousticTokenizer {
    config: VibeVoiceTokenizerConfig,
}

impl AcousticTokenizer {
    pub fn new(config: VibeVoiceTokenizerConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self { config })
    }

    /// Encode raw 24 kHz audio `(B, 1, T_samples)` → continuous tokens
    /// `(B, T_tokens, latent_size)`.
    pub fn encode(&self, _audio: &Tensor) -> Result<Tensor> {
        candle::bail!("AcousticTokenizer::encode skeleton; see modular_vibevoice_tokenizer.py")
    }

    /// Decode continuous tokens `(B, T_tokens, latent_size)` → 24 kHz
    /// audio `(B, 1, T_samples)`.
    pub fn decode(&self, _tokens: &Tensor) -> Result<Tensor> {
        candle::bail!("AcousticTokenizer::decode skeleton; see modular_vibevoice_tokenizer.py")
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }
}

/// Semantic tokenizer — encoder-only at inference. Produces the speaker
/// conditioning embedding used to bias decoding for voice cloning /
/// multi-speaker dialogue.
#[derive(Debug, Clone)]
pub struct SemanticTokenizer {
    config: VibeVoiceTokenizerConfig,
}

impl SemanticTokenizer {
    pub fn new(config: VibeVoiceTokenizerConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn encode(&self, _audio: &Tensor) -> Result<Tensor> {
        candle::bail!("SemanticTokenizer::encode skeleton; see modular_vibevoice_tokenizer.py")
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }
}

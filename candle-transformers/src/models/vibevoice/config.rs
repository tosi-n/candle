//! Configuration structs for [`crate::models::vibevoice`].
//!
//! Mirrors the Python dataclasses in
//! `vibevoice/modular/configuration_vibevoice.py` and the JSON files
//! under `vibevoice/configs/` (canonical example: `qwen2.5_1.5b_64k.json`).
//!
//! Inference-only port — training-only fields (`ddpm_batch_mul`,
//! tensor-parallel plans, `_attn_implementation_autoset`) are dropped.
//! ASR-variant config is also dropped; this module is TTS-only.

use serde::{Deserialize, Serialize};

/// Top-level VibeVoice config. Wraps the LLM backbone config plus the
/// diffusion head and tokenizer configs.
///
/// Matches `VibeVoiceConfig` at line 182 of `configuration_vibevoice.py`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VibeVoiceConfig {
    pub acoustic_tokenizer_config: VibeVoiceAcousticTokenizerConfig,
    pub semantic_tokenizer_config: VibeVoiceSemanticTokenizerConfig,
    /// Decoder LM. Plug into [`crate::models::qwen2`] downstream.
    pub decoder_config: serde_json::Value,
    pub diffusion_head_config: VibeVoiceDiffusionHeadConfig,
    /// Required to equal `acoustic_tokenizer_config.vae_dim`. The JSON
    /// tracks both for convenience; we validate equality on load.
    #[serde(default)]
    pub acoustic_vae_dim: Option<usize>,
    /// Required to equal `semantic_tokenizer_config.vae_dim`.
    #[serde(default)]
    pub semantic_vae_dim: Option<usize>,
    /// `"bfloat16"`, `"float16"`, or `"float32"`. The Rust port maps
    /// these to candle dtype at load time.
    #[serde(default = "default_torch_dtype")]
    pub torch_dtype: String,
    /// Permits both `"vibepod"` (legacy in checkpoint JSON) and
    /// `"vibevoice"` (Python class default).
    #[serde(default = "default_model_type")]
    pub model_type: String,
}

fn default_torch_dtype() -> String {
    "float32".to_string()
}
fn default_model_type() -> String {
    "vibevoice".to_string()
}

impl VibeVoiceConfig {
    /// Cheap sanity check that the cached `*_vae_dim` fields agree with
    /// the nested tokenizer configs. Returns `Err` on mismatch.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(d) = self.acoustic_vae_dim {
            if d != self.acoustic_tokenizer_config.vae_dim {
                return Err(format!(
                    "acoustic_vae_dim={d} but acoustic_tokenizer_config.vae_dim={}",
                    self.acoustic_tokenizer_config.vae_dim
                ));
            }
        }
        if let Some(d) = self.semantic_vae_dim {
            if d != self.semantic_tokenizer_config.vae_dim {
                return Err(format!(
                    "semantic_vae_dim={d} but semantic_tokenizer_config.vae_dim={}",
                    self.semantic_tokenizer_config.vae_dim
                ));
            }
        }
        Ok(())
    }
}

/// Acoustic tokenizer config — `VibeVoiceAcousticTokenizerConfig`
/// at lines 31–91 of `configuration_vibevoice.py`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VibeVoiceAcousticTokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub corpus_normalize: f32,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_acoustic_vae_dim")]
    pub vae_dim: usize,
    #[serde(default = "default_acoustic_fix_std")]
    pub fix_std: f32,
    #[serde(default = "default_std_dist_type_gaussian")]
    pub std_dist_type: String,
    #[serde(default = "default_mixer_layer")]
    pub mixer_layer: String,
    #[serde(default = "default_conv_norm")]
    pub conv_norm: String,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_layernorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale_init_value")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_weight_init_value")]
    pub weight_init_value: f64,
    #[serde(default = "default_n_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_encoder_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,
    #[serde(default = "default_n_filters")]
    pub decoder_n_filters: usize,
    /// `None` ⇒ mirrors `encoder_ratios` (Python L89). Use
    /// [`Self::effective_decoder_ratios`].
    #[serde(default)]
    pub decoder_ratios: Option<Vec<usize>>,
    #[serde(default)]
    pub decoder_depths: Option<String>,
    #[serde(default)]
    pub model_type: String,
}

impl VibeVoiceAcousticTokenizerConfig {
    /// If `decoder_ratios` is `None`, the Python class clones
    /// `encoder_ratios`. We surface a borrow that mirrors that.
    pub fn effective_decoder_ratios(&self) -> &[usize] {
        self.decoder_ratios
            .as_deref()
            .unwrap_or(&self.encoder_ratios)
    }

    /// Total downsample factor through the encoder
    /// (= total upsample factor through the decoder). For the 1.5B
    /// variant this is `2·2·4·5·5·8 = 1600` per the encoder ratios — but
    /// the **first** stage is a stride-1 stem conv, and the time domain
    /// follows `1·prod(ratios)`. Verify against checkpoint at runtime.
    pub fn total_downsample(&self) -> usize {
        self.encoder_ratios.iter().product()
    }
}

fn default_channels() -> usize {
    1
}
fn default_true() -> bool {
    true
}
fn default_acoustic_vae_dim() -> usize {
    64
}
fn default_acoustic_fix_std() -> f32 {
    0.5
}
fn default_std_dist_type_gaussian() -> String {
    "gaussian".to_string()
}
fn default_mixer_layer() -> String {
    "depthwise_conv".to_string()
}
fn default_conv_norm() -> String {
    "none".to_string()
}
fn default_pad_mode() -> String {
    "constant".to_string()
}
fn default_layernorm() -> String {
    "RMSNorm".to_string()
}
fn default_layernorm_eps() -> f64 {
    1e-5
}
fn default_layer_scale_init_value() -> f64 {
    1e-6
}
fn default_weight_init_value() -> f64 {
    1e-2
}
fn default_n_filters() -> usize {
    32
}
fn default_encoder_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}
fn default_encoder_depths() -> String {
    "3-3-3-3-3-3-8".to_string()
}

/// Semantic tokenizer config — same field set as acoustic minus the
/// decoder fields. `VibeVoiceSemanticTokenizerConfig` at lines 94–145.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VibeVoiceSemanticTokenizerConfig {
    #[serde(default = "default_channels")]
    pub channels: usize,
    #[serde(default)]
    pub corpus_normalize: f32,
    #[serde(default = "default_true")]
    pub causal: bool,
    /// JSON 1.5B value is **128** (not the Python default 64).
    #[serde(default = "default_semantic_vae_dim")]
    pub vae_dim: usize,
    #[serde(default)]
    pub fix_std: f32,
    #[serde(default = "default_std_dist_type_none")]
    pub std_dist_type: String,
    #[serde(default = "default_mixer_layer")]
    pub mixer_layer: String,
    #[serde(default = "default_conv_norm")]
    pub conv_norm: String,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
    #[serde(default = "default_true")]
    pub disable_last_norm: bool,
    #[serde(default = "default_layernorm")]
    pub layernorm: String,
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,
    #[serde(default = "default_true")]
    pub layernorm_elementwise_affine: bool,
    #[serde(default = "default_true")]
    pub conv_bias: bool,
    #[serde(default = "default_layer_scale_init_value")]
    pub layer_scale_init_value: f64,
    #[serde(default = "default_weight_init_value")]
    pub weight_init_value: f64,
    #[serde(default = "default_n_filters")]
    pub encoder_n_filters: usize,
    #[serde(default = "default_encoder_ratios")]
    pub encoder_ratios: Vec<usize>,
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,
    #[serde(default)]
    pub model_type: String,
}

fn default_semantic_vae_dim() -> usize {
    128
}
fn default_std_dist_type_none() -> String {
    "none".to_string()
}

/// Diffusion-head config — `VibeVoiceDiffusionHeadConfig` at lines
/// 148–180. JSON values for the 1.5B variant override the Python defaults
/// in two places (`hidden_size = 1536` matches Qwen2-1.5B's hidden,
/// `speech_vae_dim = 64` is set explicitly).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VibeVoiceDiffusionHeadConfig {
    #[serde(default = "default_dh_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_dh_head_layers")]
    pub head_layers: usize,
    #[serde(default = "default_dh_head_ffn_ratio")]
    pub head_ffn_ratio: f64,
    #[serde(default = "default_dh_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_dh_latent_size")]
    pub latent_size: usize,
    /// VibeVoice-1.5B sets this explicitly to 64 to match
    /// `acoustic_tokenizer.vae_dim`. Optional in the schema.
    #[serde(default)]
    pub speech_vae_dim: Option<usize>,
    /// `"epsilon"` or `"v_prediction"` — VibeVoice ships v_prediction.
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,
    /// `"ddpm"` (the Microsoft-flavoured cosine schedule) is the only
    /// shipped value.
    #[serde(default = "default_diffusion_type")]
    pub diffusion_type: String,
    /// Length of the underlying noise schedule (training-time T).
    #[serde(default = "default_ddpm_num_steps")]
    pub ddpm_num_steps: usize,
    /// DPM-Solver inference steps. VibeVoice-1.5B ships **20**.
    #[serde(default = "default_ddpm_num_inference_steps")]
    pub ddpm_num_inference_steps: usize,
    /// Beta schedule. VibeVoice-1.5B uses `"cosine"`.
    #[serde(default = "default_ddpm_beta_schedule")]
    pub ddpm_beta_schedule: String,
    #[serde(default)]
    pub model_type: String,
}

fn default_dh_hidden_size() -> usize {
    1536
}
fn default_dh_head_layers() -> usize {
    4
}
fn default_dh_head_ffn_ratio() -> f64 {
    3.0
}
fn default_dh_rms_norm_eps() -> f64 {
    1e-5
}
fn default_dh_latent_size() -> usize {
    64
}
fn default_prediction_type() -> String {
    "v_prediction".to_string()
}
fn default_diffusion_type() -> String {
    "ddpm".to_string()
}
fn default_ddpm_num_steps() -> usize {
    1000
}
fn default_ddpm_num_inference_steps() -> usize {
    20
}
fn default_ddpm_beta_schedule() -> String {
    "cosine".to_string()
}

/// Combined tokenizer config used by [`crate::models::vibevoice::tokenizer`]
/// when only one of acoustic / semantic is needed in isolation. Built
/// from the two underlying configs — this is a convenience type, not a
/// wire format.
#[derive(Debug, Clone)]
pub struct VibeVoiceTokenizerConfig {
    pub channels: usize,
    pub vae_dim: usize,
    pub n_filters: usize,
    pub ratios: Vec<usize>,
    pub depths: String,
    pub causal: bool,
    pub pad_mode: String,
    pub conv_norm: String,
    pub layernorm: String,
    pub layernorm_eps: f64,
    pub disable_last_norm: bool,
    pub layer_scale_init_value: f64,
}

impl From<&VibeVoiceAcousticTokenizerConfig> for VibeVoiceTokenizerConfig {
    fn from(c: &VibeVoiceAcousticTokenizerConfig) -> Self {
        Self {
            channels: c.channels,
            vae_dim: c.vae_dim,
            n_filters: c.encoder_n_filters,
            ratios: c.encoder_ratios.clone(),
            depths: c.encoder_depths.clone(),
            causal: c.causal,
            pad_mode: c.pad_mode.clone(),
            conv_norm: c.conv_norm.clone(),
            layernorm: c.layernorm.clone(),
            layernorm_eps: c.layernorm_eps,
            disable_last_norm: c.disable_last_norm,
            layer_scale_init_value: c.layer_scale_init_value,
        }
    }
}

impl From<&VibeVoiceSemanticTokenizerConfig> for VibeVoiceTokenizerConfig {
    fn from(c: &VibeVoiceSemanticTokenizerConfig) -> Self {
        Self {
            channels: c.channels,
            vae_dim: c.vae_dim,
            n_filters: c.encoder_n_filters,
            ratios: c.encoder_ratios.clone(),
            depths: c.encoder_depths.clone(),
            causal: c.causal,
            pad_mode: c.pad_mode.clone(),
            conv_norm: c.conv_norm.clone(),
            layernorm: c.layernorm.clone(),
            layernorm_eps: c.layernorm_eps,
            disable_last_norm: c.disable_last_norm,
            layer_scale_init_value: c.layer_scale_init_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The acoustic config defaults round-trip through serde.
    #[test]
    fn acoustic_defaults_match_python_class() {
        let json = r#"{"vae_dim": 64}"#;
        let cfg: VibeVoiceAcousticTokenizerConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.vae_dim, 64);
        assert_eq!(cfg.fix_std, 0.5);
        assert_eq!(cfg.encoder_ratios, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(cfg.encoder_depths, "3-3-3-3-3-3-8");
        assert!(cfg.causal);
    }

    #[test]
    fn semantic_defaults_use_vae_dim_128() {
        let cfg: VibeVoiceSemanticTokenizerConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.vae_dim, 128);
        assert_eq!(cfg.fix_std, 0.0);
        assert_eq!(cfg.std_dist_type, "none");
    }

    #[test]
    fn diffusion_head_defaults_match_15b() {
        let cfg: VibeVoiceDiffusionHeadConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.hidden_size, 1536);
        assert_eq!(cfg.head_layers, 4);
        assert_eq!(cfg.head_ffn_ratio, 3.0);
        assert_eq!(cfg.latent_size, 64);
        assert_eq!(cfg.ddpm_num_inference_steps, 20);
        assert_eq!(cfg.ddpm_beta_schedule, "cosine");
        assert_eq!(cfg.prediction_type, "v_prediction");
    }

    #[test]
    fn effective_decoder_ratios_falls_back_to_encoder() {
        let cfg: VibeVoiceAcousticTokenizerConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.effective_decoder_ratios(), &[8, 5, 5, 4, 2, 2]);
    }
}

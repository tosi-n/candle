//! Qwen2.5-Omni configuration — serde structs mirroring the released
//! `config.json` of `Qwen/Qwen2.5-Omni-3B` VERBATIM.
//!
//! The model is a 3-stage pipeline: Thinker (Qwen2.5 LLM) → Talker
//! (Qwen2 LLM + codec head, autoregressive discrete codec tokens) →
//! Token2Wav (DiT flow-matching vocoder + BigVGAN). See
//! `hybrie/docs/qwen2_5_omni-port.md` for the full architecture lock.
//!
//! Fields marked "defaults-only" are NOT present in `config.json`; their
//! values come from `configuration_qwen2_5_omni.py` and are supplied here
//! via `#[serde(default = ...)]` so a real config.json deserialises
//! cleanly while still exposing the field.

use serde::Deserialize;

/// Top-level Qwen2.5-Omni config. `model_type: "qwen2_5_omni"`.
#[derive(Debug, Clone, Deserialize)]
pub struct OmniConfig {
    pub thinker_config: ThinkerConfig,
    pub talker_config: TalkerConfig,
    pub token2wav_config: Token2WavConfig,
    #[serde(default = "default_true")]
    pub enable_audio_output: bool,
    #[serde(default = "default_true")]
    pub enable_talker: bool,
}

// ---------------------------------------------------------------------------
// Thinker — Qwen2.5 multimodal LLM (text + audio + vision in → text out)
// ---------------------------------------------------------------------------

/// `thinker_config` (`model_type: qwen2_5_omni_thinker`). Nests text,
/// audio, and vision sub-configs plus the multimodal placeholder token ids.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub text_config: ThinkerTextConfig,
    pub audio_config: AudioEncoderConfig,
    pub vision_config: VisionEncoderConfig,
    #[serde(default = "default_audio_token_index")]
    pub audio_token_index: usize,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: usize,
    #[serde(default = "default_video_token_index")]
    pub video_token_index: usize,
    #[serde(default = "default_vision_start_token_id")]
    pub vision_start_token_id: usize,
    #[serde(default = "default_vision_end_token_id")]
    pub vision_end_token_id: usize,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: usize,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: usize,
}

/// `thinker_config.text_config` (`model_type: qwen2_5_omni_text`).
/// Standard Qwen2.5 decoder with M-RoPE. 3B: 2048h / 36L / 16H / 2kv.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// M-RoPE section split `[t, h, w]`. Thinker = `[16, 24, 24]`.
    #[serde(default = "default_thinker_mrope_section")]
    pub rope_scaling: RopeScaling,
    /// Not in JSON; derived `hidden_size / num_attention_heads` = 128.
    #[serde(default)]
    pub head_dim: Option<usize>,
}

impl ThinkerTextConfig {
    /// Head dim — explicit if present, else `hidden_size / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// `thinker_config.audio_config` (`model_type: qwen2_5_omni_audio_encoder`).
/// Whisper-style encoder but **128 mel bins** and a net ÷4 downsample
/// (conv2 stride-2 + a stride-2 avgpool), output projected to `output_dim`.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncoderConfig {
    pub d_model: usize,
    // The released config carries BOTH `encoder_layers` and
    // `num_hidden_layers` (= 32). Read the standardized key directly; an
    // `alias` here would trip serde's duplicate-field check.
    pub num_hidden_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    /// Windowed-attention span. `n_window = 100`.
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    /// Projection target — the thinker hidden size (2048).
    pub output_dim: usize,
    #[serde(default)]
    pub scale_embedding: bool,
}

/// `thinker_config.vision_config` (`model_type: qwen2_5_omni_vision_encoder`).
/// Qwen2.5-VL windowed-attention ViT + PatchMerger → `out_hidden_size`.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionEncoderConfig {
    // The released config carries BOTH `embed_dim` and `hidden_size`
    // (= 1280). Read `hidden_size` directly; an `alias` would trip serde's
    // duplicate-field check.
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    /// Block indices that use full (not windowed) attention. `[7,15,23,31]`.
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_tokens_per_second")]
    pub tokens_per_second: usize,
}

// ---------------------------------------------------------------------------
// Talker — Qwen2 LLM + codec head (AR discrete codec tokens)
// ---------------------------------------------------------------------------

/// `talker_config` (`model_type: qwen2_5_omni_talker`). A Qwen2 decoder
/// (896h / 24L / 14H / 2kv / head_dim 64) whose lm_head is replaced by a
/// `codec_head: Linear(hidden, vocab_size)`. Generates discrete codec
/// tokens autoregressively. `vocab_size = 8448` includes the codec
/// specials; the actual audio codebook the DiT consumes is ~8192.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    /// Thinker hidden size feeding `thinker_to_talker_proj` (2048→896).
    pub embedding_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    /// Codec head width — 8448 (codebook + specials).
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    /// M-RoPE section. Talker = `[16, 16, 0]`.
    #[serde(default = "default_talker_mrope_section")]
    pub rope_scaling: RopeScaling,
    #[serde(default = "default_position_id_per_seconds")]
    pub position_id_per_seconds: usize,
    #[serde(default = "default_seconds_per_chunk")]
    pub seconds_per_chunk: usize,
    #[serde(default = "default_tts_codec_start")]
    pub tts_codec_start_token_id: usize,
    #[serde(default = "default_tts_codec_end")]
    pub tts_codec_end_token_id: usize,
    #[serde(default = "default_tts_codec_pad")]
    pub tts_codec_pad_token_id: usize,
    #[serde(default = "default_tts_codec_mask")]
    pub tts_codec_mask_token_id: usize,
}

// ---------------------------------------------------------------------------
// Token2Wav — DiT flow-matching vocoder + BigVGAN
// ---------------------------------------------------------------------------

/// `token2wav_config` (`model_type: qwen2_5_omni_token2wav`). Two stages:
/// the DiT turns codec tokens into an 80-dim mel spectrogram via
/// flow-matching (RK4-ODE-solved, CFG, sway schedule); BigVGAN turns the
/// mel into a 16 kHz waveform. **Forced fp32** in the reference.
#[derive(Debug, Clone, Deserialize)]
pub struct Token2WavConfig {
    pub dit_config: DitConfig,
    pub bigvgan_config: BigVganConfig,
}

/// `token2wav_config.dit_config` (`model_type: qwen2_5_omni_dit`).
/// Diffusion-transformer flow-matching vocoder. Block-causal streaming
/// attention. Embeds codes via one `Embedding(num_embeds+1, ...)`; each
/// code expands to `repeats` mel frames.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    #[serde(alias = "dim")]
    pub hidden_size: usize,
    #[serde(alias = "depth")]
    pub num_hidden_layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub ff_mult: usize,
    pub mel_dim: usize,
    pub emb_dim: usize,
    /// Audio codebook size (8193). The `nn.Embedding` is `num_embeds + 1`.
    pub num_embeds: usize,
    /// Each codec token expands to this many mel frames. `repeats = 2`.
    pub repeats: usize,
    // ECAPA-TDNN speaker encoder dims.
    pub enc_dim: usize,
    pub enc_emb_dim: usize,
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<usize>,
    pub enc_attention_channels: usize,
    pub enc_res2net_scale: usize,
    pub enc_se_channels: usize,
    pub enc_lin_neurons: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    // --- defaults-only (from configuration_qwen2_5_omni.py, not in JSON) ---
    /// Block-causal streaming attention block size. Default 24.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default = "default_look_ahead_layers")]
    pub look_ahead_layers: Vec<usize>,
    #[serde(default = "default_look_backward_layers")]
    pub look_backward_layers: Vec<usize>,
}

/// `token2wav_config.bigvgan_config` (`model_type: qwen2_5_omni_bigvgan`).
/// SnakeBeta + AMPBlock + ConvTranspose1d upsampler. mel(80) → 16 kHz wav.
#[derive(Debug, Clone, Deserialize)]
pub struct BigVganConfig {
    pub mel_dim: usize,
    pub upsample_initial_channel: usize,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    #[serde(default)]
    pub use_bias_at_final: bool,
}

// ---------------------------------------------------------------------------
// M-RoPE
// ---------------------------------------------------------------------------

/// Multimodal RoPE scaling. The `mrope_section` splits the rotary dims
/// across temporal / height / width position streams. Thinker uses
/// `[16, 24, 24]`; Talker uses `[16, 16, 0]`.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

// ---------------------------------------------------------------------------
// serde default helpers (values verbatim from the released config sources)
// ---------------------------------------------------------------------------

fn default_true() -> bool {
    true
}
fn default_audio_token_index() -> usize {
    151646
}
fn default_image_token_index() -> usize {
    151655
}
fn default_video_token_index() -> usize {
    151656
}
fn default_vision_start_token_id() -> usize {
    151652
}
fn default_vision_end_token_id() -> usize {
    151653
}
fn default_bos_token_id() -> usize {
    151644
}
fn default_eos_token_id() -> usize {
    151645
}
fn default_pad_token_id() -> usize {
    151643
}
fn default_thinker_mrope_section() -> RopeScaling {
    RopeScaling {
        mrope_section: vec![16, 24, 24],
    }
}
fn default_talker_mrope_section() -> RopeScaling {
    RopeScaling {
        mrope_section: vec![16, 16, 0],
    }
}
fn default_n_window() -> usize {
    100
}
fn default_fullatt_block_indexes() -> Vec<usize> {
    vec![7, 15, 23, 31]
}
fn default_window_size() -> usize {
    112
}
fn default_tokens_per_second() -> usize {
    25
}
fn default_position_id_per_seconds() -> usize {
    25
}
fn default_seconds_per_chunk() -> usize {
    2
}
fn default_tts_codec_start() -> usize {
    8293
}
fn default_tts_codec_end() -> usize {
    8294
}
fn default_tts_codec_pad() -> usize {
    8292
}
fn default_tts_codec_mask() -> usize {
    8296
}
fn default_dropout() -> f64 {
    0.1
}
fn default_block_size() -> usize {
    24
}
fn default_look_ahead_layers() -> Vec<usize> {
    vec![10]
}
fn default_look_backward_layers() -> Vec<usize> {
    vec![0, 20]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deserialize the **real released** `config.json` of
    /// `Qwen/Qwen2.5-Omni-3B` (vendored under `test_fixtures/`) and assert
    /// the locked dims (docs/qwen2_5_omni-port.md §4).
    ///
    /// This is the test that actually catches transcription bugs: the real
    /// file carries extra/standardized keys our structs must tolerate, and
    /// duplicate keys (`encoder_layers`+`num_hidden_layers`,
    /// `embed_dim`+`hidden_size`) that a `#[serde(alias)]` would reject. A
    /// hand-written minimal JSON cannot surface those.
    #[test]
    fn omni_3b_real_config_deserialises_with_correct_dims() {
        let json = include_str!("test_fixtures/qwen2_5_omni_3b_config.json");
        let cfg: OmniConfig =
            serde_json::from_str(json).expect("deserialise REAL Qwen2.5-Omni-3B config.json");

        // Thinker text LLM.
        assert_eq!(cfg.thinker_config.text_config.hidden_size, 2048);
        assert_eq!(cfg.thinker_config.text_config.num_hidden_layers, 36);
        assert_eq!(cfg.thinker_config.text_config.head_dim(), 128); // derived 2048/16
        assert_eq!(
            cfg.thinker_config.text_config.rope_scaling.mrope_section,
            vec![16, 24, 24]
        );
        // Audio encoder — 128 mel bins (NOT 80), proj → 2048.
        assert_eq!(cfg.thinker_config.audio_config.num_mel_bins, 128);
        assert_eq!(cfg.thinker_config.audio_config.num_hidden_layers, 32); // encoder_layers alias
        assert_eq!(cfg.thinker_config.audio_config.output_dim, 2048);
        assert_eq!(cfg.thinker_config.audio_config.n_window, 100); // default
                                                                   // Vision encoder.
        assert_eq!(cfg.thinker_config.vision_config.hidden_size, 1280); // embed_dim alias
        assert_eq!(
            cfg.thinker_config.vision_config.fullatt_block_indexes,
            vec![7, 15, 23, 31]
        );
        // Talker — codec vocab 8448, M-RoPE [16,16,0].
        assert_eq!(cfg.talker_config.vocab_size, 8448);
        assert_eq!(cfg.talker_config.hidden_size, 896);
        assert_eq!(
            cfg.talker_config.rope_scaling.mrope_section,
            vec![16, 16, 0]
        );
        assert_eq!(cfg.talker_config.tts_codec_start_token_id, 8293); // default
                                                                      // Token2Wav DiT — codebook 8193, repeats 2, block-causal defaults.
        assert_eq!(cfg.token2wav_config.dit_config.hidden_size, 1024); // dim alias
        assert_eq!(cfg.token2wav_config.dit_config.num_embeds, 8193);
        assert_eq!(cfg.token2wav_config.dit_config.repeats, 2);
        assert_eq!(cfg.token2wav_config.dit_config.block_size, 24); // default
        assert_eq!(
            cfg.token2wav_config.dit_config.look_backward_layers,
            vec![0, 20]
        );
        // BigVGAN upsampler.
        assert_eq!(
            cfg.token2wav_config.bigvgan_config.upsample_rates,
            vec![5, 3, 2, 2, 2, 2]
        );
        // Output flags default true.
        assert!(cfg.enable_audio_output);
        assert!(cfg.enable_talker);
    }
}

//! Token2Wav — codec tokens → mel (DiT flow-matching) → waveform (BigVGAN).
//!
//! Two-stage architecture inside the released `token2wav.*` checkpoint:
//!
//! 1. **DiT flow-matching vocoder** (`code2wav_dit_model`) — turns
//!    discrete codec tokens into an 80-dim mel spectrogram via a
//!    diffusion-transformer trained as a flow-matching velocity field,
//!    integrated by an ODE solver from t=0 (noise) to t=1 (clean mel).
//!    Uses classifier-free guidance with a doubled batch and a "sway"-
//!    warped time schedule that concentrates resolution near t=0.
//!
//! 2. **BigVGAN vocoder** (`code2wav_bigvgan_model`) — turns the 80-dim
//!    mel at 100 Hz frame rate into a 24 kHz waveform via an
//!    anti-aliased Snake-activation residual conv stack (240× upsample
//!    = 5·3·2·2·2·2).
//!
//! The released model **forces both stages to fp32** (config field
//! `token2wav_config.dit_config.torch_dtype="float32"` and the wrapper
//! does `self.token2wav.float()` twice — `enable_talker` at
//! modeling_qwen2_5_omni.py:3693 and again defensively right before
//! invocation at 3972-3973). On candle this means: load the safetensors
//! at `DType::F32` and never autocast.
//!
//! ## Phase 4 scope
//!
//! `flow_match.rs` lands fully: RK 3/8-rule integrator + sway schedule
//! + CFG combine. Each is a pure numerical primitive that unit-tests
//! against closed-form references with no model weights — so it's
//! immediately verifiable on a laptop. The DiT module + BigVGAN are
//! placeholder skeletons keeping the file structure in place; their
//! forward passes are net-new ports landing in Phase 4.5 (DiT) and
//! Phase 5 (BigVGAN + ECAPA-TDNN).
//!
//! Source-of-truth specs (each catalogues every silent-misload trap):
//! `hybrie/docs/qwen2_5_omni-phases-2-6-spec.md`.

pub mod bigvgan;
pub mod dit;
pub mod ecapa_tdnn;
pub mod flow_match;

pub use bigvgan::{BigVgan, SAMPLE_RATE};
pub use dit::DitModel;
pub use ecapa_tdnn::EcapaTdnn;
pub use flow_match::{cfg_combine, sway_schedule, RungeKutta38Solver};

use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::models::qwen2_5_omni::config::Token2WavConfig;

/// Upstream default sampling hyperparameters for `sample()`
/// (modeling_qwen2_5_omni.py:3548-3552).
pub const DEFAULT_NUM_STEPS: usize = 10;
pub const DEFAULT_GUIDANCE_SCALE: f32 = 0.5;
pub const DEFAULT_SWAY_COEFFICIENT: f32 = -1.0;

/// Token2Wav — the full codec-tokens → waveform stack: the DiT
/// flow-matching vocoder (codes → 80-dim mel) followed by BigVGAN
/// (mel → 24 kHz waveform).
///
/// Always runs at `DType::F32` (the released model forces this — see the
/// module-level doc). Construct with the VarBuilder scoped at
/// `token2wav` so `code2wav_dit_model.*` and `code2wav_bigvgan_model.*`
/// resolve.
#[derive(Debug)]
pub struct Token2WavModel {
    dit: DitModel,
    bigvgan: BigVgan,
}

impl Token2WavModel {
    pub fn new(cfg: &Token2WavConfig, vb: VarBuilder) -> Result<Self> {
        let dit = DitModel::new(&cfg.dit_config, vb.pp("code2wav_dit_model"))?;
        let bigvgan = BigVgan::new(&cfg.bigvgan_config, vb.pp("code2wav_bigvgan_model"))?;
        Ok(Self { dit, bigvgan })
    }

    /// codes `(B, n_codes)` i64 → DiT mel `(B, 80, T)` → BigVGAN
    /// waveform `(B, 1, 240·T)` at 24 kHz.
    ///
    /// - `conditioning` `(B, enc_emb_dim=192)`: the precomputed speaker
    ///   embedding (the `cond` field of a `spk_dict.pt` voice).
    /// - `reference_mel` `(B, T_ref, 80)`: the reference mel fed to the
    ///   DiT's in-block ECAPA speaker encoder (the `ref_mel` field).
    pub fn forward(
        &self,
        codes: &Tensor,
        conditioning: &Tensor,
        reference_mel: &Tensor,
        num_steps: usize,
        guidance_scale: f32,
        sway_coefficient: f32,
    ) -> Result<Tensor> {
        let mel = self.dit.sample(
            codes,
            conditioning,
            reference_mel,
            num_steps,
            guidance_scale,
            sway_coefficient,
        )?;
        self.bigvgan.forward(&mel)
    }

    /// `forward` with the upstream default sampling hyperparameters.
    pub fn forward_default(
        &self,
        codes: &Tensor,
        conditioning: &Tensor,
        reference_mel: &Tensor,
    ) -> Result<Tensor> {
        self.forward(
            codes,
            conditioning,
            reference_mel,
            DEFAULT_NUM_STEPS,
            DEFAULT_GUIDANCE_SCALE,
            DEFAULT_SWAY_COEFFICIENT,
        )
    }
}

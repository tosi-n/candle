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

pub mod flow_match;

pub use flow_match::{cfg_combine, sway_schedule, RungeKutta38Solver};

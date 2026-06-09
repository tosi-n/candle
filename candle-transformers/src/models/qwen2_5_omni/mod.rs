//! Qwen2.5-Omni — multimodal (text + audio + image + video in,
//! text + speech out) model. A 3-stage pipeline:
//!
//! 1. **Thinker** — a Qwen2.5 LLM that consumes interleaved text / audio /
//!    vision embeddings and emits text tokens. (Reuses the candle `qwen2`
//!    decoder + an M-RoPE variant.)
//! 2. **Talker** — a smaller Qwen2 LLM whose lm_head is replaced by a
//!    `codec_head`; it consumes the Thinker's hidden states and emits
//!    **discrete codec tokens autoregressively** (single codebook — NOT
//!    diffusion, NOT RVQ).
//! 3. **Token2Wav** — turns codec tokens into a waveform: a **DiT**
//!    flow-matching vocoder (RK4-ODE-solved, CFG, sway schedule) predicts
//!    an 80-dim mel, then **BigVGAN** renders it to 16 kHz audio. An
//!    **ECAPA-TDNN** speaker encoder over a reference mel provides voice
//!    cloning.
//!
//! Architecture locked from the released `config.json` +
//! `modeling_qwen2_5_omni.py` on 2026-06-04. Full design + reuse table +
//! gotchas: `hybrie/docs/qwen2_5_omni-port.md`.
//!
//! ## Port status
//!
//! - [x] `config.rs` — serde structs mirroring the released config.json.
//! - [x] `audio_encoder.rs` — Whisper-style 128-mel encoder, ÷4, → 2048.
//! - [x] `vision_encoder.rs` — Qwen2.5-VL windowed-attn ViT + PatchMerger.
//! - [x] `mrope.rs` — M-RoPE sectioning (shared thinker + talker).
//! - [x] `thinker.rs` — Qwen2.5 LLM text-only forward + lm_head. Multimodal
//!       embed splice + KV cache are Phase 2.5 follow-ups.
//! - [x] `talker.rs` — Qwen2 LLM + codec_head + thinker_to_talker_proj.
//!       Per-step rolling-buffer fusion + sampling are Phase 3.5
//!       follow-ups (wired into the decode loop).
//! - [~] `token2wav/`:
//!     - [x] `flow_match.rs` — Kutta 3/8-rule ODE solver + sway schedule
//!           + CFG combine. Pure numerical primitives, closed-form
//!           verified.
//!     - [ ] `dit.rs` — DiT flow-matching velocity field (22 blocks).
//!     - [ ] `bigvgan.rs` — Anti-aliased SnakeBeta vocoder.
//!     - [ ] `ecapa_tdnn.rs` — Speaker encoder for voice cloning.
//! - [ ] `inference.rs` — end-to-end `generate()`.
//!
//! Each stage is built and weight-load/parity-tested on a Lambda GPU box
//! (x86 + CUDA), never compiled locally — the weights are ~6 GB and the
//! token2wav stack is forced fp32.

pub mod audio_encoder;
pub mod config;
pub mod mrope;
pub mod talker;
pub mod thinker;
pub mod token2wav;
pub mod vision_encoder;

pub use audio_encoder::AudioEncoder;
pub use config::{
    AudioEncoderConfig, BigVganConfig, DitConfig, OmniConfig, TalkerConfig, ThinkerConfig,
    ThinkerTextConfig, Token2WavConfig, VisionEncoderConfig,
};
pub use mrope::{text_only_position_ids, MRopeTable};
pub use talker::Talker;
pub use thinker::Thinker;
pub use token2wav::{cfg_combine, sway_schedule, RungeKutta38Solver};
pub use vision_encoder::VisionEncoder;

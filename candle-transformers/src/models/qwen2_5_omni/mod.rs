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
//! - [ ] `mrope.rs` — M-RoPE sectioning (shared thinker + talker).
//! - [ ] `thinker.rs` — Qwen2.5 LLM + multimodal embed merge.
//! - [ ] `talker.rs` — Qwen2 LLM + codec_head + AR sampling.
//! - [ ] `token2wav/` — dit.rs, flow_match.rs, ecapa_tdnn.rs, bigvgan.rs.
//! - [ ] `inference.rs` — end-to-end `generate()`.
//!
//! Each stage is built and weight-load/parity-tested on a Lambda GPU box
//! (x86 + CUDA), never compiled locally — the weights are ~6 GB and the
//! token2wav stack is forced fp32.

pub mod audio_encoder;
pub mod config;
pub mod vision_encoder;

pub use audio_encoder::AudioEncoder;
pub use config::{
    AudioEncoderConfig, BigVganConfig, DitConfig, OmniConfig, TalkerConfig, ThinkerConfig,
    ThinkerTextConfig, Token2WavConfig, VisionEncoderConfig,
};
pub use vision_encoder::VisionEncoder;

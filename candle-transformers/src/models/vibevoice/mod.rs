//! VibeVoice — Microsoft's diffusion-based text-to-speech model.
//!
//! Architecture (per the model card and the upstream Python at
//! <https://github.com/microsoft/VibeVoice>):
//!
//! - **LLM backbone**: Qwen2.5 (1.5B or 7B variant). Reused from
//!   [`crate::models::qwen2`] — not re-ported here.
//! - **Diffusion head** ([`diffusion_head`]): a DiT-style denoiser that
//!   predicts noise / velocity at a given timestep, conditioned on Qwen
//!   hidden states. ~123M parameters.
//! - **DPM-Solver scheduler** ([`dpm_solver`]): sampling schedule used to
//!   denoise the diffusion head's output across a small number of steps
//!   (VibeVoice-1.5B ships 20 steps). Self-contained — does not depend
//!   on [`crate::models::stable_diffusion`].
//! - **Acoustic tokenizer** ([`tokenizer`]): ConvNeXt-style σ-VAE
//!   operating at 7.5 Hz against a 24 kHz waveform.
//! - **Semantic tokenizer** (also in [`tokenizer`]): encoder-only at
//!   inference, used to extract speaker conditioning embeddings from a
//!   reference clip.
//!
//! ## Upstream references
//!
//! - Repo: <https://github.com/microsoft/VibeVoice>
//! - Model card (1.5B): <https://huggingface.co/microsoft/VibeVoice-1.5B>
//! - Tech report: <https://arxiv.org/abs/2508.19205>
//!
//! ## License
//!
//! Upstream VibeVoice is MIT-licensed with a research-only carve-out.
//! Per the model card, every synthesized audio file must carry an
//! audible disclaimer and an imperceptible watermark; downstream
//! consumers (e.g. HybrIE) are responsible for honouring those mitigations.
//!
//! ## Status
//!
//! - [`config`]: complete, `cargo test` validated.
//! - [`diffusion_head`]: complete, smoke tested.
//! - [`dpm_solver`]: complete (dpmsolver++/midpoint/v_prediction/cosine
//!   path only — VibeVoice's exclusive configuration). Numerical match
//!   against the Python reference still pending an end-to-end test.
//! - [`tokenizer`]: skeleton only. Encoder/decoder bodies pending the
//!   ~800–1000 LOC port plus checkpoint validation.
//! - [`model`]: skeleton only. The full inference loop (Qwen forward →
//!   diffusion → audio decode) lives in
//!   `vibevoice/modular/modeling_vibevoice_streaming_inference.py`
//!   upstream and lands as a follow-up commit.

pub mod config;
pub mod diffusion_head;
pub mod dpm_solver;
pub mod model;
pub mod tokenizer;

pub use config::{
    VibeVoiceAcousticTokenizerConfig, VibeVoiceConfig, VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig, VibeVoiceTokenizerConfig,
};
pub use diffusion_head::VibeVoiceDiffusionHead;
pub use dpm_solver::{BetaSchedule, DpmSolver, DpmSolverConfig, PredictionType};
pub use model::VibeVoiceModel;
pub use tokenizer::{AcousticTokenizer, SemanticTokenizer};

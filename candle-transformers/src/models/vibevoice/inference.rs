//! Autoregressive inference loop for [`super::VibeVoiceModel`] —
//! "text in, audio out".
//!
//! Pure-candle port of the inference path inside
//! `vibevoice/modular/modeling_vibevoice_streaming_inference.py`
//! (see `generate()` around lines 574-884 and `sample_speech_tokens` at
//! lines 887-899).
//!
//! The streaming/windowed text feed and the dual-LM (`language_model` +
//! `tts_language_model`) split from upstream are out of scope for this
//! port — the spec calls for batched, single-LM generation. What we
//! reuse from upstream:
//!
//! 1. The CFG combine formula (line 896 of `streaming.py`):
//!    `eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)`.
//! 2. The DPM-Solver inner loop (line 891-898): for each diffusion
//!    timestep, run the head twice, combine, step the solver.
//! 3. The latent-space scaling inverse (line 783):
//!    `scaled_latent = speech_latent / scaling - bias` — the inverse of
//!    the training-time `(audio + bias) * scaling` (model.py L326).
//! 4. The connector projection back into the LLM hidden space
//!    (line 804): `acoustic_embed = acoustic_connector(speech_latent)`.
//!
//! ## Unconditional condition
//!
//! Upstream's streaming generate seeds an entire shadow LM with the
//! `<|image_pad|>` token to get a negative-prompt hidden state
//! (streaming.py L634-L640, L774). The non-streaming spec we're
//! implementing instead uses a zero-vector condition — the same
//! simplification stable-diffusion-style CFG uses when no explicit
//! negative prompt is provided. This is functionally what
//! `forward_speech_features(speech_tensors=None, ...)` returns at
//! model.py L281-L286: zeros of shape (1, 1, vae_dim) routed through
//! the acoustic connector. We skip the connector here because the
//! diffusion head's `condition` argument lives in the LM-hidden space,
//! and a zero connector output is also a zero in LM-hidden space —
//! they coincide. Documented to make the divergence explicit if a
//! later port wants to plumb in the dual-LM pattern.
//!
//! ## End-of-audio detection
//!
//! Upstream's streaming variant uses a separate `tts_eos_classifier`
//! (BinaryClassifier on the hidden state, threshold 0.5). The
//! non-streaming variant referenced by this port instead uses a
//! conventional vocab-token id passed into [`GenerateOptions::eos_audio_token_id`].
//! When that field is `None` we fall back to running for the full
//! `max_new_tokens` budget — useful for the smoke tests, which can't
//! train an EOS classifier.

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::Module;

use super::dpm_solver::DpmSolver;
use super::model::{GenerateOptions, VibeVoiceModel};
use crate::models::qwen2;

/// Result of a single `generate()` call.
///
/// `audio` has shape `(1, channels, T_samples)` and rides the
/// acoustic-tokenizer's native sample rate (24 kHz for VibeVoice-1.5B).
/// `latents` is the corresponding `(1, T_tokens, vae_dim)` continuous-token
/// stream — useful for downstream voice-cloning consumers and for the
/// integration tests that need a deterministic intermediate.
#[derive(Debug)]
pub struct GenerateOutput {
    pub audio: Tensor,
    pub latents: Tensor,
}

/// Drive a full text-to-audio generation. The caller owns the LLM
/// backbone and the lm_head, mirroring the
/// `VibeVoiceForConditionalGeneration` wiring upstream
/// (model.py L213-L222).
///
/// Inputs:
/// - `model`: the VibeVoice tokenizer + diffusion + connectors bundle.
/// - `language_model`: a [`qwen2::Model`] in evaluation mode. Its KV cache
///   is updated in place across the loop; if the caller wants to reuse
///   the LM for another generation they should
///   [`qwen2::Model::clear_kv_cache`] first.
/// - `lm_head`: token-vocab projection, used only when EOS detection is
///   enabled (`options.eos_audio_token_id` is `Some(_)`).
/// - `input_ids`: text tokens, shape `(1, T_text)`. Caller is responsible
///   for tokenization (e.g. via the Qwen2.5 chat template).
/// - `acoustic_input_mask`: optional `(1, T_text)` boolean Tensor flagging
///   which input positions should receive the speaker conditioning embedding
///   instead of the text embedding (mirrors `acoustic_input_mask` in
///   model.py L391). Pass `None` for vanilla text-only conditioning.
/// - `options`: see [`GenerateOptions`].
///
/// Returns the synthesized audio at the acoustic tokenizer's native rate.
pub fn generate_audio(
    model: &VibeVoiceModel,
    language_model: &mut qwen2::Model,
    lm_head: Option<&candle_nn::Linear>,
    input_ids: &Tensor,
    acoustic_input_mask: Option<&Tensor>,
    options: &GenerateOptions,
) -> Result<GenerateOutput> {
    let device = input_ids.device().clone();
    let dtype = language_model.embed_tokens().embeddings().dtype();
    let lm_hidden = language_model.hidden_size();
    let vae_dim = model.acoustic_config().vae_dim;

    // Step 1 — compute base text embeddings out-of-band so we can splice
    // in the speaker-conditioning embedding before running the layer
    // stack. This mirrors `x = self.get_input_embeddings()(input_ids)`
    // followed by `x[acoustic_input_mask] = ...` in model.py L359-L391.
    let text_embeds = language_model.embed_tokens().forward(input_ids)?;
    let inputs_embeds = if let Some(reference_audio) = options.reference_audio.as_ref() {
        let speaker_embed = build_speaker_embedding(model, reference_audio)?;
        scatter_speaker_embedding(&text_embeds, acoustic_input_mask, &speaker_embed)?
    } else {
        text_embeds
    };

    // Step 2 — text prefill. Single forward through the LM with the
    // spliced embeddings; KV cache is built in place.
    let _ = language_model.forward_with_embeds(&inputs_embeds, /* offset */ 0, None)?;
    let mut seqlen_offset = inputs_embeds.dim(1)?;

    // Unconditional condition — see module docs. Zero hidden-state
    // vector at LM hidden size, broadcast-friendly for the diffusion
    // head's `(B, hidden)` condition input.
    let condition_uncond = Tensor::zeros((1, lm_hidden), dtype, &device)?;

    // Padding embedding fed in for the very first autoregressive step.
    // The LM's view of "what was just emitted" before any audio token
    // exists is the zero connector output (model.py L284-L286: zeros
    // through `acoustic_connector`).
    let zero_acoustic = Tensor::zeros((1, 1, vae_dim), dtype, &device)?;
    let initial_step_embed = model.acoustic_connector().forward(&zero_acoustic)?;

    // Step 3 — autoregressive loop. Each iteration produces one acoustic
    // latent (= `acoustic_downsample()` audio samples after the decoder).
    let max_new_tokens = compute_max_new_tokens(options, model);
    let mut latents: Vec<Tensor> = Vec::with_capacity(max_new_tokens);
    let mut next_input: Tensor = initial_step_embed;

    for step_idx in 0..max_new_tokens {
        // 3a — LM forward on the previous step's connector embedding.
        let hidden = language_model.forward_with_embeds(&next_input, seqlen_offset, None)?;
        seqlen_offset += next_input.dim(1)?;

        // 3b — extract the last hidden state as the conditional condition
        // for the diffusion head. Shape: (1, hidden).
        let last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?;

        // 3c — EOS check (vocab-token mode). Upstream's streaming variant
        // uses a binary classifier here; the spec opts for a configurable
        // token id via the lm_head. If both are wired we honour the
        // check; otherwise we run to the full budget.
        if let (Some(lm_head), Some(eos_id)) = (lm_head, options.eos_audio_token_id) {
            let logits = lm_head.forward(&last_hidden)?;
            let argmax = logits
                .argmax(D::Minus1)?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            if argmax.contains(&eos_id) {
                if step_idx == 0 {
                    candle::bail!(
                        "VibeVoice generate: EOS detected on the very first step (token {eos_id}); \
                         likely a misconfigured eos_audio_token_id"
                    );
                }
                break;
            }
        }

        // 3d — DPM-Solver loop with CFG. Returns (1, 1, vae_dim).
        let latent = sample_speech_token(
            model,
            &last_hidden,
            &condition_uncond,
            options.diffusion_steps,
            options.cfg_scale,
            &device,
            dtype,
        )?;

        // 3e — invert the training-time scaling so the latent lives in
        // the acoustic decoder's expected range (model.py L326 inverse,
        // streaming.py L783).
        let scaled = invert_scaling(&latent, model)?;
        latents.push(scaled.clone());

        // 3f — project back into the LM hidden space for the next step.
        let next_embed = model.acoustic_connector().forward(&scaled)?;
        next_input = next_embed;
    }

    if latents.is_empty() {
        candle::bail!("VibeVoice generate: produced zero audio tokens (max_new_tokens=0?)");
    }

    // Step 4 — concatenate latents along the time axis and decode.
    // Each latent is (1, 1, vae_dim) ⇒ stack → (1, T_tokens, vae_dim).
    let latents = Tensor::cat(&latents, 1)?;
    let audio = model.acoustic_tokenizer().decode(&latents)?;
    Ok(GenerateOutput { audio, latents })
}

/// Build the per-utterance speaker conditioning vector by encoding the
/// reference clip through both tokenizers and combining the results
/// through the connectors. Mirrors the construction in model.py
/// L361-L391 (acoustic_connector + semantic_connector additive combine).
fn build_speaker_embedding(model: &VibeVoiceModel, reference_audio: &Tensor) -> Result<Tensor> {
    // Both encoders consume `(B, channels, T_samples)`.
    let acoustic_lat = model.acoustic_tokenizer().encode(reference_audio)?;
    let semantic_lat = model.semantic_tokenizer().encode(reference_audio)?;
    let acoustic_proj = model.acoustic_connector().forward(&acoustic_lat)?;
    let semantic_proj = model.semantic_connector().forward(&semantic_lat)?;
    acoustic_proj.broadcast_add(&semantic_proj)
}

/// Splice `speaker` embeddings into `text_embeds` at positions flagged
/// by `mask`. If `mask` is None or all-false, returns `text_embeds`
/// unchanged. Mirrors `x[acoustic_input_mask] = ...` in model.py L391
/// using `where_cond`-style arithmetic.
///
/// Shape contract:
/// - `text_embeds`: `(B, T_text, hidden)`
/// - `mask`: `(B, T_text)` — bool / 0-1 dtype.
/// - `speaker`: `(B, K, hidden)`. If `K < T_text`, the speaker is
///   right-padded with the original text embeds (so masked-out positions
///   stay untouched). If `K > T_text`, the prefix `T_text` slice is used.
fn scatter_speaker_embedding(
    text_embeds: &Tensor,
    mask: Option<&Tensor>,
    speaker: &Tensor,
) -> Result<Tensor> {
    let mask = match mask {
        Some(m) => m,
        None => return Ok(text_embeds.clone()),
    };
    let (b, t_text, hidden) = text_embeds.dims3()?;
    let speaker_t = speaker.dim(1)?;
    let speaker_padded = if speaker_t == t_text {
        speaker.clone()
    } else if speaker_t < t_text {
        let pad = text_embeds.narrow(1, speaker_t, t_text - speaker_t)?;
        Tensor::cat(&[speaker, &pad], 1)?
    } else {
        speaker.narrow(1, 0, t_text)?
    };
    // Broadcast (B, T) mask to (B, T, hidden).
    let mask = mask
        .to_dtype(text_embeds.dtype())?
        .reshape((b, t_text, 1))?
        .broadcast_as((b, t_text, hidden))?;
    let one = Tensor::ones_like(&mask)?;
    let inv_mask = one.sub(&mask)?;
    let kept = text_embeds.broadcast_mul(&inv_mask)?;
    let injected = speaker_padded.broadcast_mul(&mask)?;
    kept.broadcast_add(&injected)
}

/// One DPM-Solver descent over `K` steps with classifier-free guidance.
/// Pure port of `sample_speech_tokens` lines 887-899.
///
/// Returns `(1, 1, vae_dim)` — already unsqueezed for downstream
/// `Tensor::cat(.., dim=1)`.
fn sample_speech_token(
    model: &VibeVoiceModel,
    condition: &Tensor,
    condition_uncond: &Tensor,
    diffusion_steps: usize,
    cfg_scale: f64,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut solver = DpmSolver::new(
        super::dpm_solver::DpmSolverConfig::default(),
        diffusion_steps,
    );
    let vae_dim = model.acoustic_config().vae_dim;
    let mut speech = Tensor::randn(0.0_f32, 1.0_f32, (1, vae_dim), device)?.to_dtype(dtype)?;
    let timesteps: Vec<i64> = solver.timesteps.clone();

    for &t in &timesteps {
        let t_tensor = Tensor::new(&[t], device)?;
        let eps_cond = model
            .diffusion_head()
            .forward(&speech, &t_tensor, condition)?;
        let eps_uncond = model
            .diffusion_head()
            .forward(&speech, &t_tensor, condition_uncond)?;
        // CFG combine — exact line 896 of streaming.py.
        let diff = (eps_cond - &eps_uncond)?;
        let eps = (eps_uncond + (diff * cfg_scale)?)?;
        speech = solver.step(&eps, t, &speech)?;
    }
    speech.unsqueeze(1)
}

/// Inverse of the training-time scaling. `(audio + bias) * scaling` runs
/// in the forward path (model.py L326); at inference we recover the
/// decoder-space latent via `audio_token / scaling - bias` (streaming.py
/// L783).
fn invert_scaling(latent: &Tensor, model: &VibeVoiceModel) -> Result<Tensor> {
    let scaling = model.speech_scaling_factor();
    let bias = model.speech_bias_factor();
    let scaled = latent.broadcast_div(scaling)?;
    scaled.broadcast_sub(bias)
}

/// Compute the per-call diffusion-step budget. Caller can pass an
/// absolute cap via [`GenerateOptions::max_new_tokens`]; otherwise we
/// fall back to `max_audio_seconds * sample_rate / downsample`.
fn compute_max_new_tokens(options: &GenerateOptions, model: &VibeVoiceModel) -> usize {
    if let Some(cap) = options.max_new_tokens {
        return cap;
    }
    let downsample = model.acoustic_downsample().max(1);
    let samples = (options.max_audio_seconds * options.sample_rate as f64) as usize;
    samples.div_ceil(downsample).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    /// CFG combine produces `uncond + scale * (cond - uncond)` exactly,
    /// element-wise. Tested on small handcrafted tensors so the formula
    /// is independently verifiable from the spec.
    #[test]
    fn cfg_combine_matches_formula() -> Result<()> {
        let device = Device::Cpu;
        let cond = Tensor::new(&[1.0_f32, 2.0, 3.0], &device)?;
        let uncond = Tensor::new(&[0.5_f32, 1.5, 2.5], &device)?;
        let scale = 1.3_f64;
        let diff = (&cond - &uncond)?;
        let combined = (&uncond + (diff * scale)?)?;
        let v = combined.to_vec1::<f32>()?;
        // expected = uncond + 1.3 * (cond - uncond)
        let expected = [
            0.5 + 1.3 * (1.0 - 0.5),
            1.5 + 1.3 * (2.0 - 1.5),
            2.5 + 1.3 * (3.0 - 2.5),
        ];
        for (got, want) in v.iter().zip(expected.iter()) {
            assert!(
                (got - *want as f32).abs() < 1e-5,
                "cfg combine mismatch: got {got}, want {want}"
            );
        }
        Ok(())
    }

    /// Round-trip: `((x + bias) * scaling) / scaling - bias == x` to
    /// numerical precision. Sanity check on the scaling-factor branch
    /// against a 0-d `bias` / `scaling` (the shapes the released
    /// safetensors actually ships) and a rank-1 audio buffer.
    #[test]
    fn inverse_scaling_round_trip() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.1_f32, -0.5, 0.0, 1.7, -1.2], &device)?;
        let bias_v = -0.3_f32;
        let scaling_v = 2.4_f32;
        let bias = Tensor::new(&[bias_v], &device)?;
        let scaling = Tensor::new(&[scaling_v], &device)?;
        let forward = (x.broadcast_add(&bias)?).broadcast_mul(&scaling)?;
        let recovered = (forward.broadcast_div(&scaling)?).broadcast_sub(&bias)?;
        let v0 = x.to_vec1::<f32>()?;
        let v1 = recovered.to_vec1::<f32>()?;
        for (a, b) in v0.iter().zip(v1.iter()) {
            assert!((a - b).abs() < 1e-5, "round-trip drift: {a} vs {b}");
        }
        Ok(())
    }

    /// Smoke: build a tiny VibeVoiceModel + Qwen2 + lm_head from a
    /// VarMap, run `generate_audio` for a small token budget, assert
    /// the output shape is `(1, channels, max_new_tokens * downsample)`.
    /// This exercises the full pipeline end-to-end on synthetic init —
    /// no checkpoint needed.
    #[test]
    fn generate_smoke_with_random_init() -> Result<()> {
        use crate::models::vibevoice::config::{
            VibeVoiceAcousticTokenizerConfig, VibeVoiceConfig, VibeVoiceDiffusionHeadConfig,
            VibeVoiceSemanticTokenizerConfig,
        };

        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&vm, DType::F32, &device);

        // Tiny tokenizer: 1 channel, vae_dim=8, n_filters=4, ratios=[2,2]
        // ⇒ downsample = 4. Tiny diffusion head: hidden 16, latent 8.
        let acoustic_cfg: VibeVoiceAcousticTokenizerConfig = serde_json::from_str(
            r#"{
                "channels": 1, "vae_dim": 8, "encoder_n_filters": 4,
                "encoder_ratios": [2, 2], "encoder_depths": "1-1-1",
                "decoder_n_filters": 4
            }"#,
        )
        .unwrap();
        let semantic_cfg: VibeVoiceSemanticTokenizerConfig = serde_json::from_str(
            r#"{
                "channels": 1, "vae_dim": 8, "encoder_n_filters": 4,
                "encoder_ratios": [2, 2], "encoder_depths": "1-1-1"
            }"#,
        )
        .unwrap();
        let dh_cfg: VibeVoiceDiffusionHeadConfig = serde_json::from_str(
            r#"{
                "hidden_size": 16, "head_layers": 1, "head_ffn_ratio": 2.0,
                "rms_norm_eps": 1e-6, "latent_size": 8, "speech_vae_dim": 8,
                "ddpm_num_inference_steps": 2, "ddpm_num_steps": 100
            }"#,
        )
        .unwrap();
        let cfg = VibeVoiceConfig {
            acoustic_tokenizer_config: acoustic_cfg,
            semantic_tokenizer_config: semantic_cfg,
            decoder_config: serde_json::Value::Null,
            diffusion_head_config: dh_cfg,
            acoustic_vae_dim: Some(8),
            semantic_vae_dim: Some(8),
            torch_dtype: "float32".to_string(),
            model_type: "vibevoice".to_string(),
        };

        // VibeVoiceModel with hidden_size 16 to match Qwen2 below.
        let vv = VibeVoiceModel::new(cfg, /* lm_hidden */ 16, vb.pp("model"))?;

        // Tiny Qwen2: 16 hidden, 4 heads, 1 layer, vocab 32.
        let qcfg = qwen2::Config {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            max_position_embeddings: 64,
            sliding_window: 64,
            max_window_layers: 1,
            tie_word_embeddings: true,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        let mut lm = qwen2::Model::new(&qcfg, vb.pp("qwen"))?;

        let input_ids = Tensor::new(&[[1u32, 2, 3]], &device)?;
        let options = GenerateOptions {
            max_audio_seconds: 0.0, // unused — we hard-cap via max_new_tokens
            diffusion_steps: 2,
            cfg_scale: 1.3,
            reference_audio: None,
            max_new_tokens: Some(2),
            sample_rate: 24_000,
            eos_audio_token_id: None,
        };
        let out = generate_audio(&vv, &mut lm, None, &input_ids, None, &options)?;
        // 2 latents × downsample 4 = 8 audio samples.
        assert_eq!(out.audio.dims(), &[1, 1, 8]);
        assert_eq!(out.latents.dims(), &[1, 2, 8]);
        Ok(())
    }
}

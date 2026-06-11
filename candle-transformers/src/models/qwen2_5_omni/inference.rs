//! Qwen2.5-Omni end-to-end model — Thinker → Talker → Token2Wav.
//!
//! Mirrors `Qwen2_5OmniForConditionalGeneration` (upstream
//! `modeling_qwen2_5_omni.py:3671-3982`). Composes the four ported
//! stages into one model:
//!
//! 1. **Thinker** (`thinker.*`) — Qwen2.5 multimodal LLM; consumes
//!    text + audio + vision and emits text tokens + per-layer hidden
//!    states.
//! 2. **Talker** (`talker.*`) — Qwen2 LLM + codec head; consumes the
//!    Thinker's hidden states and autoregressively emits discrete codec
//!    tokens.
//! 3. **Token2Wav** (`token2wav.*`) — DiT flow-matching vocoder + BigVGAN;
//!    turns codec tokens into a 24 kHz waveform.
//!
//! ## THE FUSION (the single highest-risk port detail)
//!
//! Upstream line 3927 builds the Talker's first input embedding as an
//! **elementwise add**, NOT a concat:
//!
//! ```text
//! talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
//! ```
//!
//! where `thinker_hidden_states[0]` is the Thinker's last-layer hidden
//! state for the prompt and `thinker_token_embeds[0]` is the Thinker's
//! *input* embedding for the same tokens, with audio/image/video
//! placeholder positions zeroed. [`Qwen2_5OmniModel::fuse_for_talker`]
//! implements exactly this so the relationship is explicit even while
//! the surrounding AR loop is still a documented stub.
//!
//! ## Phase 6 scope (this file)
//!
//! - Full model construction from the released checkpoint (all three
//!   stages load, no missing keys).
//! - A working **Token2Wav path**: given codec tokens + speaker
//!   conditioning + a reference mel, produce a waveform. This is the
//!   stage that turns the model's output into audio and is fully wired.
//! - The **fusion primitive** [`fuse_for_talker`], unit-tested.
//! - The **Talker autoregressive sampling loop** is real and KV-cached:
//!   prefill `[fused(prompt), tts_bos, fused(gen[0])]` once, then one
//!   new token per step with top-k/top-p/temperature + repetition
//!   penalty sampling (see `generate_speech` §5). The **`get_rope_index`
//!   multimodal position builder** + **`spk_dict.pt` voice loader**
//!   remain out of scope (text-only positions; speaker conditioning is
//!   preprocessed to safetensors externally).
//!   [`Qwen2_5OmniModel::generate_codes_stub`] survives only for shape
//!   tests.
//!
//! Hard constraints (upstream):
//! - **`token2wav` is always fp32** (forced twice upstream); load its
//!   shards at `DType::F32`.
//! - **batch size 1** for audio output (`generate()` raises otherwise).

#[cfg(test)]
use candle::DType;
use candle::{Device, IndexOp, Result, Tensor, D};
use candle_nn::VarBuilder;

use super::config::OmniConfig;
use super::talker::Talker;
use super::thinker::Thinker;
use super::token2wav::Token2WavModel;
use crate::generation::{LogitsProcessor, Sampling};

/// Talker codec-token sampling config — mirrors upstream's `talker_*`
/// GenerationConfig defaults (`modeling_qwen2_5_omni.py:3765-3771`).
///
/// **Greedy degenerates** on codec tokens (verified Lambda run: 11 unique
/// codes out of 256 → audibly not speech). Upstream defaults
/// (`TopKThenTopP { k=40, p=0.8, temperature=0.9 }` + `repetition_penalty=1.05`)
/// are what makes the AR loop produce intelligible speech.
#[derive(Debug, Clone, Copy)]
pub struct TalkerSampling {
    pub seed: u64,
    pub top_k: usize,
    pub top_p: f64,
    pub temperature: f64,
    pub repetition_penalty: f32,
}

impl Default for TalkerSampling {
    /// Upstream `talker_*` defaults — the sampling shape proven to yield
    /// intelligible speech in the released Python pipeline.
    fn default() -> Self {
        Self {
            seed: 0,
            top_k: 40,
            top_p: 0.8,
            temperature: 0.9,
            repetition_penalty: 1.05,
        }
    }
}

impl TalkerSampling {
    /// Pure-greedy (argmax) — useful for deterministic tests and as the
    /// known-degenerate baseline. Do NOT ship this for audio.
    pub const GREEDY: Self = Self {
        seed: 0,
        top_k: 0,
        top_p: 0.0,
        temperature: 0.0,
        repetition_penalty: 1.0,
    };

    fn is_greedy(&self) -> bool {
        self.top_k == 0 || self.temperature < 1e-7
    }

    fn make_processor(&self) -> LogitsProcessor {
        let sampling = if self.is_greedy() {
            Sampling::ArgMax
        } else {
            Sampling::TopKThenTopP {
                k: self.top_k,
                p: self.top_p,
                temperature: self.temperature,
            }
        };
        LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

/// Text special-token ids consumed by the Talker prefill / reply stream.
/// These are NOT all present in `config.json`, so they're pinned here as
/// constants (verbatim from the released tokenizer + upstream
/// `modeling_qwen2_5_omni.py`). The text BOS the Talker prefill expects
/// is the speaker's `bos_token` (passed in as a parameter, sourced from
/// `spk_dict.pt`); `text_eos` / `text_pad` terminate + pad the rolling
/// reply stream.
/// TTS text-stream terminators appended to the Talker reply stream
/// (upstream `talker.text_eos_token` / `text_pad_token` =
/// `tts_text_end_token_id` / `tts_text_pad_token_id`).
const TEXT_EOS_TOKEN_ID: i64 = 151861;
const TEXT_PAD_TOKEN_ID: i64 = 151859;
/// The Thinker's CHAT stop token `<|im_end|>`. This is what an
/// instruct-tuned Qwen emits at the end of a chat reply — NOT 151861
/// (a TTS special that never appears in chat text). Stopping greedy
/// decode on the wrong id made the Thinker always run to max_new tokens
/// (runs 6-7: text_ids [1, 64]) and the Talker speak runaway text.
const THINKER_CHAT_EOS_TOKEN_ID: i64 = 151645;

/// Streaming decode constants. 50 codes = 2 s — the DiT's native
/// block-causal chunk (`seconds_per_chunk = 2`, `position_id_per_seconds
/// = 25`), so chunked windows are in-distribution. 25 codes (1 s) of left
/// context anchor each chunk's prosody to what was already spoken;
/// 1 200 samples = 50 ms of linear crossfade hide the noise-realization
/// difference between consecutive flow-matching decodes.
pub const STREAM_CHUNK_CODES: usize = 50;
const CHUNK_CTX_CODES: usize = 25;
const CROSSFADE_SAMPLES: usize = 1_200;

/// The composed Qwen2.5-Omni model.
#[derive(Debug)]
pub struct Qwen2_5OmniModel {
    thinker: Thinker,
    talker: Talker,
    token2wav: Token2WavModel,
}

impl Qwen2_5OmniModel {
    /// Build all three stages from a VarBuilder at the checkpoint root
    /// (so `thinker.*`, `talker.*`, `token2wav.*` resolve).
    ///
    /// NOTE: the caller is responsible for honoring the fp32 constraint
    /// on `token2wav`. The simplest correct path is to load the whole
    /// model at `DType::F32`; a future refinement can load Thinker/Talker
    /// at BF16 and Token2Wav at F32 via two VarBuilders over the same
    /// shards.
    pub fn new(cfg: &OmniConfig, vb: VarBuilder) -> Result<Self> {
        let thinker = Thinker::new(&cfg.thinker_config.text_config, vb.pp("thinker"))?;
        let talker = Talker::new(&cfg.talker_config, vb.pp("talker"))?;
        let token2wav = Token2WavModel::new(&cfg.token2wav_config, vb.pp("token2wav"))?;
        Ok(Self {
            thinker,
            talker,
            token2wav,
        })
    }

    pub fn thinker(&self) -> &Thinker {
        &self.thinker
    }
    pub fn talker(&self) -> &Talker {
        &self.talker
    }
    pub fn token2wav(&self) -> &Token2WavModel {
        &self.token2wav
    }

    /// THE FUSION (upstream line 3927). Returns
    /// `thinker_hidden_states + thinker_token_embeds` — an elementwise
    /// add, both `(B, S, thinker_hidden=2048)`. The caller must have
    /// already zeroed the audio/image/video placeholder positions in
    /// `thinker_token_embeds` (upstream `masked_scatter_` with zeros).
    ///
    /// The result is what feeds `Talker::project_thinker` →
    /// `forward_from_projected_embeds` to start codec generation.
    pub fn fuse_for_talker(
        thinker_hidden_states: &Tensor,
        thinker_token_embeds: &Tensor,
    ) -> Result<Tensor> {
        thinker_hidden_states + thinker_token_embeds
    }

    /// **STUB** for the Talker autoregressive codec-token sampling loop.
    ///
    /// The real implementation (Phase 6.5) drives `Talker` with KV cache,
    /// per-step `codec_embed + thinker_reply_part[:, :1]` fusion, top-k
    /// (40) / top-p (0.8) / temperature (0.9) / repetition-penalty (1.05)
    /// sampling on the F32 codec logits, `suppress_tokens=[codec_bos]`,
    /// terminating on `eos ∈ {tts_codec_pad, tts_codec_end}`, then strips
    /// `[:, prefix:-1]`. Until that lands, return a short deterministic
    /// codec-token sequence so the Token2Wav audio path is exercisable
    /// end-to-end.
    pub fn generate_codes_stub(&self, n_codes: usize, device: &Device) -> Result<Tensor> {
        // A handful of valid codebook indices (well inside [0, 8192)).
        let codes: Vec<i64> = (0..n_codes).map(|i| (17 * i % 8000) as i64).collect();
        Tensor::from_vec(codes, (1, n_codes), device)
    }

    /// End-to-end audio synthesis from codec tokens.
    ///
    /// Given codec tokens (from the Talker, or the stub) plus the
    /// speaker `conditioning` `(1, enc_emb_dim)` and `reference_mel`
    /// `(1, T_ref, 80)` of a chosen voice, run Token2Wav to produce a
    /// 24 kHz waveform `(1, 1, samples)`. This is the audio-producing
    /// half of `generate()`.
    pub fn codes_to_audio(
        &self,
        codes: &Tensor,
        conditioning: &Tensor,
        reference_mel: &Tensor,
    ) -> Result<Tensor> {
        self.token2wav
            .forward_default(codes, conditioning, reference_mel)
    }

    /// Load one speaker's voice conditioning from `spk_dict.pt`.
    ///
    /// `spk_dict.pt` is a PyTorch pickle of a nested dict
    /// `{speaker: {bos_token, cond: (1,192), ref_mel: (1,T_ref,80)}}`.
    /// candle's `candle_core::pickle::read_all` returns a flat
    /// `Vec<(String, Tensor)>` and does not decode arbitrarily-nested
    /// Python dicts (the per-speaker scalar `bos_token` in particular is
    /// not a tensor), so reconstructing the speaker map is a TODO.
    ///
    /// Until that lands, callers should supply synthetic (or otherwise
    /// externally-sourced) `conditioning` + `reference_mel` to
    /// [`Self::generate_speech`]. This returns a documented error rather
    /// than blocking the end-to-end build on the pickle decode.
    pub fn load_speaker(_path: &str, _name: &str) -> Result<(i64, Tensor, Tensor)> {
        candle::bail!(
            "load_speaker: spk_dict.pt nested-dict pickle decode is a TODO; \
             supply synthetic conditioning + reference_mel to generate_speech for now"
        )
    }

    /// End-to-end text → speech (sampled Talker AR, no KV cache).
    ///
    /// Mirrors `Qwen2_5OmniForConditionalGeneration.generate`
    /// (`modeling_qwen2_5_omni.py:3759-3982`) with the full-recompute
    /// simplification: the Thinker text is generated greedily first, then
    /// ONE full forward over `[prompt + generated]` captures every
    /// position's layer-0 embedding + final hidden state (causal masking
    /// makes this identical to step-by-step capture). The Talker then
    /// decodes codec tokens autoregressively from the fused embeddings,
    /// and Token2Wav renders them to a waveform.
    ///
    /// Arguments:
    /// - `input_ids` `(1, P)`: the text prompt (post-tokenization).
    /// - `bos_token`: the speaker's role/chat BOS id (from `spk_dict.pt`).
    /// - `conditioning` `(1, enc_emb_dim)`: speaker embedding for the DiT.
    /// - `reference_mel` `(1, T_ref, mel_dim)`: reference mel for the DiT.
    /// - `thinker_max_new` / `talker_max_new`: greedy decode budgets.
    ///
    /// Returns `(gen_text_ids (1, G), codes (1, n), waveform (1, 1, samples))`.
    /// The raw codec-token sequence is returned so callers can inspect it
    /// for degeneration (repeated token / collapse) and range-check it
    /// against the DiT codebook before claiming audio quality.
    ///
    /// Batch size 1 only (upstream `generate()` raises otherwise). All
    /// embeddings stay in the Thinker hidden space (2048) until
    /// `project_thinker` drops to the Talker hidden (896) at each step.
    /// `cancel`: optional barge-in flag, polled at every Talker AR step
    /// and between pipeline stages. When it flips true the function bails
    /// with a `"generation cancelled"` error (the caller discards the
    /// turn) — this is what lets a realtime session abort a long
    /// utterance the moment the user starts speaking.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speech(
        &mut self,
        input_ids: &Tensor,
        bos_token: i64,
        conditioning: &Tensor,
        reference_mel: &Tensor,
        thinker_max_new: usize,
        talker_max_new: usize,
        sampling: TalkerSampling,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // One-shot wrapper around the streaming path: collect every emitted
        // chunk into a single waveform. `chunk_codes = 0` → a single decode
        // over all codes, byte-identical to the historical behavior.
        let mut samples: Vec<f32> = Vec::new();
        let (gen_ids, codes_t) = self.generate_speech_streaming(
            input_ids,
            bos_token,
            conditioning,
            reference_mel,
            thinker_max_new,
            talker_max_new,
            sampling,
            cancel,
            0,
            &mut |chunk: &[f32]| {
                samples.extend_from_slice(chunk);
                Ok(())
            },
        )?;
        let n = samples.len();
        let wav = Tensor::from_vec(samples, (1, 1, n), gen_ids.device())?;
        Ok((gen_ids, codes_t, wav))
    }

    /// Streaming text→speech: identical pipeline to [`Self::generate_speech`]
    /// but audio is EMITTED INCREMENTALLY through `on_audio` instead of
    /// returned at the end — this is what takes a realtime session's
    /// first-audio latency from "whole utterance" to "first chunk".
    ///
    /// `chunk_codes` is the emission granularity in codec tokens: every
    /// `chunk_codes` sampled codes (50 ≈ 2 s — the DiT's native
    /// `seconds_per_chunk`), the window is decoded through Token2Wav and
    /// the NEW samples are pushed to `on_audio`. `0` disables chunking
    /// (single decode at the end).
    ///
    /// Chunked decodes use `CHUNK_CTX_CODES` of left context and a short
    /// crossfade at each seam: consecutive chunks are decoded with fresh
    /// flow-matching noise, so the boundary sample values differ slightly
    /// — the held-back tail of the previous chunk is linearly blended with
    /// the new decode's rendition of the same positions.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_speech_streaming(
        &mut self,
        input_ids: &Tensor,
        bos_token: i64,
        conditioning: &Tensor,
        reference_mel: &Tensor,
        thinker_max_new: usize,
        talker_max_new: usize,
        sampling: TalkerSampling,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        chunk_codes: usize,
        on_audio: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<(Tensor, Tensor)> {
        // ── 1. Thinker greedy text generation (stops on `<|im_end|>`). ─
        let gen_ids = self.thinker_reply(input_ids, thinker_max_new)?;
        let codes_t = self.speak_reply_streaming(
            input_ids,
            &gen_ids,
            bos_token,
            conditioning,
            reference_mel,
            talker_max_new,
            sampling,
            cancel,
            chunk_codes,
            on_audio,
        )?;
        Ok((gen_ids, codes_t))
    }

    /// Stage 1 alone: greedy Thinker text generation (stops on
    /// `<|im_end|>`). Exposed separately so callers can INSPECT the reply
    /// text between the stages — e.g. parse a `<tool_call>` block and skip
    /// (or trim) speech instead of reading JSON aloud — then feed the ids
    /// they actually want spoken to [`Self::speak_reply_streaming`].
    pub fn thinker_reply(&mut self, input_ids: &Tensor, thinker_max_new: usize) -> Result<Tensor> {
        self.thinker
            .generate_greedy(input_ids, thinker_max_new, THINKER_CHAT_EOS_TOKEN_ID)
    }

    /// Stages 2-6: speak `gen_ids` as the assistant reply to `input_ids`.
    /// `gen_ids` is usually [`Self::thinker_reply`]'s output, but ANY token
    /// sequence works — the fused hiddens are recomputed by one causal
    /// forward over `[prompt + gen_ids]`, so e.g. a tool-call preamble
    /// trimmed out of the raw reply speaks naturally.
    #[allow(clippy::too_many_arguments)]
    pub fn speak_reply_streaming(
        &mut self,
        input_ids: &Tensor,
        gen_ids: &Tensor,
        bos_token: i64,
        conditioning: &Tensor,
        reference_mel: &Tensor,
        talker_max_new: usize,
        sampling: TalkerSampling,
        cancel: Option<&std::sync::atomic::AtomicBool>,
        chunk_codes: usize,
        on_audio: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<Tensor> {
        use std::sync::atomic::Ordering;
        let cancelled =
            |c: Option<&std::sync::atomic::AtomicBool>| c.is_some_and(|f| f.load(Ordering::Relaxed));
        let (b, prompt_len) = input_ids.dims2()?;
        if b != 1 {
            candle::bail!("generate_speech: batch size must be 1, got {b}");
        }
        let device = input_ids.device().clone();

        // ── 2. ONE full forward over [prompt + generated], capturing
        //       layer-0 embeds + final hidden for ALL positions. ────────
        let full_ids = Tensor::cat(&[input_ids, gen_ids], D::Minus1)?;
        let (last_hidden, tok_embeds) = self.thinker.forward_collect(&full_ids)?;
        // ── 3. FUSION — prompt prefix + generated reply (upstream 3890–3976). ──
        //
        // CRITICAL: upstream `modeling_qwen2_5_omni.py` builds the Talker
        // prefill from the FULL PROMPT plus the first generated token:
        //
        //   talker_inputs_embeds = thinker_hidden_states[0] + thinker_token_embeds[0]
        //                        = fused(prompt[0..P])              (1, P, H)   line 3951
        //   talker_inputs_embeds = cat([fused(prompt), bos_embed,
        //                               thinker_reply_part[:, :1]])             lines 3954-3961
        //
        // `hidden_states[0]` is the PREFILL step's per-layer tuple — shape
        // (1, prompt_len, H) — NOT the first generated token. The reply
        // stream consumed per codec step is fused(gen[1:]) (line 3969:
        // `thinker_reply_part[:, 1:, :]` after the prefill takes `[:, :1]`).
        // Dropping the prompt prefix (the run-5/6 implementation) leaves the
        // Talker without the semantic context of what to say — it babbles
        // prosodically plausible but unintelligible codec streams.
        //
        // fused(x[i]) = last_layer_hidden(x[i]) + input_embed(x[i]). Causal
        // masking makes one full forward over [prompt + gen] equivalent to
        // upstream's incremental hidden-state capture, EXCEPT the final
        // generated token: upstream never forwards it (the loop stops after
        // sampling it), so its fused hidden never enters the reply stream.
        let prompt_fused = {
            let h = last_hidden.narrow(1, 0, prompt_len)?;
            let e = tok_embeds.narrow(1, 0, prompt_len)?;
            (h + e)?
        };
        let reply_len = last_hidden.dim(1)? - prompt_len;
        let reply = if reply_len > 0 {
            let h = last_hidden.narrow(1, prompt_len, reply_len)?;
            let e = tok_embeds.narrow(1, prompt_len, reply_len)?;
            (h + e)?
        } else {
            // G==0: Thinker emitted nothing. Synthesise a single fused EOS
            // embed so the Talker has exactly one token to start its prefill.
            let eos_ids = Tensor::from_vec(vec![TEXT_EOS_TOKEN_ID], (1, 1), &device)?;
            self.thinker.embed(&eos_ids)?
        };
        let reply_len = reply.dim(1)?;

        // ── 4. Assemble the Talker prefill `inputs_embeds` (2048-space). ─────
        // Upstream 3954-3961:
        //   cat([fused(prompt[0..P]), bos_embed, fused(gen[0])]) → (1, P+2, H)
        // The prompt prefix carries NO codec embedding (the codec_mask fill in
        // `talker_input_ids`, line 3942, only sizes the rope index — at prefill
        // upstream adds codec embeds to the LAST TWO positions only, 2354-2359).
        let tcfg = self.talker.config();
        let codec_bos = tcfg.tts_codec_start_token_id as i64;
        let codec_pad = tcfg.tts_codec_pad_token_id as i64;
        let codec_eos = tcfg.tts_codec_end_token_id as i64;

        // Speaker BOS embed via the THINKER embedding table (text-token space).
        let bos_ids = Tensor::from_vec(vec![bos_token], (1, 1), &device)?;
        let bos_embed = self.thinker.embed(&bos_ids)?; // (1,1,2048)

        let gen0 = reply.narrow(1, 0, 1)?; // fused(gen[0]) — always present
        let mut seq_embeds = Tensor::cat(&[&prompt_fused, &bos_embed, &gen0], 1)?; // (1, P+2, 2048)

        // Codec scaffold added ONCE at the last two positions (upstream 2354-2359):
        //   seq[-2] += embed_codec(codec_pad)   → lands on bos_embed
        //   seq[-1] += embed_codec(codec_bos)   → lands on fused(gen[0])
        // The prompt prefix positions get NO codec embedding.
        seq_embeds = {
            let len = seq_embeds.dim(1)?;
            let head = seq_embeds.narrow(1, 0, len - 2)?;
            let pad_id = Tensor::from_vec(vec![codec_pad], (1, 1), &device)?;
            let bos_codec_id = Tensor::from_vec(vec![codec_bos], (1, 1), &device)?;
            let pos_m2 = (seq_embeds.narrow(1, len - 2, 1)?
                + self.talker.embed_codec(&pad_id)?)?;
            let pos_m1 = (seq_embeds.narrow(1, len - 1, 1)?
                + self.talker.embed_codec(&bos_codec_id)?)?;
            Tensor::cat(&[&head, &pos_m2, &pos_m1], 1)?
        };

        // reply_stream: gen[1..G-1] then eos + pad (upstream 3969-3976).
        // gen[0] is consumed by the prefill above; the FINAL generated token
        // is excluded because upstream's generate loop never forwards it, so
        // its fused hidden never enters `thinker_reply_part` (G forwards
        // produce hiddens at gen[0..G-2] only — see §3 comment).
        let eos_ids = Tensor::from_vec(vec![TEXT_EOS_TOKEN_ID], (1, 1), &device)?;
        let pad_ids = Tensor::from_vec(vec![TEXT_PAD_TOKEN_ID], (1, 1), &device)?;
        let eos_embed = self.thinker.embed(&eos_ids)?;
        let pad_embed = self.thinker.embed(&pad_ids)?;
        let mut reply_stream = if reply_len > 2 {
            let rest = reply.narrow(1, 1, reply_len - 2)?;
            Tensor::cat(&[&rest, &eos_embed, &pad_embed], 1)?
        } else {
            Tensor::cat(&[&eos_embed, &pad_embed], 1)?
        };

        // ── 5. Talker AR loop (KV-cached incremental decode, sampled). ─
        // Sampling shape mirrors upstream `talker_*` GenerationConfig
        // (`modeling_qwen2_5_omni.py:3765-3771`): TopKThenTopP k=40/p=0.8
        // + temperature=0.9 + repetition_penalty=1.05. Greedy here
        // degenerates loudly — verified on Lambda (11 unique codes out
        // of 256 → not speech). The repetition penalty is applied to
        // logits (pre-softmax) over the already-emitted code history;
        // codec_bos is suppressed every step; eos on {pad, eos}.
        //
        // The prefill chunk runs once over (1, P+2, H); every later step
        // feeds ONE new embed with its absolute position — the per-layer
        // KV cache supplies the history (mathematically identical to the
        // previous full-recompute, O(N) instead of O(N²) forwards).
        self.talker.clear_kv_cache();
        let mut codes: Vec<i64> = Vec::with_capacity(talker_max_new);
        let mut processor = sampling.make_processor();
        let rep_penalty = sampling.repetition_penalty;
        // Streaming emission state: codes already decoded+emitted, plus the
        // held-back tail awaiting the next chunk's crossfade partner.
        let mut emitted_codes = 0usize;
        let mut held_tail: Vec<f32> = Vec::new();
        // First chunk = the assembled prefill; replaced by the 1-token
        // chunk after each sampled code. `pos_offset` tracks the absolute
        // position of the chunk's first token.
        let mut chunk = seq_embeds;
        let mut pos_offset = 0usize;
        for step in 0..talker_max_new {
            if cancelled(cancel) {
                self.talker.clear_kv_cache();
                candle::bail!("generation cancelled");
            }
            if step % 50 == 0 {
                eprintln!(
                    "real_weight: talker step {step}/{talker_max_new}, codes so far {}",
                    codes.len()
                );
            }
            let s = chunk.dim(1)?;
            let pos = super::mrope::text_only_position_ids(1, s, pos_offset, &device)?;
            let projected = self.talker.project_thinker(&chunk)?;
            let logits = self.talker.forward_from_projected_embeds(&projected, &pos)?;
            pos_offset += s;
            // Last-position logits, F32 on CPU. Apply repetition penalty +
            // suppress on the raw logits, then rebuild a tensor and let
            // `LogitsProcessor` do temperature + softmax + top-k + top-p.
            let last = logits.i((0, s - 1, ..))?;
            let mut lv = last.to_vec1::<f32>()?;
            if (rep_penalty - 1.0).abs() > 1e-6 {
                for &t in &codes {
                    if let Some(l) = lv.get_mut(t as usize) {
                        if *l > 0.0 {
                            *l /= rep_penalty;
                        } else {
                            *l *= rep_penalty;
                        }
                    }
                }
            }
            if (codec_bos as usize) < lv.len() {
                lv[codec_bos as usize] = f32::NEG_INFINITY;
            }
            let logits_1d = Tensor::from_vec(lv, (last.dim(0)?,), &device)?;
            let code = processor.sample(&logits_1d)? as i64;

            // Terminate on pad/eos — but never on the very first step, so
            // at least one code is always produced (token2wav needs ≥1).
            if step > 0 && (code == codec_pad || code == codec_eos) {
                break;
            }
            codes.push(code);

            // Streaming: a full chunk of fresh codes → decode + emit NOW,
            // while the AR loop continues. This is where first-audio
            // latency drops from whole-utterance to first-chunk.
            if chunk_codes > 0 && codes.len() - emitted_codes >= chunk_codes {
                self.decode_emit_chunk(
                    &codes,
                    &mut emitted_codes,
                    &mut held_tail,
                    conditioning,
                    reference_mel,
                    false,
                    &device,
                    on_audio,
                )?;
            }

            // next chunk = embed_codec(code) + reply_stream[0]  (2048-space)
            let code_t = Tensor::from_vec(vec![code], (1, 1), &device)?;
            let code_embed = self.talker.embed_codec(&code_t)?;
            let rs_head = reply_stream.narrow(1, 0, 1)?;
            chunk = (code_embed + rs_head)?;
            // Pop the front of the reply stream (keep the last slot when
            // it's down to one — the trailing pad embed repeats).
            let rs_len = reply_stream.dim(1)?;
            if rs_len > 1 {
                reply_stream = reply_stream.narrow(1, 1, rs_len - 1)?;
            }
        }
        self.talker.clear_kv_cache();

        // ── 6. Token2Wav: decode + emit whatever codes remain. ────────
        if cancelled(cancel) {
            candle::bail!("generation cancelled");
        }
        self.decode_emit_chunk(
            &codes,
            &mut emitted_codes,
            &mut held_tail,
            conditioning,
            reference_mel,
            true,
            &device,
            on_audio,
        )?;

        let n = codes.len();
        let codes_t = Tensor::from_vec(codes, (1, n), &device)?;
        Ok(codes_t)
    }

    /// Decode the not-yet-emitted codes through Token2Wav (with
    /// [`CHUNK_CTX_CODES`] of left context when this isn't the first
    /// chunk) and push the NEW samples to `on_audio`, crossfading the
    /// held-back tail of the previous chunk over [`CROSSFADE_SAMPLES`].
    /// On `is_final` the tail is flushed instead of held.
    #[allow(clippy::too_many_arguments)]
    fn decode_emit_chunk(
        &self,
        codes: &[i64],
        emitted_codes: &mut usize,
        held_tail: &mut Vec<f32>,
        conditioning: &Tensor,
        reference_mel: &Tensor,
        is_final: bool,
        device: &Device,
        on_audio: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<()> {
        let end = codes.len();
        if end == *emitted_codes {
            if is_final && !held_tail.is_empty() {
                let tail = std::mem::take(held_tail);
                on_audio(&tail)?;
            }
            return Ok(());
        }
        let ctx = (*emitted_codes).min(CHUNK_CTX_CODES);
        let window = &codes[*emitted_codes - ctx..end];
        let w = window.len();
        let window_t = Tensor::from_vec(window.to_vec(), (1, w), device)?;
        let wav = self
            .token2wav
            .forward_default(&window_t, conditioning, reference_mel)?;
        let samples = wav.flatten_all()?.to_vec1::<f32>()?;
        // Samples per code is fixed by the pipeline (2 mel frames × hop
        // 240 = 480 @ 24 kHz) but derive it so config changes can't skew
        // the seam arithmetic.
        let spc = samples.len() / w;
        let new_start = ctx * spc;

        // Crossfade: the previous chunk held back its last F samples;
        // this decode rendered the SAME positions at the end of its
        // context region — blend linearly.
        let f = held_tail.len();
        if f > 0 && new_start >= f {
            let alt = &samples[new_start - f..new_start];
            let blended: Vec<f32> = held_tail
                .iter()
                .zip(alt.iter())
                .enumerate()
                .map(|(i, (&a, &b))| {
                    let t = (i + 1) as f32 / (f + 1) as f32;
                    a * (1.0 - t) + b * t
                })
                .collect();
            on_audio(&blended)?;
            held_tail.clear();
        } else if f > 0 {
            // Context too short to re-render the held positions — emit
            // the held tail verbatim (no blend partner).
            let tail = std::mem::take(held_tail);
            on_audio(&tail)?;
        }

        let new = &samples[new_start..];
        if is_final {
            on_audio(new)?;
        } else {
            let hold = CROSSFADE_SAMPLES.min(new.len());
            on_audio(&new[..new.len() - hold])?;
            *held_tail = new[new.len() - hold..].to_vec();
        }
        *emitted_codes = end;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::{
        AudioEncoderConfig, BigVganConfig, DitConfig, OmniConfig, RopeScaling, TalkerConfig,
        ThinkerConfig, ThinkerTextConfig, Token2WavConfig, VisionEncoderConfig,
    };
    use candle::{Device, Tensor};
    use candle_nn::VarMap;

    /// A complete, trivially-small `OmniConfig` for the CPU end-to-end
    /// shape test. Each sub-config mirrors the structural invariants of
    /// the per-module `tiny_cfg()` helpers (which are private `#[cfg(test)]`
    /// fns, so their values are replicated here):
    ///
    /// - Thinker `hidden == Talker embedding_size` (the per-step
    ///   `embed_codec + reply_stream` add lives in Thinker space).
    /// - DiT `mel_dim == BigVGAN mel_dim` (DiT mel feeds BigVGAN).
    /// - M-RoPE sums: Thinker `[2,3,3]=8=16/2`, Talker `[2,2,0]=4=8/2`.
    /// - Thinker `vocab_size` must cover the text-special ids
    ///   (151859–151861) embedded into the Talker reply stream → keep the
    ///   real 151936.
    fn tiny_omni_cfg() -> OmniConfig {
        let text_config = ThinkerTextConfig {
            vocab_size: 151936, // must cover text_pad/eos ids (151859–151861)
            hidden_size: 64,    // == talker.embedding_size
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            rope_scaling: RopeScaling {
                mrope_section: vec![2, 3, 3], // sum 8 = head_dim(16)/2
            },
            head_dim: None,
        };
        // Audio / vision sub-configs are required by serde but never touched
        // by `Qwen2_5OmniModel::new` (it only builds thinker/talker/token2wav)
        // — populate with dummy-but-valid dims.
        let audio_config = AudioEncoderConfig {
            d_model: 16,
            num_hidden_layers: 1,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 32,
            num_mel_bins: 128,
            max_source_positions: 32,
            n_window: 4,
            output_dim: 64,
            scale_embedding: false,
        };
        let vision_config = VisionEncoderConfig {
            hidden_size: 16,
            depth: 1,
            num_heads: 2,
            intermediate_size: 32,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            in_channels: 3,
            out_hidden_size: 64,
            fullatt_block_indexes: vec![],
            window_size: 16,
            tokens_per_second: 25,
        };
        let thinker_config = ThinkerConfig {
            text_config,
            audio_config,
            vision_config,
            audio_token_index: 151646,
            image_token_index: 151655,
            video_token_index: 151656,
            vision_start_token_id: 151652,
            vision_end_token_id: 151653,
            bos_token_id: 151644,
            eos_token_id: 151645,
            pad_token_id: 151643,
        };
        let talker_config = TalkerConfig {
            hidden_size: 32,
            embedding_size: 64, // == thinker.hidden_size
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 48,
            vocab_size: 24, // codec head width (covers codec specials 18–20)
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
            rope_scaling: RopeScaling {
                mrope_section: vec![2, 2, 0], // sum 4 = head_dim(8)/2
            },
            position_id_per_seconds: 25,
            seconds_per_chunk: 2,
            tts_codec_start_token_id: 19,
            tts_codec_end_token_id: 20,
            tts_codec_pad_token_id: 18,
            tts_codec_mask_token_id: 22,
        };
        let dit_config = DitConfig {
            hidden_size: 32, // = head_dim(8)·heads(4)
            num_hidden_layers: 2,
            heads: 4,
            head_dim: 8,
            ff_mult: 2,
            mel_dim: 8, // == bigvgan.mel_dim
            emb_dim: 16,
            num_embeds: 31,
            repeats: 2,
            enc_dim: 12,
            enc_emb_dim: 6, // conditioning width
            enc_channels: vec![16, 16, 16, 16, 48],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 2,
            enc_se_channels: 8,
            enc_lin_neurons: 6,
            dropout: 0.0,
            block_size: 4,
            look_ahead_layers: vec![1],
            look_backward_layers: vec![0],
        };
        let bigvgan_config = BigVganConfig {
            mel_dim: 8, // == dit.mel_dim
            upsample_initial_channel: 64,
            upsample_rates: vec![5, 3, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![11, 7, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_bias_at_final: false,
        };
        OmniConfig {
            thinker_config,
            talker_config,
            token2wav_config: Token2WavConfig {
                dit_config,
                bigvgan_config,
            },
            enable_audio_output: true,
            enable_talker: true,
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.02, v.as_tensor().shape().clone(), device)?)?;
        }
        Ok(())
    }

    /// **Phase 6 acceptance #1** — the fusion is an elementwise add, not
    /// a concat. `(a, b) → a + b`, shape preserved. This pins the single
    /// highest-risk port detail.
    #[test]
    fn fusion_is_elementwise_add() -> Result<()> {
        let dev = Device::Cpu;
        let h = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), &dev)?;
        let e = Tensor::from_vec(vec![10.0f32, 20.0, 30.0, 40.0], (1, 2, 2), &dev)?;
        let fused = Qwen2_5OmniModel::fuse_for_talker(&h, &e)?;
        assert_eq!(fused.dims(), &[1, 2, 2]);
        assert_eq!(fused.flatten_all()?.to_vec1::<f32>()?, vec![11.0, 22.0, 33.0, 44.0]);
        Ok(())
    }

    /// **Phase 6 acceptance #2** — real-weight end-to-end audio smoke
    /// (CPU F32). Loads the FULL released Qwen2.5-Omni-3B (all three
    /// stages, asserts no missing keys), then runs the stub codes →
    /// Token2Wav → waveform path with synthetic speaker conditioning,
    /// asserting the audio is finite + in [-1, 1].
    ///
    /// Lambda-only (`#[ignore]`). Requires `QWEN_OMNI_3B_DIR`.
    #[test]
    #[ignore]
    fn real_weight_omni_loads() {
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");

        let shards: Vec<_> = std::fs::read_dir(&model_dir)
            .expect("list model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        assert!(!shards.is_empty(), "no safetensors shards in {model_dir:?}");

        // Whole model F32 (Token2Wav requires it; simplest correct path).
        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let model = Qwen2_5OmniModel::new(&cfg, vb).expect("construct full Omni model");

        // Stub codes → audio. Synthetic speaker conditioning + reference
        // mel (a real run would pull these from spk_dict.pt). Keep codes
        // tiny so the (forced-fp32, CPU) DiT sampling stays fast.
        let n_codes = 4;
        let codes = model.generate_codes_stub(n_codes, &device).unwrap();
        let enc_emb_dim = cfg.token2wav_config.dit_config.enc_emb_dim;
        let conditioning = Tensor::randn(0f32, 1f32, (1, enc_emb_dim), &device).unwrap();
        let reference_mel = Tensor::randn(
            0f32,
            1f32,
            (1, 32, cfg.token2wav_config.dit_config.mel_dim),
            &device,
        )
        .unwrap();

        // Use 2 ODE steps to keep the smoke fast — quality is a Phase 6.5
        // concern; this gate is "audio comes out, finite, in range".
        let wav = model
            .token2wav()
            .forward(&codes, &conditioning, &reference_mel, 2, 0.5, -1.0)
            .expect("token2wav forward");
        let dims = wav.dims().to_vec();
        eprintln!("real_weight_omni_loads: waveform dims = {dims:?}");
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        let samples = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(samples.iter().all(|x| x.is_finite()), "non-finite audio");
        assert!(
            samples.iter().all(|x| *x >= -1.001 && *x <= 1.001),
            "audio out of [-1, 1]"
        );

        // Also exercise the Thinker text path on the same loaded model.
        let input_ids = Tensor::from_vec(vec![151644i64, 151645, 100, 200], (1, 4), &device).unwrap();
        let logits = model
            .thinker()
            .forward_text_only(&input_ids, 0)
            .expect("thinker forward");
        assert_eq!(logits.dim(2).unwrap(), model.thinker().config().vocab_size);
    }

    /// **Phase 6.5 acceptance** — end-to-end `generate_speech` produces a
    /// finite waveform from a tiny random-weight model. Exercises the full
    /// pipeline: Thinker greedy → `forward_collect` → ADD fusion → Talker
    /// prefill (codec scaffold at -2/-1) → Talker AR decode → Token2Wav.
    /// Greedy may hit eos early; the ≥1-code floor (no break on step 0)
    /// guarantees Token2Wav receives at least one code.
    #[test]
    fn generate_speech_tiny_shapes() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_omni_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let mut model = Qwen2_5OmniModel::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let input_ids = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
        let enc_emb_dim = cfg.token2wav_config.dit_config.enc_emb_dim;
        let mel_dim = cfg.token2wav_config.dit_config.mel_dim;
        let conditioning = Tensor::randn(0f32, 1f32, (1, enc_emb_dim), &device)?;
        let reference_mel = Tensor::randn(0f32, 1f32, (1, 16, mel_dim), &device)?;

        let (text_ids, _codes, wav) = model.generate_speech(
            &input_ids,
            1, // bos token
            &conditioning,
            &reference_mel,
            4, // thinker_max_new
            6, // talker_max_new
            TalkerSampling::GREEDY,
            None,
        )?;

        // Text ids: (1, G) with 0 <= G <= thinker_max_new.
        assert_eq!(text_ids.dim(0)?, 1);
        assert!(text_ids.dim(1)? <= 4);

        // Waveform: (1, 1, >0), all finite.
        let dims = wav.dims().to_vec();
        eprintln!("generate_speech_tiny_shapes: waveform dims = {dims:?}");
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        assert!(dims[2] > 0, "empty waveform");
        let samples = wav.flatten_all()?.to_vec1::<f32>()?;
        assert!(samples.iter().all(|x| x.is_finite()), "non-finite audio");
        Ok(())
    }

    /// **Streaming acceptance** — chunked emission must (a) fire the
    /// callback more than once when codes exceed the chunk size, (b)
    /// produce exactly as many total samples as the one-shot path for
    /// the same greedy generation (values differ across decodes — fresh
    /// flow-matching noise per chunk — but the seam arithmetic must not
    /// drop or duplicate a single sample), and (c) stay finite.
    #[test]
    fn generate_speech_streaming_chunks_tiny() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_omni_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let mut model = Qwen2_5OmniModel::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let input_ids = Tensor::from_vec(vec![1i64, 2, 3], (1, 3), &device)?;
        let enc_emb_dim = cfg.token2wav_config.dit_config.enc_emb_dim;
        let mel_dim = cfg.token2wav_config.dit_config.mel_dim;
        let conditioning = Tensor::randn(0f32, 1f32, (1, enc_emb_dim), &device)?;
        let reference_mel = Tensor::randn(0f32, 1f32, (1, 16, mel_dim), &device)?;

        // One-shot reference (greedy → deterministic codes).
        let (_, codes_a, wav) = model.generate_speech(
            &input_ids,
            1,
            &conditioning,
            &reference_mel,
            4,
            6,
            TalkerSampling::GREEDY,
            None,
        )?;
        let oneshot_len = wav.dim(2)?;

        // Chunked: 2 codes per emission.
        let mut emissions = 0usize;
        let mut streamed: Vec<f32> = Vec::new();
        let (_, codes_b) = model.generate_speech_streaming(
            &input_ids,
            1,
            &conditioning,
            &reference_mel,
            4,
            6,
            TalkerSampling::GREEDY,
            None,
            2,
            &mut |chunk: &[f32]| {
                emissions += 1;
                streamed.extend_from_slice(chunk);
                Ok(())
            },
        )?;

        assert_eq!(
            codes_a.dims(),
            codes_b.dims(),
            "greedy code count must match across modes"
        );
        let n_codes = codes_b.dim(1)?;
        if n_codes > 2 {
            assert!(emissions > 1, "expected multiple emissions, got {emissions}");
        }
        assert_eq!(
            streamed.len(),
            oneshot_len,
            "chunked emission dropped or duplicated samples"
        );
        assert!(streamed.iter().all(|x| x.is_finite()), "non-finite audio");
        Ok(())
    }

    /// **Phase 6.5 real-weight smoke** (Lambda-only, `#[ignore]`). Loads
    /// the full released Qwen2.5-Omni-3B at CPU F32 and runs the real
    /// `generate_speech` path with synthetic speaker conditioning,
    /// asserting a finite waveform comes out. Requires `QWEN_OMNI_3B_DIR`.
    #[test]
    #[ignore]
    fn generate_speech_real_weight() {
        use std::path::PathBuf;

        let model_dir = std::env::var("QWEN_OMNI_3B_DIR")
            .expect("QWEN_OMNI_3B_DIR must point at the local Qwen2.5-Omni-3B snapshot");
        let model_dir = PathBuf::from(model_dir);
        let cfg_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config.json");
        let cfg: OmniConfig = serde_json::from_str(&cfg_text).expect("parse omni config");

        let shards: Vec<_> = std::fs::read_dir(&model_dir)
            .expect("list model dir")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|e| e.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        assert!(!shards.is_empty(), "no safetensors shards in {model_dir:?}");

        // Use CUDA when available (compiled with --features cuda on Lambda);
        // falls back to CPU for local non-CUDA builds. 3B F32 needs ~12 GB
        // VRAM — fine on A100. CPU mode would time out for longer utterances
        // due to the O(N²) full-recompute Talker AR loop (lessons.md #9).
        let device = Device::cuda_if_available(0).expect("device init");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let mut model =
            Qwen2_5OmniModel::new(&cfg, vb).expect("construct full Omni model");

        // ── Real speaker conditioning + tokenized prompt. ──────────────
        // A Python preprocessing step on the box flattens spk_dict.pt's
        // "Chelsie" entry → `omni_spk.safetensors` {cond:(1,192),
        // ref_mel:(1,T,80)} and tokenizes a TTS prompt (proper system
        // prompt + chat template) → `omni_prompt.safetensors` {input_ids:
        // (1,P) i64}. This sidesteps candle's lack of nested-pickle decode
        // and gives a run that tests INTELLIGIBLE speech, not noise.
        let spk_path = std::env::var("QWEN_OMNI_SPK")
            .expect("QWEN_OMNI_SPK must point at omni_spk.safetensors (cond + ref_mel)");
        let prompt_path = std::env::var("QWEN_OMNI_PROMPT")
            .expect("QWEN_OMNI_PROMPT must point at omni_prompt.safetensors (input_ids)");
        let bos_token: i64 = std::env::var("QWEN_OMNI_BOS")
            .expect("QWEN_OMNI_BOS must be Chelsie's bos_token int")
            .parse()
            .expect("QWEN_OMNI_BOS must parse as i64");

        let spk = candle::safetensors::load(&spk_path, &device).expect("load omni_spk.safetensors");
        let conditioning = spk
            .get("cond")
            .expect("omni_spk.safetensors must contain `cond`")
            .to_dtype(DType::F32)
            .unwrap();
        let reference_mel = spk
            .get("ref_mel")
            .expect("omni_spk.safetensors must contain `ref_mel`")
            .to_dtype(DType::F32)
            .unwrap();
        let prompt =
            candle::safetensors::load(&prompt_path, &device).expect("load omni_prompt.safetensors");
        let input_ids = prompt
            .get("input_ids")
            .expect("omni_prompt.safetensors must contain `input_ids`")
            .to_dtype(DType::I64)
            .unwrap();
        eprintln!(
            "real_weight: prompt {:?}, cond {:?}, ref_mel {:?}, bos={bos_token}",
            input_ids.dims(),
            conditioning.dims(),
            reference_mel.dims()
        );

        // Real generation budgets: a coherent Thinker reply + enough codec
        // tokens for a short utterance.
        let thinker_max_new: usize = std::env::var("QWEN_OMNI_THINKER_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(128);
        let talker_max_new: usize = std::env::var("QWEN_OMNI_TALKER_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);

        // Sampling shape: env-overridable so we can A/B greedy vs the
        // upstream defaults from the Lambda runner. Default is the
        // upstream `talker_*` GenerationConfig (k=40, p=0.8, T=0.9,
        // rep=1.05) — the only known-working shape for codec decode.
        let seed: u64 = std::env::var("QWEN_OMNI_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let greedy = std::env::var("QWEN_OMNI_GREEDY").ok().is_some();
        let sampling = if greedy {
            TalkerSampling::GREEDY
        } else {
            TalkerSampling {
                seed,
                ..TalkerSampling::default()
            }
        };
        eprintln!("real_weight: sampling = {sampling:?}");

        let (text_ids, codes, wav) = model
            .generate_speech(
                &input_ids,
                bos_token,
                &conditioning,
                &reference_mel,
                thinker_max_new,
                talker_max_new,
                sampling,
                None,
            )
            .expect("generate_speech on real weights");

        // ── Diagnostics: inspect codes for degeneration + range. ───────
        let code_vec = codes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        let n = code_vec.len();
        // DiT embeds via `Embedding(num_embeds + 1)`, so valid code
        // indices are [0, num_embeds] inclusive.
        let cb = cfg.token2wav_config.dit_config.num_embeds as i64 + 1;
        let cmin = code_vec.iter().copied().min().unwrap_or(0);
        let cmax = code_vec.iter().copied().max().unwrap_or(0);
        let unique: std::collections::BTreeSet<i64> = code_vec.iter().copied().collect();
        let head: Vec<i64> = code_vec.iter().copied().take(32).collect();
        eprintln!(
            "real_weight: {n} codes, range [{cmin},{cmax}], {} unique, head={head:?}",
            unique.len()
        );
        eprintln!(
            "real_weight: text_ids {:?}, waveform {:?}",
            text_ids.dims(),
            wav.dims()
        );
        // Decodable offline: paste into tokenizer.decode to read the reply.
        let text_vec = text_ids.flatten_all().unwrap().to_vec1::<i64>().unwrap();
        eprintln!("real_weight: thinker reply ids = {text_vec:?}");
        // Clamp assert: a valid code must index the DiT codebook, else the
        // panic would be an opaque embedding-lookup OOB. cb==0 means the
        // config field is absent — skip the bound in that case.
        if cb > 0 {
            assert!(
                cmax < cb,
                "code {cmax} >= codebook_size {cb} — talker emitted an out-of-range token"
            );
        }
        // Degeneration smell test (log, don't fail): greedy codec decode
        // can collapse to one repeated token. If unique<=2 the next
        // iteration should switch to top-k/top-p/temperature sampling.
        if unique.len() <= 2 {
            eprintln!(
                "real_weight: WARNING greedy looks degenerate ({} unique codes) — try sampling",
                unique.len()
            );
        }

        let dims = wav.dims().to_vec();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        assert!(dims[2] > 0, "empty waveform");
        let samples = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(samples.iter().all(|x| x.is_finite()), "non-finite audio");

        // ── Write a 24 kHz mono PCM-16 WAV for astats + human listening. ─
        if let Ok(out) = std::env::var("QWEN_OMNI_WAV_OUT") {
            write_wav_pcm16(&out, &samples, 24_000).expect("write wav");
            eprintln!("real_weight: wrote {} ({} samples) to {out}", samples.len(), samples.len());
        }

        // ── Export the raw codec tokens for cross-decoding diagnostics. ─
        // The Python reference Token2Wav can decode these directly: if the
        // result is intelligible the Talker is correct and OUR Token2Wav
        // is the bug; if it's garbled the codes themselves are bad.
        if let Ok(out) = std::env::var("QWEN_OMNI_CODES_OUT") {
            candle::safetensors::save(
                &std::collections::HashMap::from([("codes".to_string(), codes.clone())]),
                &out,
            )
            .expect("write codes safetensors");
            eprintln!("real_weight: wrote {n} codes to {out}");
        }
    }

    /// Minimal 16-bit PCM mono WAV writer (no external deps) so the Lambda
    /// run can emit a file for `ffmpeg astats` + listening. Samples are
    /// peak-normalised to ≤ -1 dBFS (lessons.md #7) then scaled to i16.
    fn write_wav_pcm16(path: &str, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
        use std::io::Write;
        let n = samples.len() as u32;
        let byte_rate = sample_rate * 2;
        let data_len = n * 2;
        // Peak-normalise to ≤ -1 dBFS (lessons.md #7): scale down whenever the
        // loudest sample exceeds 0.891 (≈ -1 dBFS). This covers both:
        //   (a) BigVGAN outputs that clip above ±1.0 (run 4: 27 samples)
        //   (b) BigVGAN outputs in (0.891, 1.0] that land at 0 dBFS after i16
        //       conversion (run 6: peak exactly = 1.0 → Peak level dB: 0.000).
        // No amplification when peak < 0.891 so near-silent clips aren't boosted.
        let peak = samples.iter().map(|x| x.abs()).fold(0.0_f32, f32::max);
        let scale = if peak > 0.891 { peak / 0.891 } else { 1.0 };
        let mut f = std::fs::File::create(path)?;
        f.write_all(b"RIFF")?;
        f.write_all(&(36 + data_len).to_le_bytes())?;
        f.write_all(b"WAVE")?;
        f.write_all(b"fmt ")?;
        f.write_all(&16u32.to_le_bytes())?; // fmt chunk size
        f.write_all(&1u16.to_le_bytes())?; // PCM
        f.write_all(&1u16.to_le_bytes())?; // mono
        f.write_all(&sample_rate.to_le_bytes())?;
        f.write_all(&byte_rate.to_le_bytes())?;
        f.write_all(&2u16.to_le_bytes())?; // block align
        f.write_all(&16u16.to_le_bytes())?; // bits per sample
        f.write_all(b"data")?;
        f.write_all(&data_len.to_le_bytes())?;
        for &x in samples {
            let v = ((x / scale).clamp(-1.0, 1.0) * 32767.0) as i16;
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}

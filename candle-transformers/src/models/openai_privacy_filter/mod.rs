//! `openai/privacy-filter` and its derivatives (e.g. `OpenMed/privacy-filter-nemotron`).
//!
//! A 1.4B-parameter sparse Mixture-of-Experts model with a token-classification
//! head, bespoke to OpenAI. The base model is autoregressive; the LM head is
//! replaced by a 221-class BIOES head over 55 PII categories. Inference uses
//! constrained Viterbi decoding (see [`viterbi`]).
//!
//! Architecture summary (see model `config.json`):
//! - 8 decoder layers, hidden 640, intermediate 640
//! - GQA: 14 query heads, 2 KV heads, head_dim 64 (note: head_dim×heads ≠ hidden)
//! - Sparse MoE per layer: 128 experts, top-4 routing
//! - YaRN RoPE: factor=32 over original_max=4096, theta=150_000
//! - Sliding-window attention: 128 tokens
//! - `attention_bias=true` (Q/K/V/O projections all carry bias)
//! - SiLU activation, RMSNorm pre-norm
//! - bf16 weights, o200k_base tokenizer (200,064 vocab)

pub mod attention;
pub mod moe;
pub mod token_classifier;
pub mod viterbi;
pub mod yarn_rope;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm};

use attention::Attention;
use moe::SparseMoeBlock;
use yarn_rope::YarnRotaryEmbedding;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub original_max_position_embeddings: usize,
    #[serde(default)]
    pub truncate: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: usize,
    pub rope_theta: f64,
    pub rope_scaling: Option<RopeScaling>,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub classifier_dropout: f64,
    pub tie_word_embeddings: bool,
    // MoE
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub router_aux_loss_coef: f64,
    #[serde(default)]
    pub output_router_logits: bool,
    pub initializer_range: f64,
    // Token classifier head
    pub num_labels: usize,
}

#[derive(Debug, Clone)]
pub struct DecoderLayer {
    self_attn: Attention,
    moe: SparseMoeBlock,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(
        cfg: &Config,
        rotary: Arc<YarnRotaryEmbedding>,
        layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary, layer_idx, vb.pp("self_attn"))?;
        let moe = SparseMoeBlock::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            crate::models::with_tracing::RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = crate::models::with_tracing::RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            moe,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attn_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs2 = self.post_attention_layernorm.forward(&xs)?;
        let xs2 = self.moe.forward(&xs2)?;
        residual + xs2
    }
}

#[derive(Debug, Clone)]
pub struct Backbone {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: usize,
    dtype: DType,
    device: Device,
}

impl Backbone {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dtype = vb.dtype();
        let device = vb.device().clone();
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let rotary = Arc::new(YarnRotaryEmbedding::new(dtype, cfg, &device)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary.clone(),
                idx,
                vb_l.pp(idx),
            )?);
        }
        let norm = crate::models::with_tracing::RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            dtype,
            device,
        })
    }

    /// Build a sliding-window attention mask. Bidirectional inside the
    /// window; the mask is 0.0 for attended positions and -inf otherwise.
    /// Result shape: `(seq_len, seq_len)`.
    fn sliding_window_mask(&self, seq_len: usize) -> Result<Tensor> {
        let w = self.sliding_window as i64;
        let mut data = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for i in 0..seq_len {
            let lo = (i as i64 - w).max(0) as usize;
            let hi = ((i as i64 + w + 1) as usize).min(seq_len);
            for j in lo..hi {
                data[i * seq_len + j] = 0.0;
            }
        }
        let m = Tensor::from_vec(data, (seq_len, seq_len), &self.device)?.to_dtype(self.dtype)?;
        Ok(m)
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let mask = if seq_len > 1 {
            Some(self.sliding_window_mask(seq_len)?)
        } else {
            None
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref())?;
        }
        self.norm.forward(&xs)
    }
}

/// Top-level entry point: backbone + linear classification head.
/// Returns logits of shape `(batch, seq_len, num_labels)`.
#[derive(Debug, Clone)]
pub struct OpenAIPrivacyFilterForTokenClassification {
    backbone: Backbone,
    classifier: Linear,
}

impl OpenAIPrivacyFilterForTokenClassification {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        // HF saves the backbone under `model.` and the head at the top-level `classifier.`.
        let backbone = Backbone::new(cfg, vb.pp("model"))?;
        let classifier = linear_no_bias(cfg.hidden_size, cfg.num_labels, vb.pp("classifier"))?;
        Ok(Self {
            backbone,
            classifier,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.backbone.forward(input_ids)?;
        hidden.apply(&self.classifier)
    }
}

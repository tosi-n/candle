//! GQA self-attention with bias and bidirectional sliding-window mask.
//!
//! Differs from `qwen3` in three ways:
//! 1. No q_norm/k_norm (privacy-filter doesn't ship those tensors)
//! 2. Bias is on by default for Q/K/V/O (`attention_bias=true`)
//! 3. The attention is bidirectional inside the sliding window — there
//!    is no causal mask, since the model is used as a token classifier.

use candle::{DType, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

use super::yarn_rope::YarnRotaryEmbedding;
use super::Config;
use crate::models::with_tracing::{linear_b, Linear};
use crate::utils::repeat_kv;

#[derive(Debug, Clone)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary: Arc<YarnRotaryEmbedding>,
    scale: f64,
}

impl Attention {
    pub fn new(
        cfg: &Config,
        rotary: Arc<YarnRotaryEmbedding>,
        _layer_idx: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let bias = cfg.attention_bias;

        let q_proj = linear_b(cfg.hidden_size, q_dim, bias, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.hidden_size, kv_dim, bias, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.hidden_size, kv_dim, bias, vb.pp("v_proj"))?;
        let o_proj = linear_b(q_dim, cfg.hidden_size, bias, vb.pp("o_proj"))?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary,
            scale,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, q_len, _h) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // (b, seq, heads, head_dim) → (b, heads, seq, head_dim)
        let q = q
            .reshape((b, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary.apply(&q, &k, 0)?;

        // Repeat KV heads up to num_heads to match Q.
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        let scores = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let scores = match mask {
            Some(m) => scores.broadcast_add(m)?,
            None => scores,
        };
        let probs = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(v.dtype())?;
        let ctx = probs.matmul(&v)?;
        let ctx = ctx
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, q_len, self.num_heads * self.head_dim))?;
        // Cast to dtype expected by the output projection.
        let ctx = ctx.to_dtype(DType::F32).unwrap_or(ctx);
        self.o_proj.forward(&ctx)
    }
}

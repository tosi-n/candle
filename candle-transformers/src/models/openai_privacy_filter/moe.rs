//! Sparse MoE block: top-K routing over N expert MLPs.
//!
//! Layout: `mlp.gate.weight`, `mlp.experts.{i}.{gate,up,down}_proj.weight`.
//! Routing: softmax over gate logits, take top-K expert indices, optionally
//! renormalize so the kept weights sum to 1, then expert-dispatch.

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};

use super::Config;
use crate::models::with_tracing::{linear_no_bias, Linear};

#[derive(Debug, Clone)]
struct Expert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act: Activation,
}

impl Expert {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("down_proj"),
            )?,
            act: cfg.hidden_act,
        })
    }
}

impl Module for Expert {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct SparseMoeBlock {
    gate: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
}

impl SparseMoeBlock {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_local_experts, vb.pp("gate"))?;
        let mut experts = Vec::with_capacity(cfg.num_local_experts);
        let vb_e = vb.pp("experts");
        for i in 0..cfg.num_local_experts {
            experts.push(Expert::new(cfg, vb_e.pp(i))?);
        }
        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
        })
    }
}

impl Module for SparseMoeBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let xs = xs.reshape(((), hidden))?;

        let logits = xs.apply(&self.gate)?;
        let weights = candle_nn::ops::softmax_last_dim(&logits)?;

        // Top-K experts per token.
        let topk_idx = weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let topk_w = weights.gather(&topk_idx, D::Minus1)?;

        let topk_w = topk_w.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let topk_idx = topk_idx.to_vec2::<u32>()?;

        // Bucket tokens per expert.
        let mut tokens_per_expert: Vec<Vec<u32>> = vec![Vec::new(); self.experts.len()];
        let mut weights_per_expert: Vec<Vec<f32>> = vec![Vec::new(); self.experts.len()];

        for (row, (ws, idxs)) in topk_w.iter().zip(topk_idx.iter()).enumerate() {
            // Renormalize so kept weights sum to 1 (numerical stability).
            let sum: f32 = ws.iter().sum::<f32>().max(1e-9);
            for (&w, &e) in ws.iter().zip(idxs.iter()) {
                tokens_per_expert[e as usize].push(row as u32);
                weights_per_expert[e as usize].push(w / sum);
            }
        }

        let mut out = xs.zeros_like()?;
        for (i, expert) in self.experts.iter().enumerate() {
            let toks = &tokens_per_expert[i];
            if toks.is_empty() {
                continue;
            }
            let idx = Tensor::new(toks.as_slice(), xs.device())?;
            let w = Tensor::new(weights_per_expert[i].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            let inputs = xs.index_select(&idx, 0)?;
            let outputs = expert.forward(&inputs)?;
            let outputs = outputs.broadcast_mul(&w)?;
            out = out.index_add(&idx, &outputs, 0)?;
        }

        out.reshape((b, seq, hidden))
    }
}

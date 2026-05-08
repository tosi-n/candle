//! YaRN (Yet another RoPE extensioN) rotary embedding.
//!
//! Reference: https://arxiv.org/abs/2309.00071
//!
//! For dimensions whose effective frequency falls outside the original
//! context window, the inverse frequencies are interpolated rather than
//! extrapolated. A linear ramp blends extrapolation (high-freq) and
//! interpolation (low-freq) zones, controlled by `beta_fast`/`beta_slow`.
//! A magnitude scaling factor `mscale = 0.1 * ln(factor) + 1` is applied
//! to the cos/sin tensors so attention scores stay in a comparable range
//! after the longer effective context.

use candle::{DType, Device, Result, Tensor};

use super::Config;

#[derive(Debug, Clone)]
pub struct YarnRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl YarnRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let theta = cfg.rope_theta;

        let (factor, beta_fast, beta_slow, original_max) = match &cfg.rope_scaling {
            Some(s) if s.rope_type == "yarn" => (
                s.factor,
                s.beta_fast,
                s.beta_slow,
                s.original_max_position_embeddings,
            ),
            _ => (1.0, 32.0, 1.0, max_seq_len),
        };

        // Per-dim base frequencies in cycles per token (extrapolation, i.e.
        // standard RoPE) and interpolated (scaled by 1/factor).
        let half = dim / 2;
        let mut inv_freq_extra = Vec::with_capacity(half);
        let mut inv_freq_interp = Vec::with_capacity(half);
        for i in 0..half {
            let pos = (2 * i) as f64;
            let pos_freq = theta.powf(pos / dim as f64);
            inv_freq_extra.push(1.0 / pos_freq);
            inv_freq_interp.push(1.0 / (factor * pos_freq));
        }

        // Ramp boundaries in dim-space.
        let log_theta = theta.ln();
        let low = (dim as f64 * (original_max as f64 / (beta_fast * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * log_theta))
            .floor()
            .max(0.0);
        let high = (dim as f64 * (original_max as f64 / (beta_slow * 2.0 * std::f64::consts::PI)).ln()
            / (2.0 * log_theta))
            .ceil()
            .min((dim - 1) as f64);
        let denom = (high - low).max(1e-3);

        // 0 in interp zone, 1 in extrap zone, linear ramp between.
        let mut blended = Vec::with_capacity(half);
        for i in 0..half {
            let r = ((i as f64 - low) / denom).clamp(0.0, 1.0);
            blended.push(inv_freq_interp[i] * (1.0 - r) + inv_freq_extra[i] * r);
        }

        let mscale = if factor > 1.0 {
            0.1 * factor.ln() + 1.0
        } else {
            1.0
        };

        let inv_freq = Tensor::from_vec(
            blended.iter().map(|x| *x as f32).collect::<Vec<_>>(),
            (1, half),
            dev,
        )?
        .to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = (freqs.cos()? * mscale)?.to_dtype(dtype)?;
        let sin = (freqs.sin()? * mscale)?.to_dtype(dtype)?;
        Ok(Self { sin, cos })
    }

    /// q, k shape: `(batch, num_heads, seq_len, head_dim)`.
    pub fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

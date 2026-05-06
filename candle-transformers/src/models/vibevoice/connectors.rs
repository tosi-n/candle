//! Speech connectors — small Linear → RMSNorm → Linear blocks that
//! project tokenizer latents into the LLM's hidden-state space (or a
//! near-identity transform for already-aligned features).
//!
//! Mirrors `SpeechConnector` in
//! `vibevoice/modular/modeling_vibevoice.py` (lines 59–70 upstream):
//!
//! ```python
//! class SpeechConnector(nn.Module):
//!     def __init__(self, input_dim, output_dim):
//!         super().__init__()
//!         self.fc1  = nn.Linear(input_dim, output_dim)
//!         self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
//!         self.fc2  = nn.Linear(output_dim, output_dim)
//!
//!     def forward(self, features, **kwargs):
//!         x = self.fc1(features)
//!         x = self.norm(x)
//!         x = self.fc2(x)
//!         return x
//! ```
//!
//! Used twice in the parent `VibeVoiceModel`:
//!
//! - `acoustic_connector`: maps `(B, T, acoustic_vae_dim=64)` continuous
//!   latents into `(B, T, lm_hidden)` so they can be spliced into the
//!   LLM's `inputs_embeds` (model.py L128, L391).
//! - `semantic_connector`: same shape contract, fed from the semantic
//!   tokenizer's encoder output (model.py L129, L361).

use candle::{DType, Result, Tensor, D};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// LLaMA-flavoured RMSNorm — matches the Python `LlamaRMSNorm`:
///
/// ```python
/// x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
/// ```
///
/// Same algebra as the diffusion-head RMSNorm but kept local to avoid
/// reaching into a sibling module for a private helper.
#[derive(Debug, Clone)]
struct ConnectorRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ConnectorRmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_rms = (variance + self.eps)?.sqrt()?.recip()?;
        let normalized = x.broadcast_mul(&inv_rms)?;
        let scaled = normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        scaled.to_dtype(in_dtype)
    }
}

/// `SpeechConnector` — projects audio-tokenizer latents into the LLM's
/// hidden-state space.
#[derive(Debug, Clone)]
pub struct SpeechConnector {
    fc1: Linear,
    norm: ConnectorRmsNorm,
    fc2: Linear,
}

impl SpeechConnector {
    pub fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(input_dim, output_dim, vb.pp("fc1"))?;
        let norm = ConnectorRmsNorm::new(output_dim, 1e-6, vb.pp("norm"))?;
        let fc2 = linear(output_dim, output_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, norm, fc2 })
    }

    /// Forward — `(.., input_dim)` → `(.., output_dim)`. Final-axis
    /// linear, so any rank-2 or rank-3 input is fine.
    pub fn forward(&self, features: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(features)?;
        let x = self.norm.forward(&x)?;
        self.fc2.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};
    use candle_nn::VarMap;

    /// The connector loads under the standard `fc1 / norm / fc2` weight
    /// layout used in the released checkpoint and round-trips a
    /// (B, T, input_dim) → (B, T, output_dim) tensor.
    #[test]
    fn connectors_load_with_correct_keys() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let input_dim = 8;
        let output_dim = 16;
        let conn = SpeechConnector::new(input_dim, output_dim, vb.pp("acoustic_connector"))?;

        let x = Tensor::randn(0.0_f32, 1.0, (2, 3, input_dim), &device)?;
        let y = conn.forward(&x)?;
        assert_eq!(y.dims(), &[2, 3, output_dim]);
        Ok(())
    }
}

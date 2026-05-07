//! Diffusion head for VibeVoice — DiT-style denoiser that predicts noise
//! / velocity at a given timestep, conditioned on the LLM's hidden state.
//!
//! Mirrors `vibevoice/modular/modular_vibevoice_diffusion_head.py`
//! upstream. ~123 M parameters in the 1.5B variant.
//!
//! ## Forward shape contract
//!
//! ```text
//!   noisy_latents : (B, latent_size)
//!   timesteps     : (B,)
//!   condition     : (B, hidden_size)        # from LLM hidden states
//!         |
//!         v
//!   predicted_noise : (B, latent_size)
//! ```
//!
//! Classifier-free guidance is applied **outside** this module — the
//! head sees a single (potentially zeroed) condition tensor and returns
//! a single prediction per pass. The orchestration code combines two
//! passes per step (`eps_uncond` and `eps_cond`) per the standard CFG
//! formula `eps = eps_uncond + scale * (eps_cond - eps_uncond)`.

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{linear_no_bias as linear, Linear, Module, VarBuilder};

use super::config::VibeVoiceDiffusionHeadConfig;

/// RMSNorm with optional learned scale, matching the Python code:
///
/// ```python
/// x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
/// ```
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // We compute in f32 internally for numerical stability and cast
        // back to the input dtype at the end; matches PyTorch behaviour
        // for fp16/bf16 inputs.
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_rms = (variance + self.eps)?.sqrt()?.recip()?;
        let normalized = x.broadcast_mul(&inv_rms)?;
        let scaled = normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        scaled.to_dtype(in_dtype)
    }
}

/// Sinusoidal timestep embedding — the canonical
/// `sin(t / 10000^(2k/d)), cos(t / 10000^(2k/d))` construction.
///
/// Mirrors the Python `TimestepEmbedder.timestep_embedding` static
/// method. Returns shape `(B, embedding_size)`.
fn sinusoidal_timestep_embedding(
    t: &Tensor,
    embedding_size: usize,
    max_period: f64,
    device: &Device,
) -> Result<Tensor> {
    let half = embedding_size / 2;
    let log_max = max_period.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_max * (i as f64) / (half as f64)).exp() as f32)
        .collect();
    let freqs = Tensor::new(freqs.as_slice(), device)?;
    // (B,) × (half,) → (B, half)
    let t = t.to_dtype(DType::F32)?.unsqueeze(D::Minus1)?;
    let args = t.broadcast_mul(&freqs.unsqueeze(0)?)?;
    let cos = args.cos()?;
    let sin = args.sin()?;
    let mut embedding = Tensor::cat(&[&cos, &sin], D::Minus1)?;
    if embedding_size % 2 == 1 {
        // Odd dim → zero-pad one extra column on the right.
        let zero = Tensor::zeros((embedding.dim(0)?, 1), DType::F32, device)?;
        embedding = Tensor::cat(&[&embedding, &zero], D::Minus1)?;
    }
    Ok(embedding)
}

/// `TimestepEmbedder` — sinusoidal embedding followed by a small MLP.
///
/// Constructor in the Python source uses `frequency_embedding_size=256`
/// hardcoded; we honour that.
#[derive(Debug, Clone)]
struct TimestepEmbedder {
    proj_in: Linear,
    proj_out: Linear,
    frequency_embedding_size: usize,
}

const FREQUENCY_EMBEDDING_SIZE: usize = 256;
const TIMESTEP_MAX_PERIOD: f64 = 10_000.0;

impl TimestepEmbedder {
    fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        // The Python code uses `nn.Sequential(Linear, SiLU, Linear)`;
        // sub-modules numbered "0" and "2" inside the sequential. We
        // load them by their numeric prefix to match upstream weight keys.
        let proj_in = linear(FREQUENCY_EMBEDDING_SIZE, hidden_size, vb.pp("mlp.0"))?;
        let proj_out = linear(hidden_size, hidden_size, vb.pp("mlp.2"))?;
        Ok(Self {
            proj_in,
            proj_out,
            frequency_embedding_size: FREQUENCY_EMBEDDING_SIZE,
        })
    }

    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let device = t.device();
        let freq = sinusoidal_timestep_embedding(
            t,
            self.frequency_embedding_size,
            TIMESTEP_MAX_PERIOD,
            device,
        )?;
        let h = self.proj_in.forward(&freq)?;
        let h = h.silu()?;
        self.proj_out.forward(&h)
    }
}

/// Apply AdaLN modulation:
///
/// ```text
///   modulate(x, shift, scale) = x * (1 + scale) + shift
/// ```
///
/// `shift` and `scale` are broadcasted across the inner dim of `x`.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let scale_p1 = scale.add(&Tensor::ones_like(scale)?)?;
    x.broadcast_mul(&scale_p1)?.broadcast_add(shift)
}

/// `HeadLayer` — one residual block with AdaLN modulation:
///
/// 1. `norm = RMSNorm(x)`
/// 2. `(shift, scale, gate) = chunk(SiLU then Linear(c), 3, dim=-1)`
/// 3. `mod_x = modulate(norm, shift, scale)`
/// 4. `ffn_out = SwiGLU FFN(mod_x)`  (Linear, gate * up, Linear)
/// 5. `x = x + gate * ffn_out`
///
/// In the Python source the FFN is a SwiGLU: two parallel linears
/// `gate_proj` and `up_proj`, multiplied with SiLU on the gate, then a
/// final `down_proj`.
#[derive(Debug, Clone)]
struct HeadLayer {
    norm: RmsNorm,
    cond_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl HeadLayer {
    fn new(hidden_size: usize, ffn_dim: usize, rms_eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm = RmsNorm::new(hidden_size, rms_eps, vb.pp("norm"))?;
        // adaLN: project condition into 3*hidden_size for shift/scale/gate.
        let cond_proj = linear(hidden_size, 3 * hidden_size, vb.pp("adaLN_modulation.1"))?;
        let gate_proj = linear(hidden_size, ffn_dim, vb.pp("ffn.gate_proj"))?;
        let up_proj = linear(hidden_size, ffn_dim, vb.pp("ffn.up_proj"))?;
        let down_proj = linear(ffn_dim, hidden_size, vb.pp("ffn.down_proj"))?;
        Ok(Self {
            norm,
            cond_proj,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        // adaLN: SiLU(c) → Linear → split into (shift, scale, gate)
        let c = c.silu()?;
        let proj = self.cond_proj.forward(&c)?;
        let chunks = proj.chunk(3, D::Minus1)?;
        let (shift, scale, gate) = (&chunks[0], &chunks[1], &chunks[2]);

        let normalized = self.norm.forward(x)?;
        let modulated = modulate(&normalized, shift, scale)?;

        // SwiGLU FFN
        let gate_act = self.gate_proj.forward(&modulated)?.silu()?;
        let up = self.up_proj.forward(&modulated)?;
        let ffn_out = self.down_proj.forward(&(gate_act * up)?)?;

        // Gated residual
        x.add(&ffn_out.broadcast_mul(gate)?)
    }
}

/// Final layer that produces the noise/velocity prediction. Same AdaLN
/// modulation pattern but only `(shift, scale)` — no FFN, just a direct
/// linear projection from `hidden_size` to `latent_size`.
#[derive(Debug, Clone)]
struct FinalLayer {
    norm: RmsNorm,
    cond_proj: Linear,
    out_proj: Linear,
}

impl FinalLayer {
    fn new(hidden_size: usize, latent_size: usize, rms_eps: f64, vb: VarBuilder) -> Result<Self> {
        let norm = RmsNorm::new(hidden_size, rms_eps, vb.pp("norm"))?;
        let cond_proj = linear(hidden_size, 2 * hidden_size, vb.pp("adaLN_modulation.1"))?;
        let out_proj = linear(hidden_size, latent_size, vb.pp("linear"))?;
        Ok(Self {
            norm,
            cond_proj,
            out_proj,
        })
    }

    fn forward(&self, x: &Tensor, c: &Tensor) -> Result<Tensor> {
        let c = c.silu()?;
        let proj = self.cond_proj.forward(&c)?;
        let chunks = proj.chunk(2, D::Minus1)?;
        let (shift, scale) = (&chunks[0], &chunks[1]);

        let normalized = self.norm.forward(x)?;
        let modulated = modulate(&normalized, shift, scale)?;
        self.out_proj.forward(&modulated)
    }
}

/// VibeVoice diffusion head. Constructed once and reused across every
/// diffusion step / CFG pass.
#[derive(Debug, Clone)]
pub struct VibeVoiceDiffusionHead {
    noisy_proj: Linear,
    cond_proj: Linear,
    timestep_embedder: TimestepEmbedder,
    layers: Vec<HeadLayer>,
    final_layer: FinalLayer,
}

impl VibeVoiceDiffusionHead {
    pub fn new(config: &VibeVoiceDiffusionHeadConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let latent_size = config.latent_size;
        let ffn_dim = (hidden_size as f64 * config.head_ffn_ratio) as usize;
        let rms_eps = config.rms_norm_eps;

        let noisy_proj = linear(latent_size, hidden_size, vb.pp("noisy_images_proj"))?;
        let cond_proj = linear(hidden_size, hidden_size, vb.pp("cond_proj"))?;
        let timestep_embedder = TimestepEmbedder::new(hidden_size, vb.pp("t_embedder"))?;
        let layers: Vec<HeadLayer> = (0..config.head_layers)
            .map(|i| HeadLayer::new(hidden_size, ffn_dim, rms_eps, vb.pp(format!("layers.{i}"))))
            .collect::<Result<_>>()?;
        let final_layer = FinalLayer::new(hidden_size, latent_size, rms_eps, vb.pp("final_layer"))?;

        Ok(Self {
            noisy_proj,
            cond_proj,
            timestep_embedder,
            layers,
            final_layer,
        })
    }

    /// Forward pass — see module-level shape contract. The caller
    /// is responsible for CFG (running this twice with different
    /// conditions and combining the predictions).
    pub fn forward(
        &self,
        noisy_latents: &Tensor,
        timesteps: &Tensor,
        condition: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.noisy_proj.forward(noisy_latents)?;
        let t = self.timestep_embedder.forward(timesteps)?;
        let c = self.cond_proj.forward(condition)?;
        let c = (c + t)?;
        for layer in &self.layers {
            x = layer.forward(&x, &c)?;
        }
        self.final_layer.forward(&x, &c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};
    use candle_nn::VarMap;

    /// Smoke test: a small instance compiles and runs end-to-end.
    /// Numerical equivalence vs. the upstream Python implementation is
    /// validated by a separate test that loads released weights — see
    /// `examples/vibevoice` (added in a follow-up commit).
    #[test]
    fn diffusion_head_small_forward_smoke() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        // Build a tiny test config by parsing an empty JSON (so all
        // serde defaults apply) then overriding the dimensions we want.
        let mut config: VibeVoiceDiffusionHeadConfig = serde_json::from_str("{}").unwrap();
        config.hidden_size = 64;
        config.latent_size = 32;
        config.head_ffn_ratio = 4.0;
        config.head_layers = 2;
        config.rms_norm_eps = 1e-6;

        let head = VibeVoiceDiffusionHead::new(&config, vb.pp("head"))?;
        let batch = 3;
        let noisy = Tensor::randn(0.0_f32, 1.0, (batch, config.latent_size), &device)?;
        let timesteps = Tensor::new(&[0u32, 50, 99], &device)?;
        let condition = Tensor::randn(0.0_f32, 1.0, (batch, config.hidden_size), &device)?;

        let out = head.forward(&noisy, &timesteps, &condition)?;
        assert_eq!(out.dims(), &[batch, config.latent_size]);
        Ok(())
    }

    #[test]
    fn modulate_matches_python_formula() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0]], &device)?;
        let shift = Tensor::new(&[[0.5_f32, 0.0, -1.0]], &device)?;
        let scale = Tensor::new(&[[1.0_f32, 0.5, 0.0]], &device)?;
        let out = modulate(&x, &shift, &scale)?;
        // x * (1 + scale) + shift  =  [1*2+0.5, 2*1.5+0, 3*1-1] = [2.5, 3.0, 2.0]
        let v = out.to_vec2::<f32>()?;
        assert!((v[0][0] - 2.5).abs() < 1e-6);
        assert!((v[0][1] - 3.0).abs() < 1e-6);
        assert!((v[0][2] - 2.0).abs() < 1e-6);
        Ok(())
    }
}

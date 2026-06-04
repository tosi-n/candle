//! DPM-Solver scheduler for VibeVoice's diffusion head.
//!
//! Pure-candle port of `vibevoice/schedule/dpm_solver.py` — self-contained,
//! deliberately not pulling from [`crate::models::stable_diffusion`].
//!
//! Only the configuration VibeVoice actually uses is implemented:
//! `algorithm_type=dpmsolver++`, `solver_type=midpoint`, `solver_order=2`,
//! `prediction_type=v_prediction`, `beta_schedule=cosine`,
//! `final_sigmas_type=zero`, `timestep_spacing=linspace`,
//! `lower_order_final=true`, `thresholding=false`,
//! `use_karras_sigmas=false`, `use_lu_lambdas=false`,
//! `rescale_betas_zero_snr=false`. Other branches are not implemented.
//!
//! All schedule arrays live host-side as `Vec<f32>` / `Vec<i64>`.
//! Per-step coefficients are scalars; the only tensor work in `step()`
//! is `affine` / `add` / `sub` against the model output and current
//! sample.
//!
//! ## Numerical caveats
//!
//! 1. **Terminal step (`sigmas[N] = 0`).** The first-order branch is
//!    forced on the last step. The denominator `sigma_t / sigma_s` is
//!    `0/sigma_s = 0`, and `(exp(-h) - 1)` resolves cleanly to `-1`
//!    because `f64::exp(-f64::INFINITY) == 0.0`. We compute scalar
//!    coefficients in `f64` to keep the limit exact under IEEE 754.
//! 2. **`np.interp` clamping.** When the requested timestep coincides
//!    with `xp[-1] = T-1` we short-circuit to `fp[-1]` before falling
//!    into the binary-search interior path.
//! 3. **`linspace(0, T-1, N+1).round()[::-1][:-1]` cast to `i64`.**
//!    Implemented literally — integer division would diverge from the
//!    upstream timestep array.

use candle::{DType, Result, Tensor};

/// Beta schedule used by the underlying noise process. VibeVoice ships
/// `cosine`; the others are kept as enum variants for parity but are
/// not implemented in this inference-only port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaSchedule {
    Cosine,
    Linear,
    ScaledLinear,
}

/// What the diffusion head predicts at each step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}

/// Minimal config — VibeVoice exposes only these three knobs;
/// everything else is hardcoded to the upstream defaults.
#[derive(Debug, Clone, Copy)]
pub struct DpmSolverConfig {
    pub num_train_timesteps: usize,
    pub beta_schedule: BetaSchedule,
    pub prediction_type: PredictionType,
}

impl Default for DpmSolverConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_schedule: BetaSchedule::Cosine,
            prediction_type: PredictionType::VPrediction,
        }
    }
}

const SOLVER_ORDER: usize = 2;
const MAX_BETA_COSINE: f64 = 0.999;

/// Build the cosine `betas` array via `betas_for_alpha_bar(T, "cosine")`.
///
/// Mirrors lines 28–83 of `dpm_solver.py` for the cosine α-bar
/// (`cos((t+0.008)/1.008 * pi/2)^2`) used by VibeVoice.
fn betas_for_alpha_bar_cosine(num_train_timesteps: usize) -> Vec<f64> {
    let t = num_train_timesteps as f64;
    let alpha_bar = |x: f64| {
        let phase = (x + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2;
        let c = phase.cos();
        c * c
    };
    let mut betas = Vec::with_capacity(num_train_timesteps);
    for i in 0..num_train_timesteps {
        let t1 = i as f64 / t;
        let t2 = (i + 1) as f64 / t;
        let beta = (1.0 - alpha_bar(t2) / alpha_bar(t1)).min(MAX_BETA_COSINE);
        betas.push(beta);
    }
    betas
}

fn build_linear_betas(num_train_timesteps: usize, beta_start: f64, beta_end: f64) -> Vec<f64> {
    let n = num_train_timesteps;
    if n == 1 {
        return vec![beta_start];
    }
    let denom = (n - 1) as f64;
    (0..n)
        .map(|i| beta_start + (beta_end - beta_start) * (i as f64) / denom)
        .collect()
}

/// Linear `np.interp` with clamp-to-edge semantics.
///
/// `x` and `xp` are ascending-sorted and `xp.len() == fp.len()`. For
/// `query <= xp[0]` returns `fp[0]`; for `query >= xp[xp.len()-1]`
/// returns the last `fp` value (short-circuit before the search to
/// avoid off-by-one drift at the right edge).
fn np_interp(query: f64, xp: &[f64], fp: &[f64]) -> f64 {
    debug_assert_eq!(xp.len(), fp.len());
    let n = xp.len();
    if query <= xp[0] {
        return fp[0];
    }
    if query >= xp[n - 1] {
        return fp[n - 1];
    }
    // Binary search for the interval `[xp[k], xp[k+1])`.
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xp[mid] <= query {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let span = xp[hi] - xp[lo];
    if span == 0.0 {
        return fp[lo];
    }
    let t = (query - xp[lo]) / span;
    fp[lo] + t * (fp[hi] - fp[lo])
}

/// DPM-Solver state — built once per generation, mutated as `step()`
/// progresses through `timesteps`.
#[derive(Debug, Clone)]
pub struct DpmSolver {
    config: DpmSolverConfig,
    /// Per-train-timestep `sigma = sqrt((1 - α̅_t) / α̅_t)`.
    sigmas_train: Vec<f64>,
    /// Inference timesteps in **descending** order, length `N`.
    pub timesteps: Vec<i64>,
    /// Inference sigmas, length `N + 1`. Last entry is **0** (final
    /// sigma type "zero"), forcing the lower-order branch on the last
    /// step.
    sigmas: Vec<f64>,
    /// FIFO of converted model outputs (x0 predictions). Length
    /// `solver_order = 2`.
    model_outputs: [Option<Tensor>; SOLVER_ORDER],
    /// How many converted outputs we've collected so far. Gates the
    /// upgrade from order-1 to order-2.
    lower_order_nums: usize,
    /// Current index into `timesteps` / `sigmas`. `None` until the
    /// first call to `step()`.
    step_index: Option<usize>,
}

impl DpmSolver {
    /// Build the schedule. `num_inference_steps` is typically 20 for
    /// VibeVoice.
    pub fn new(config: DpmSolverConfig, num_inference_steps: usize) -> Self {
        let betas = match config.beta_schedule {
            BetaSchedule::Cosine => betas_for_alpha_bar_cosine(config.num_train_timesteps),
            BetaSchedule::Linear => build_linear_betas(config.num_train_timesteps, 1e-4, 0.02),
            BetaSchedule::ScaledLinear => {
                let raw = build_linear_betas(
                    config.num_train_timesteps,
                    1e-4_f64.sqrt(),
                    0.02_f64.sqrt(),
                );
                raw.iter().map(|b| b * b).collect()
            }
        };
        // alphas, alpha-cumprod, sigmas — all on the host.
        let mut alphas_cumprod = Vec::with_capacity(betas.len());
        let mut acc = 1.0f64;
        for b in &betas {
            acc *= 1.0 - b;
            alphas_cumprod.push(acc);
        }
        let sigmas_train: Vec<f64> = alphas_cumprod
            .iter()
            .map(|a| ((1.0 - a) / a).sqrt())
            .collect();

        let mut solver = Self {
            config,
            sigmas_train,
            timesteps: Vec::new(),
            sigmas: Vec::new(),
            model_outputs: [None, None],
            lower_order_nums: 0,
            step_index: None,
        };
        solver.set_timesteps(num_inference_steps);
        solver
    }

    /// Build the inference timestep schedule (descending) and the
    /// matching `sigmas` array (length `N + 1`, last entry 0).
    pub fn set_timesteps(&mut self, num_inference_steps: usize) {
        let t = self.config.num_train_timesteps as f64;
        let n = num_inference_steps;
        // `linspace(0, T-1, N+1)`.
        let mut ts_float = Vec::with_capacity(n + 1);
        if n == 0 {
            ts_float.push(0.0);
        } else {
            for i in 0..=n {
                ts_float.push((t - 1.0) * (i as f64) / (n as f64));
            }
        }
        // Round per-element, reverse, drop the last entry, cast to i64.
        let mut ts_round: Vec<i64> = ts_float.iter().map(|v| v.round() as i64).collect();
        ts_round.reverse();
        if !ts_round.is_empty() {
            ts_round.pop();
        }

        // np.interp into `sigmas_train` at the resampled timesteps. Since
        // `xp = arange(T)`, gathering at integer-rounded timesteps
        // reduces to direct indexing — but we route through the generic
        // interp so non-integer inputs (if ever passed) still work.
        let xp: Vec<f64> = (0..self.config.num_train_timesteps)
            .map(|i| i as f64)
            .collect();
        let mut sigmas: Vec<f64> = ts_round
            .iter()
            .map(|x| np_interp(*x as f64, &xp, &self.sigmas_train))
            .collect();
        sigmas.push(0.0); // final_sigmas_type = "zero"

        self.timesteps = ts_round;
        self.sigmas = sigmas;
        self.model_outputs = [None, None];
        self.lower_order_nums = 0;
        self.step_index = None;
    }

    fn init_step_index(&mut self, timestep: i64) {
        // Match all positions where `timesteps == timestep`. If multiple,
        // pick the second to mirror the upstream `index_for_timestep`
        // tie-breaking. If none, fall back to the last index.
        let matches: Vec<usize> = self
            .timesteps
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == timestep).then_some(i))
            .collect();
        let idx = match matches.len() {
            0 => self.timesteps.len().saturating_sub(1),
            1 => matches[0],
            _ => matches[1],
        };
        self.step_index = Some(idx);
    }

    /// Convert `(sigma)` → `(alpha_t, sigma_t)` per
    /// `_sigma_to_alpha_sigma_t` (lines 483–487).
    fn sigma_to_alpha_sigma(sigma: f64) -> (f64, f64) {
        let scale = (sigma * sigma + 1.0).sqrt();
        (1.0 / scale, sigma / scale)
    }

    /// Convert raw model output into an x0 prediction.
    /// Implements the `dpmsolver++` × `v_prediction` branch only.
    fn convert_model_output(&self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let i = self
            .step_index
            .ok_or_else(|| candle::Error::Msg("DpmSolver step_index not initialised".into()))?;
        let sigma = self.sigmas[i];
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(sigma);
        match self.config.prediction_type {
            PredictionType::VPrediction => {
                // x0 = alpha_t * sample - sigma_t * model_output
                let term_a = (sample * alpha_t)?;
                let term_b = (model_output * sigma_t)?;
                term_a.sub(&term_b)
            }
            PredictionType::Epsilon | PredictionType::Sample => {
                candle::bail!(
                    "DpmSolver currently only supports v_prediction; epsilon/sample paths \
                     are out of scope for the VibeVoice port"
                )
            }
        }
    }

    /// First-order DPM-Solver++ update.
    fn first_order_update(&self, model_x0: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let i = self.step_index.expect("step_index must be set");
        let sigma_t_raw = self.sigmas[i + 1];
        let sigma_s_raw = self.sigmas[i];
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(sigma_t_raw);
        let (alpha_s, sigma_s) = Self::sigma_to_alpha_sigma(sigma_s_raw);
        // Terminal step (sigma_t_raw = 0) → sigma_t = 0 / 1 = 0 and
        // h → +inf. exp(-h) → 0, so (exp(-h) - 1) = -1 and the result
        // collapses to alpha_t * model_x0 == 1 * x0_pred.
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s = alpha_s.ln() - sigma_s.ln();
        let h = lambda_t - lambda_s;
        let coef_sample = sigma_t / sigma_s;
        let coef_x0 = alpha_t * ((-h).exp() - 1.0);
        let term_a = (sample * coef_sample)?;
        let term_b = (model_x0 * coef_x0)?;
        term_a.sub(&term_b)
    }

    /// Multistep DPM-Solver++ second-order update (midpoint variant).
    fn second_order_update(&self, sample: &Tensor) -> Result<Tensor> {
        let i = self.step_index.expect("step_index must be set");
        let sigma_t_raw = self.sigmas[i + 1];
        let sigma_s0_raw = self.sigmas[i];
        let sigma_s1_raw = self.sigmas[i.saturating_sub(1)];
        let (alpha_t, sigma_t) = Self::sigma_to_alpha_sigma(sigma_t_raw);
        let (_alpha_s0, sigma_s0) = Self::sigma_to_alpha_sigma(sigma_s0_raw);
        let (_alpha_s1, sigma_s1) = Self::sigma_to_alpha_sigma(sigma_s1_raw);
        let lambda_t = alpha_t.ln() - sigma_t.ln();
        let lambda_s0 = (1.0 / (sigma_s0_raw * sigma_s0_raw + 1.0).sqrt()).ln() - sigma_s0.ln();
        let lambda_s1 = (1.0 / (sigma_s1_raw * sigma_s1_raw + 1.0).sqrt()).ln() - sigma_s1.ln();

        let h = lambda_t - lambda_s0;
        let h_0 = lambda_s0 - lambda_s1;
        let r0 = h_0 / h;

        let m0 = self.model_outputs[1]
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing m0 for 2nd-order update".into()))?;
        let m1 = self.model_outputs[0]
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing m1 for 2nd-order update".into()))?;

        // D0 = m0; D1 = (1/r0) * (m0 - m1)
        let d0 = m0.clone();
        let diff = m0.sub(m1)?;
        let d1 = (diff * (1.0 / r0))?;

        let coef_sample = sigma_t / sigma_s0;
        let coef_d0 = alpha_t * ((-h).exp() - 1.0);
        let coef_d1 = 0.5 * coef_d0;

        let part_a = (sample * coef_sample)?;
        let part_b = (d0 * coef_d0)?;
        let part_c = (d1 * coef_d1)?;
        part_a.sub(&part_b)?.sub(&part_c)
    }

    /// One denoising step: given the diffusion head's prediction at the
    /// current `timestep`, return the next (less-noisy) sample.
    pub fn step(
        &mut self,
        model_output: &Tensor,
        timestep: i64,
        sample: &Tensor,
    ) -> Result<Tensor> {
        if self.step_index.is_none() {
            self.init_step_index(timestep);
        }
        let i = self
            .step_index
            .ok_or_else(|| candle::Error::Msg("step_index missing".into()))?;
        let n = self.timesteps.len();
        // For VibeVoice (`final_sigmas_type="zero"`), the last step is
        // forced into the first-order branch.
        let lower_order_final = i + 1 == n;

        // Convert and shift the FIFO.
        let in_dtype = sample.dtype();
        let sample_f32 = sample.to_dtype(DType::F32)?;
        let m = self.convert_model_output(&model_output.to_dtype(DType::F32)?, &sample_f32)?;
        self.model_outputs[0] = self.model_outputs[1].clone();
        self.model_outputs[1] = Some(m.clone());

        let prev = if SOLVER_ORDER == 1 || self.lower_order_nums < 1 || lower_order_final {
            self.first_order_update(&m, &sample_f32)?
        } else {
            self.second_order_update(&sample_f32)?
        };

        if self.lower_order_nums < SOLVER_ORDER {
            self.lower_order_nums += 1;
        }
        self.step_index = Some(i + 1);
        prev.to_dtype(in_dtype)
    }

    pub fn config(&self) -> DpmSolverConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn cosine_betas_are_bounded() {
        let betas = betas_for_alpha_bar_cosine(1000);
        assert_eq!(betas.len(), 1000);
        for b in &betas {
            assert!(*b >= 0.0 && *b <= MAX_BETA_COSINE);
        }
    }

    #[test]
    fn schedule_descends_and_terminates_with_zero_sigma() {
        let solver = DpmSolver::new(DpmSolverConfig::default(), 20);
        assert_eq!(solver.timesteps.len(), 20);
        assert_eq!(solver.sigmas.len(), 21);
        for w in solver.timesteps.windows(2) {
            assert!(w[0] > w[1]);
        }
        assert_eq!(solver.sigmas[20], 0.0);
    }

    #[test]
    fn np_interp_matches_at_endpoints_and_midpoints() {
        let xp = vec![0.0, 1.0, 2.0, 3.0];
        let fp = vec![0.0, 10.0, 20.0, 30.0];
        assert_eq!(np_interp(-5.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(0.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(0.5, &xp, &fp), 5.0);
        assert_eq!(np_interp(1.5, &xp, &fp), 15.0);
        assert_eq!(np_interp(3.0, &xp, &fp), 30.0);
        assert_eq!(np_interp(99.0, &xp, &fp), 30.0);
    }

    #[test]
    fn first_order_update_handles_terminal_step() -> Result<()> {
        // Numerically: with sigma_t = 0, the result should equal
        // `alpha_t * x0` = `1.0 * x0`. We construct a 2-step solver and
        // step from the next-to-last to the last position to exercise
        // the terminal branch.
        let mut solver = DpmSolver::new(DpmSolverConfig::default(), 2);
        let device = Device::Cpu;
        let model_v = Tensor::new(&[1.0_f32, -0.5, 0.25], &device)?;
        let sample = Tensor::new(&[0.0_f32, 0.0, 0.0], &device)?;
        // First step (uses solver.timesteps[0], goes to sigmas[1])
        let _intermediate = solver.step(&model_v, solver.timesteps[0], &sample)?;
        // Second step lands at sigmas[2] = 0 (terminal)
        let final_sample = solver.step(&model_v, solver.timesteps[1], &_intermediate)?;
        // Just confirm the output is finite — actual numerical match
        // against Python upstream is validated by an end-to-end test.
        let v = final_sample.to_vec1::<f32>()?;
        for x in v {
            assert!(x.is_finite(), "terminal-step output produced NaN/Inf: {x}");
        }
        Ok(())
    }
}

//! Flow-matching primitives for the Token2Wav DiT vocoder.
//!
//! Mirrors `Qwen2_5OmniToken2WavDiTModel.sample()` and the integrator it
//! uses (upstream `modeling_qwen2_5_omni.py:3368-3421` for the solver +
//! 3526-3592 for the sampling driver).
//!
//! Three independent primitives — each unit-testable on a laptop
//! against closed-form references, **before** the DiT itself is wired:
//!
//! 1. [`RungeKutta38Solver::integrate_final`] — the integrator that the
//!    Token2Wav DiT actually uses. **Despite the upstream class being
//!    named `RungeKutta4ODESolver`, it is the Kutta 3/8 rule, not
//!    classical RK4.** Weights `(1, 3, 3, 1)/8` at `c = (0, 1/3, 2/3, 1)`,
//!    not the Simpson `(1, 2, 2, 1)/6` at `(0, 1/2, 1/2, 1)`. Both are
//!    4th-order but they're different methods.
//! 2. [`sway_schedule`] — `t' = t + s·(cos(π·t/2) − 1 + t)`. At
//!    `s = -1.0` (upstream default) collapses to the clean closed form
//!    `t' = 1 − cos(π·u/2)`, biasing solver steps toward `t = 0` (the
//!    noise end of the flow).
//! 3. [`cfg_combine`] — `out = cond + scale · (cond − null)`. **Base
//!    point is `cond`, NOT `null`.** With the upstream default
//!    `scale = 0.5` this is `1.5·cond − 0.5·null` — half the strength
//!    of the more common `null + (1+s)·(cond − null)` form. Easy to
//!    accidentally flip the base.

use candle::{Result, Tensor};

/// Kutta 3/8-rule 4th-order Runge-Kutta integrator.
///
/// One step of size `h = t1 - t0` from `y` at `t0`:
///
/// ```text
///   k1 = f(t0,        y)
///   k2 = f(t0 + h/3,  y + h·k1/3)
///   k3 = f(t0 + 2h/3, y + h·(k2 − k1/3))
///   k4 = f(t0 + h,    y + h·(k1 − k2 + k3))
///   Δy = h · (k1 + 3·k2 + 3·k3 + k4) / 8
/// ```
///
/// Mirror of upstream `RungeKutta4ODESolver._rk4_step`
/// (modeling_qwen2_5_omni.py:3376-3381).
pub struct RungeKutta38Solver;

/// Velocity field trait. `eval(t, y)` returns `dy/dt`. Closures and DiT
/// forward calls both implement this — the integrator is generic over
/// the field.
pub trait OdeField {
    fn eval(&self, t: f32, y: &Tensor) -> Result<Tensor>;
}

impl RungeKutta38Solver {
    /// Single Kutta 3/8 step. Returns Δy (the increment to add to `y`).
    /// Exposed for unit tests.
    pub fn rk38_step<F: OdeField>(
        field: &F,
        t0: f32,
        h: f32,
        y: &Tensor,
    ) -> Result<Tensor> {
        let one_third = 1.0f32 / 3.0;
        let two_thirds = 2.0f32 / 3.0;

        let k1 = field.eval(t0, y)?;
        // k2 = f(t0 + h/3, y + h·k1/3)
        let y2 = (y + (&k1 * (h * one_third) as f64)?)?;
        let k2 = field.eval(t0 + h * one_third, &y2)?;
        // k3 = f(t0 + 2h/3, y + h·(k2 − k1/3))
        let inner = (&k2 - (&k1 * one_third as f64)?)?;
        let y3 = (y + (&inner * h as f64)?)?;
        let k3 = field.eval(t0 + h * two_thirds, &y3)?;
        // k4 = f(t0 + h, y + h·(k1 − k2 + k3))
        let inner2 = ((&k1 - &k2)? + &k3)?;
        let y4 = (y + (&inner2 * h as f64)?)?;
        let k4 = field.eval(t0 + h, &y4)?;
        // Δy = h · (k1 + 3·k2 + 3·k3 + k4) / 8
        let weighted = ((&k1 + (&k2 * 3.0)?)? + (&k3 * 3.0)?)?;
        let weighted = (weighted + &k4)?;
        Ok((weighted * (h as f64 / 8.0))?)
    }

    /// Integrate `field` from `y0` along `time_points` and return ONLY
    /// the final state — `sample()` upstream only reads
    /// `solution[-1]`, so we never need the full trajectory.
    pub fn integrate_final<F: OdeField>(
        field: &F,
        y0: &Tensor,
        time_points: &[f32],
    ) -> Result<Tensor> {
        if time_points.len() < 2 {
            candle::bail!("integrate_final needs at least 2 time points; got {}", time_points.len());
        }
        let mut y = y0.clone();
        for w in time_points.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            let h = t1 - t0;
            let dy = Self::rk38_step(field, t0, h, &y)?;
            y = (y + dy)?;
        }
        Ok(y)
    }
}

/// Build the warped time schedule used by Qwen2.5-Omni Token2Wav:
///
/// `t' = u + s·(cos(π·u/2) − 1 + u)`  where `u ∈ linspace(0, 1, num_steps)`.
///
/// At `s = -1.0` this simplifies to `t' = 1 − cos(π·u/2)` (verified
/// in the docs §4 Phase 4b lock). With s = -1 the derivative
/// `t'(u) = 1 + s − s·π/2·sin(π·u/2)` gives `t'(0) = 0` (slow start) and
/// `t'(1) ≈ 1.571` (fast finish), i.e. **more solver steps near
/// `t = 0`** (the noise end of the flow).
pub fn sway_schedule(num_steps: usize, coefficient: f32) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..num_steps)
        .map(|i| {
            let u = if num_steps == 1 {
                0.0
            } else {
                i as f32 / (num_steps - 1) as f32
            };
            u + coefficient * ((PI * u / 2.0).cos() - 1.0 + u)
        })
        .collect()
}

/// Classifier-free guidance combine. Returns `cond + scale · (cond − null)`.
///
/// **Base point is `cond`, NOT `null`.** This matches upstream
/// modeling_qwen2_5_omni.py:3577 exactly:
///
/// ```python
/// return guided_prediction + (guided_prediction - null_prediction) * guidance_scale
/// ```
///
/// With upstream default `guidance_scale = 0.5` this gives
/// `1.5·cond − 0.5·null` — half the strength of the more common
/// `null + (1+s)·(cond − null)` formulation. Easy to silently flip.
pub fn cfg_combine(cond: &Tensor, null: &Tensor, scale: f32) -> Result<Tensor> {
    let diff = (cond - null)?;
    cond + (diff * scale as f64)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};

    /// Field for `dy/dt = y` (exponential growth). Closed form
    /// `y(t) = y(0) · e^t` — the canonical RK4 sanity check.
    struct ExpField;
    impl OdeField for ExpField {
        fn eval(&self, _t: f32, y: &Tensor) -> Result<Tensor> {
            Ok(y.clone())
        }
    }

    /// Field for `dy/dt = cos(t)`. Closed form `y(t) = sin(t) + C`.
    /// Time-dependent — catches bugs where `t0 + h/3` or `t0 + 2h/3`
    /// substep times are wrong (which `dy/dt = y` would miss).
    struct CosField;
    impl OdeField for CosField {
        fn eval(&self, t: f32, y: &Tensor) -> Result<Tensor> {
            let val = Tensor::full(t.cos(), y.shape(), y.device())?;
            Ok(val)
        }
    }

    /// **Phase 4 acceptance #1** — RK 3/8 integrates `dy/dt = y` to
    /// `e^1 ≈ 2.71828` within 1e-5 over 10 uniform steps. The 3/8 rule
    /// is 4th-order, so global error scales as O(h^4) — with h=0.1 we
    /// expect `|err| ≲ 1e-5`.
    #[test]
    fn rk38_integrates_exp_growth() -> Result<()> {
        let dev = Device::Cpu;
        let y0 = Tensor::ones((1,), DType::F32, &dev)?;
        let ts: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();
        let y_final = RungeKutta38Solver::integrate_final(&ExpField, &y0, &ts)?;
        let got: f32 = y_final.to_vec1::<f32>()?[0];
        let expected = std::f32::consts::E;
        let err = (got - expected).abs();
        assert!(err < 1e-5, "RK 3/8 |err| = {err}, got = {got}, expected = {expected}");
        Ok(())
    }

    /// **Phase 4 acceptance #2** — RK 3/8 integrates `dy/dt = cos(t)`
    /// from y(0)=0 to y(1)=sin(1) ≈ 0.8414710 within 1e-4. Forces the
    /// substep times t0+h/3, t0+2h/3 to be evaluated correctly.
    #[test]
    fn rk38_integrates_cos_field() -> Result<()> {
        let dev = Device::Cpu;
        let y0 = Tensor::zeros((1,), DType::F32, &dev)?;
        let ts: Vec<f32> = (0..=10).map(|i| i as f32 * 0.1).collect();
        let y_final = RungeKutta38Solver::integrate_final(&CosField, &y0, &ts)?;
        let got: f32 = y_final.to_vec1::<f32>()?[0];
        let expected = 1.0f32.sin();
        let err = (got - expected).abs();
        assert!(err < 1e-4, "RK 3/8 cos |err| = {err}, got = {got}, expected = {expected}");
        Ok(())
    }

    /// **Phase 4 acceptance #3** — sway schedule endpoints are exactly
    /// `(0, 1)`, monotonic in between, and at `s = -1.0` collapses to
    /// the closed form `1 − cos(π·u/2)` element-wise.
    #[test]
    fn sway_schedule_endpoints_and_closed_form() {
        let ts = sway_schedule(10, -1.0);
        assert_eq!(ts.len(), 10);
        assert!(ts[0].abs() < 1e-6, "t(0) = {}", ts[0]);
        assert!((ts[9] - 1.0).abs() < 1e-6, "t(1) = {}", ts[9]);
        // Monotonic increasing.
        for w in ts.windows(2) {
            assert!(w[1] > w[0], "non-monotonic: {} -> {}", w[0], w[1]);
        }
        // Closed form at s = -1: t' = 1 - cos(π·u/2).
        for (i, &t) in ts.iter().enumerate() {
            let u = i as f32 / 9.0;
            let expected = 1.0 - (std::f32::consts::PI * u / 2.0).cos();
            assert!(
                (t - expected).abs() < 1e-6,
                "sway[{i}] = {t}, expected closed form {expected}"
            );
        }
        // With s = -1, steps are SHORTER near t=0 than near t=1
        // (more solver resolution near the noise end of the flow).
        let h_start = ts[1] - ts[0];
        let h_end = ts[9] - ts[8];
        assert!(h_start < h_end, "expected h_start({h_start}) < h_end({h_end})");

        // s = 0 should recover plain linspace.
        let ts0 = sway_schedule(10, 0.0);
        for (i, &t) in ts0.iter().enumerate() {
            assert!((t - i as f32 / 9.0).abs() < 1e-6, "linspace at s=0: ts0[{i}] = {t}");
        }
    }

    /// **Phase 4 acceptance #4** — CFG combine arithmetic. Locks in the
    /// `cond + s·(cond − null)` base — easy to accidentally write the
    /// `null + s·(cond − null)` form, which differs by an offset of
    /// `s·(cond − null)`.
    #[test]
    fn cfg_combine_arithmetic() -> Result<()> {
        let dev = Device::Cpu;
        let cond = Tensor::from_vec(vec![2.0f32, 4.0, 6.0], (3,), &dev)?;
        let null = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &dev)?;
        // s = 0.5 (upstream default): cond + 0.5·(cond − null) = [2.5, 5, 7.5]
        let out = cfg_combine(&cond, &null, 0.5)?;
        let got: Vec<f32> = out.to_vec1()?;
        let expected = [2.5f32, 5.0, 7.5];
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-6, "CFG combine mismatch: got {g}, expected {e}");
        }
        // s = 0: identity to cond.
        let zero = cfg_combine(&cond, &null, 0.0)?;
        assert_eq!(zero.to_vec1::<f32>()?, vec![2.0, 4.0, 6.0]);
        Ok(())
    }
}

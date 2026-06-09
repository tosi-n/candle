//! Qwen2.5-Omni Token2Wav BigVGAN vocoder (mel → 24 kHz waveform).
//!
//! Mirrors `Qwen2_5OmniToken2WavBigVGANModel` and its building blocks
//! (`SnakeBeta`, `UpSample1d`, `DownSample1d`, `TorchActivation1d`,
//! `AMPBlock`, `kaiser_sinc_filter1d`) in the upstream
//! `modeling_qwen2_5_omni.py` (lines 85-128 + 3079-3365).
//!
//! ## Pipeline
//!
//! ```text
//!   mel (B, 80, T)
//!     │  process_mel_spectrogram  (exp → amplitude_to_db → −20 → normalize)
//!     ▼
//!   conv_pre  Conv1d(80 → 1536, k=7, pad=3, bias=true)
//!     │
//!   6× upsample level i ∈ 0..6:
//!     │   ups[i][0]  ConvTranspose1d  (1536→768→384→192→96→48→24)
//!     │   hidden = (resblocks[3i] + resblocks[3i+1] + resblocks[3i+2]) / 3
//!     ▼                                  (AVG-summed — NOT residual-added)
//!   activation_post  TorchActivation1d(SnakeBeta(24))
//!     │
//!   conv_post  Conv1d(24 → 1, k=7, pad=3, bias=FALSE)
//!     │  clamp(-1, 1)
//!     ▼
//!   waveform (B, 1, 240·T)                240 = 5·3·2·2·2·2
//! ```
//!
//! ## Critical gotchas (audited against upstream + the Phase-5 spec
//! `hybrie/docs/qwen2_5_omni-phases-2-6-spec.md`)
//!
//! - **SnakeBeta has TWO separate `(C,)` log-space parameters** `alpha`
//!   and `beta`. `y = x + (1/(exp(beta)+1e-9)) · sin²(x·exp(alpha))`.
//!   Not one tensor, not magnitude-shared. Keys live under `.act.alpha`
//!   / `.act.beta`.
//! - **Anti-aliasing filters are NOT in safetensors** (non-persistent
//!   buffers upstream). Recompute via [`kaiser_sinc_filter1d`] in
//!   `new()` and store as plain `Tensor`s — they must never pass through
//!   the `VarBuilder` or the 449-key audit gains phantom entries. Up and
//!   down filters are identical here (cutoff 0.25, half-width 0.3,
//!   k=12), so we compute the filter once and reuse it.
//! - **`UpSample1d` multiplies by `ratio`**: output is
//!   `ratio · conv_transpose1d(...)`, then sliced `[15:-15]`. Padding is
//!   **replicate** (repeat edge column), NOT reflect (reflect is the
//!   ECAPA trap). The anti-alias up/down convs are **grouped/depthwise**
//!   (`groups=C`), **bias-less**, `padding=0` (the manual replicate pad
//!   does the padding).
//! - **Two distinct ConvTranspose1d uses, do not cross configs**: the 6
//!   main `ups` are `groups=1`, `bias=true`, `padding=(k−s)/2`; the
//!   anti-alias upsample is `groups=C`, no bias, `padding=0`.
//! - **`conv_post` has NO bias** (weight only); `conv_pre` has bias.
//! - **Per-level resblock combine is an AVERAGE-SUM, not a residual
//!   add**: `hidden = Σ_j resblock_j(hidden) / 3`. The upsampled tensor
//!   survives only through each AMPBlock's *internal* residual.
//! - **`use_bias_at_final` config field is DEAD** — `conv_post`
//!   `bias=False` is hard-coded upstream.
//! - The output keeps `(B, 1, L)` dims (we do NOT mimic upstream's
//!   `.squeeze().cpu()` — the dictated interface returns `(B, 1, 240·T)`).
//!
//! ## Phase-5 scope
//!
//! Forward inference only (`(B, 80, T)` → `(B, 1, 240·T)`). Loads the
//! `token2wav.code2wav_bigvgan_model.*` slice (449 keys) of the released
//! checkpoint. Always F32 on CPU (candle has no bf16 CPU matmul/conv
//! kernel); a `#[cfg(feature="cuda")]` smoke runs native bf16.

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    VarBuilder,
};

use super::super::config::BigVganConfig;

/// Output sample rate of the BigVGAN vocoder. Inferred from the total
/// upsample product `5·3·2·2·2·2 = 240` over the 100 Hz mel frame rate —
/// this is NOT stored as a config field upstream.
pub const SAMPLE_RATE: u32 = 24_000;

/// `min_db` used by `amplitude_to_db` / `normalize_spectrogram`. Literal
/// upstream (lines 3347-3348), not a config field.
const MIN_DB: f64 = -115.0;

/// Kernel size of every anti-aliasing Kaiser-sinc filter (up & down,
/// `up_kernel_size = down_kernel_size = 12` upstream).
const AA_KERNEL_SIZE: usize = 12;

/// `no_div_by_zero` epsilon inside SnakeBeta (upstream `1e-9`).
const SNAKE_EPS: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Kaiser-sinc filter (ports modeling_qwen2_5_omni.py lines 85-128)
// ---------------------------------------------------------------------------

/// Modified Bessel function of the first kind, order 0 — `I0(x)`.
///
/// `I0(x) = Σ_{k≥0} ((x/2)^k / k!)²`. ~30 terms is f32-exact for the
/// `beta ≈ 4.66` regime used here. Needed by [`kaiser_window`] because
/// candle has no `kaiser_window` builtin.
fn bessel_i0(x: f64) -> f64 {
    let half_x = x / 2.0;
    let mut term = 1.0f64; // k = 0 term: (x/2)^0 / 0! = 1
    let mut sum = 1.0f64;
    for k in 1..40 {
        // term_k = ((x/2)^k / k!)² ; ratio term_k/term_{k-1} = (x/2 / k)²
        let r = half_x / k as f64;
        term *= r * r;
        sum += term;
        if term < 1e-18 * sum {
            break;
        }
    }
    sum
}

/// `torch.kaiser_window(window_length, beta, periodic=false)`:
/// `w[n] = I0(beta·√(1 − (2n/(N−1) − 1)²)) / I0(beta)`, n = 0..N−1.
fn kaiser_window(window_length: usize, beta: f64) -> Vec<f64> {
    if window_length == 1 {
        return vec![1.0];
    }
    let denom = bessel_i0(beta);
    let n_minus_1 = (window_length - 1) as f64;
    (0..window_length)
        .map(|n| {
            let ratio = 2.0 * n as f64 / n_minus_1 - 1.0;
            let arg = beta * (1.0 - ratio * ratio).max(0.0).sqrt();
            bessel_i0(arg) / denom
        })
        .collect()
}

/// Normalized cardinal sine: `sinc(x) = sin(πx)/(πx)`, `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// 1-D Kaiser-windowed sinc filter. Returns the `kernel_size` taps as an
/// `(1, 1, kernel_size)` tensor in `dtype` on `device`. Exact port of
/// upstream `kaiser_sinc_filter1d` (lines 85-128). The filter sums to 1.
fn kaiser_sinc_filter1d(
    cutoff: f64,
    half_width: f64,
    kernel_size: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let is_even = kernel_size % 2 == 0;
    let half_size = kernel_size / 2;

    // Kaiser-window parameters (Kaiser's design formula).
    let delta_f = 4.0 * half_width;
    let attenuation = 2.285 * (half_size as f64 - 1.0) * std::f64::consts::PI * delta_f + 7.95;
    let beta = if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    };

    let window = kaiser_window(kernel_size, beta);

    // Time indices: half-integer grid when even, integer grid when odd.
    let time_indices: Vec<f64> = if is_even {
        (0..kernel_size)
            .map(|i| (i as f64 - half_size as f64) + 0.5)
            .collect()
    } else {
        (0..kernel_size)
            .map(|i| i as f64 - half_size as f64)
            .collect()
    };

    let taps: Vec<f32> = if cutoff == 0.0 {
        vec![0.0; kernel_size]
    } else {
        // raw = 2·cutoff · window · sinc(2·cutoff·t), then normalize to Σ=1.
        let raw: Vec<f64> = time_indices
            .iter()
            .zip(window.iter())
            .map(|(&t, &w)| 2.0 * cutoff * w * sinc(2.0 * cutoff * t))
            .collect();
        let s: f64 = raw.iter().sum();
        raw.iter().map(|&v| (v / s) as f32).collect()
    };

    Tensor::from_vec(taps, (1, 1, kernel_size), device)?.to_dtype(dtype)
}

// ---------------------------------------------------------------------------
// Replicate (edge) padding along the last (time) axis
// ---------------------------------------------------------------------------

/// PyTorch `F.pad(x, (left, right), mode="replicate")` along dim 2 of a
/// `(B, C, T)` tensor: repeat the first column `left` times on the left
/// and the last column `right` times on the right.
fn replicate_pad1d(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    let t = x.dim(D::Minus1)?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(left + 1 + right);
    if left > 0 {
        let first = x.narrow(D::Minus1, 0, 1)?;
        for _ in 0..left {
            parts.push(first.clone());
        }
    }
    parts.push(x.clone());
    if right > 0 {
        let last = x.narrow(D::Minus1, t - 1, 1)?;
        for _ in 0..right {
            parts.push(last.clone());
        }
    }
    Tensor::cat(&parts, D::Minus1)
}

// ---------------------------------------------------------------------------
// SnakeBeta
// ---------------------------------------------------------------------------

/// SnakeBeta activation with separate per-channel `alpha`/`beta` in
/// log-space. `y = x + (1/(exp(beta)+1e-9)) · sin²(x·exp(alpha))`.
///
/// Both parameters are `(C,)` tensors loaded from the checkpoint under
/// `alpha` / `beta`; they are reshaped to `(1, C, 1)` for broadcasting
/// against the `(B, C, T)` activations.
#[derive(Debug, Clone)]
struct SnakeBeta {
    /// `(1, C, 1)` — exp'd at forward.
    alpha: Tensor,
    /// `(1, C, 1)` — exp'd at forward.
    beta: Tensor,
}

impl SnakeBeta {
    /// `vb` is scoped to the SnakeBeta's `.act` namespace (so keys are
    /// `<vb>.alpha` / `<vb>.beta`).
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?.reshape((1, channels, 1))?;
        let beta = vb.get(channels, "beta")?.reshape((1, channels, 1))?;
        Ok(Self { alpha, beta })
    }

    /// `x`: `(B, C, T)`. Returns the same shape. Computed in `x`'s dtype
    /// (upstream does NOT force fp32 here).
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let exp_alpha = self.alpha.exp()?;
        let exp_beta = self.beta.exp()?;
        // 1 / (exp(beta) + 1e-9)
        let inv_beta = exp_beta.affine(1.0, SNAKE_EPS)?.recip()?;
        // sin(x · exp(alpha))²
        let sin_sq = x.broadcast_mul(&exp_alpha)?.sin()?.sqr()?;
        // x + inv_beta · sin²(...)
        let add = sin_sq.broadcast_mul(&inv_beta)?;
        x + add
    }
}

// ---------------------------------------------------------------------------
// Anti-aliased activation (TorchActivation1d)
// ---------------------------------------------------------------------------

/// `TorchActivation1d`: Kaiser-sinc upsample(×2) → SnakeBeta → Kaiser-sinc
/// downsample(×2). The filters are shared (`filter`, an `(1,1,12)`
/// tensor), broadcast to depthwise `(C,1,12)` kernels per call. Mirrors
/// upstream `UpSample1d` / `DownSample1d` / `TorchActivation1d`.
///
/// Geometry (ratio=2, kernel_size=12), all from the upstream constructors:
/// - **Up**: `pad = kernel_size/ratio − 1 = 5` (replicate, both sides);
///   `pad_left  = pad·stride + (k−s)/2     = 10 + 5 = 15`;
///   `pad_right = pad·stride + (k−s+1)/2   = 10 + 5 = 15`; slice
///   `[15:-15]`. Output is `ratio · conv_transpose1d(stride=2, groups=C)`.
/// - **Down**: even kernel → `pad_left = k/2 − 1 = 5`,
///   `pad_right = k/2 = 6` (replicate); `conv1d(stride=2, groups=C)`.
#[derive(Debug, Clone)]
struct TorchActivation1d {
    act: SnakeBeta,
    /// Shared Kaiser-sinc taps `(1, 1, 12)`. Re-broadcast to `(C, 1, 12)`
    /// per forward so the same buffer serves any channel count.
    filter: Tensor,
}

impl TorchActivation1d {
    fn new(channels: usize, filter: Tensor, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act: SnakeBeta::new(channels, vb.pp("act"))?,
            filter,
        })
    }

    /// Depthwise kernel `(C, 1, K)` from the shared `(1, 1, K)` filter.
    fn depthwise_kernel(&self, channels: usize) -> Result<Tensor> {
        self.filter
            .broadcast_as((channels, 1, AA_KERNEL_SIZE))?
            .contiguous()
    }

    /// `UpSample1d.forward` (ratio=2, kernel=12).
    fn upsample(&self, x: &Tensor) -> Result<Tensor> {
        let channels = x.dim(1)?;
        // pad = kernel_size/ratio − 1 = 12/2 − 1 = 5 (replicate, both sides)
        let padded = replicate_pad1d(x, 5, 5)?;
        let kernel = self.depthwise_kernel(channels)?;
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride: 2,
            dilation: 1,
            groups: channels,
        };
        let convt = ConvTranspose1d::new(kernel, None, cfg);
        // output = ratio · conv_transpose1d(...)
        let y = convt.forward(&padded)?.affine(2.0, 0.0)?;
        // slice [..., pad_left : -pad_right] = [..., 15 : T−15]
        let t = y.dim(D::Minus1)?;
        y.narrow(D::Minus1, 15, t - 30)
    }

    /// `DownSample1d.forward` (ratio=2, kernel=12).
    fn downsample(&self, x: &Tensor) -> Result<Tensor> {
        let channels = x.dim(1)?;
        // even kernel: pad_left = k/2 − 1 = 5, pad_right = k/2 = 6 (replicate)
        let padded = replicate_pad1d(x, 5, 6)?;
        let kernel = self.depthwise_kernel(channels)?;
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 2,
            dilation: 1,
            groups: channels,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(kernel, None, cfg);
        conv.forward(&padded)
    }

    /// upsample → SnakeBeta → downsample.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.upsample(x)?;
        let h = self.act.forward(&h)?;
        self.downsample(&h)
    }
}

// ---------------------------------------------------------------------------
// AMPBlock
// ---------------------------------------------------------------------------

/// AMPBlock: 3 residual sub-blocks. `convs1[j]` use dilation `[1,3,5][j]`
/// (pad = `(k·d − d)/2`); `convs2[j]` use dilation 1. Six
/// `TorchActivation1d(SnakeBeta(C))` activations, paired as
/// `activations[::2]` (→ act1) and `activations[1::2]` (→ act2).
///
/// Sub-block j: `h = act1·conv1·act2·conv2`, then `h = residual + h`.
#[derive(Debug, Clone)]
struct AmpBlock {
    convs1: Vec<Conv1d>,
    convs2: Vec<Conv1d>,
    /// 6 activations in checkpoint order (index 0..6). act1 = indices
    /// 0,2,4 ; act2 = indices 1,3,5.
    activations: Vec<TorchActivation1d>,
}

impl AmpBlock {
    fn new(
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
        filter: Tensor,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut convs1 = Vec::with_capacity(dilations.len());
        let vb_c1 = vb.pp("convs1");
        for (j, &d) in dilations.iter().enumerate() {
            // padding = (kernel_size·d − d) / 2
            let padding = (kernel_size * d - d) / 2;
            let cfg = Conv1dConfig {
                padding,
                stride: 1,
                dilation: d,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            convs1.push(conv1d(channels, channels, kernel_size, cfg, vb_c1.pp(j))?);
        }

        let mut convs2 = Vec::with_capacity(dilations.len());
        let vb_c2 = vb.pp("convs2");
        for j in 0..dilations.len() {
            // convs2 are all dilation 1 → padding = (kernel_size − 1) / 2
            let padding = (kernel_size - 1) / 2;
            let cfg = Conv1dConfig {
                padding,
                stride: 1,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            };
            convs2.push(conv1d(channels, channels, kernel_size, cfg, vb_c2.pp(j))?);
        }

        // num_layers = len(convs1) + len(convs2) = 6 activations.
        let num_layers = convs1.len() + convs2.len();
        let mut activations = Vec::with_capacity(num_layers);
        let vb_act = vb.pp("activations");
        for k in 0..num_layers {
            activations.push(TorchActivation1d::new(
                channels,
                filter.clone(),
                vb_act.pp(k),
            )?);
        }

        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // acts1 = activations[::2], acts2 = activations[1::2]
        let mut h = x.clone();
        for j in 0..self.convs1.len() {
            let residual = h.clone();
            let act1 = &self.activations[2 * j];
            let act2 = &self.activations[2 * j + 1];
            let t = act1.forward(&h)?;
            let t = self.convs1[j].forward(&t)?;
            let t = act2.forward(&t)?;
            let t = self.convs2[j].forward(&t)?;
            h = (residual + t)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// BigVgan
// ---------------------------------------------------------------------------

/// Qwen2.5-Omni Token2Wav BigVGAN vocoder. Construct from the released
/// checkpoint's `token2wav.code2wav_bigvgan_model.*` slice of the
/// VarBuilder.
#[derive(Debug, Clone)]
pub struct BigVgan {
    conv_pre: Conv1d,
    /// `ups[i]` is the single ConvTranspose1d under the extra ModuleList
    /// wrap (key `ups.{i}.0.weight`). 6 levels.
    ups: Vec<ConvTranspose1d>,
    /// 18 AMPBlocks, flat. Level `i` owns blocks `[3i, 3i+1, 3i+2]`.
    resblocks: Vec<AmpBlock>,
    activation_post: TorchActivation1d,
    conv_post: Conv1d,
    /// resblocks per upsample level (= len(resblock_kernel_sizes) = 3).
    num_residual_blocks: usize,
    /// number of upsample levels (= len(upsample_rates) = 6).
    num_upsample_layers: usize,
}

impl BigVgan {
    /// Build from config + a VarBuilder scoped to
    /// `token2wav.code2wav_bigvgan_model`.
    pub fn new(cfg: &BigVganConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Shared anti-aliasing filter. Up and down use identical params
        // here (cutoff = 0.5/ratio = 0.25, half_width = 0.6/ratio = 0.3,
        // kernel_size = 12), so one filter serves every TorchActivation1d.
        let aa_filter = kaiser_sinc_filter1d(0.25, 0.3, AA_KERNEL_SIZE, dtype, &device)?;

        // conv_pre: Conv1d(mel_dim → upsample_initial_channel, k=7, pad=3, bias=true)
        let conv_pre_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv_pre = conv1d(
            cfg.mel_dim,
            cfg.upsample_initial_channel,
            7,
            conv_pre_cfg,
            vb.pp("conv_pre"),
        )?;

        let num_upsample_layers = cfg.upsample_rates.len();
        let num_residual_blocks = cfg.resblock_kernel_sizes.len();

        // Upsamplers. Each lives under an extra ModuleList → key
        // `ups.{i}.0.*`. groups=1, bias=true, padding=(k−s)/2.
        let vb_ups = vb.pp("ups");
        let mut ups = Vec::with_capacity(num_upsample_layers);
        for (i, (&stride, &kernel_size)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let c_in = cfg.upsample_initial_channel / (1 << i);
            let c_out = cfg.upsample_initial_channel / (1 << (i + 1));
            let cfg_t = ConvTranspose1dConfig {
                padding: (kernel_size - stride) / 2,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            };
            // The extra `.0` ModuleList index.
            let convt = conv_transpose1d(c_in, c_out, kernel_size, cfg_t, vb_ups.pp(i).pp(0))?;
            ups.push(convt);
        }

        // 18 AMPBlocks: for each upsample level, 3 blocks (one per
        // (kernel_size, dilation) pair), flattened in checkpoint order.
        let vb_res = vb.pp("resblocks");
        let mut resblocks = Vec::with_capacity(num_upsample_layers * num_residual_blocks);
        let mut block_idx = 0usize;
        for layer_idx in 0..num_upsample_layers {
            let channels = cfg.upsample_initial_channel / (1 << (layer_idx + 1));
            for (kernel_size, dilation) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
            {
                resblocks.push(AmpBlock::new(
                    channels,
                    *kernel_size,
                    dilation,
                    aa_filter.clone(),
                    vb_res.pp(block_idx),
                )?);
                block_idx += 1;
            }
        }

        // activation_post: TorchActivation1d(SnakeBeta(final_channels)).
        let final_channels = cfg.upsample_initial_channel / (1 << num_upsample_layers);
        let activation_post =
            TorchActivation1d::new(final_channels, aa_filter.clone(), vb.pp("activation_post"))?;

        // conv_post: Conv1d(final_channels → 1, k=7, pad=3, bias=FALSE).
        // We build it via `conv1d` then drop the bias so the key
        // `conv_post.bias` is never requested (matches the released
        // index, which has only `conv_post.weight`).
        let conv_post_cfg = Conv1dConfig {
            padding: 3,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv_post_w =
            vb.pp("conv_post")
                .get((1, final_channels, 7), "weight")?;
        let conv_post = Conv1d::new(conv_post_w, None, conv_post_cfg);

        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            num_residual_blocks,
            num_upsample_layers,
        })
    }

    /// `process_mel_spectrogram` (upstream lines 3336-3348):
    ///
    /// ```text
    ///   amp = exp(mel)
    ///   db  = 20·log10(max(amp, 10^(min_db/20·ln10))) − 20
    ///   out = clamp(2·(db − min_db)/(−min_db) − 1, −1, 1)      (max=1)
    /// ```
    ///
    /// with `min_db = −115`. The lower clamp `max(amp, min_level)` is
    /// one-sided (no upper cap on the amplitude). `log10 = ln/ln(10)`.
    fn process_mel_spectrogram(&self, mel: &Tensor) -> Result<Tensor> {
        let ln10 = std::f64::consts::LN_10;
        // amplitude = exp(mel)
        let amplitude = mel.exp()?;
        // amplitude_to_db: min_level = exp(min_db/20 · ln10)
        let min_level = (MIN_DB / 20.0 * ln10).exp();
        // 20 · log10(clamp(amp, min=min_level)) ; clamp lower only.
        let clamped = amplitude.clamp(min_level, f64::MAX)?;
        // log10(x) = ln(x) / ln(10)
        let db = clamped.log()?.affine(20.0 / ln10, 0.0)?;
        // amplitude_to_db(...) − 20
        let db = db.affine(1.0, -20.0)?;
        // normalize_spectrogram(db, max_value=1, min_db=−115):
        //   clamp(2·max·((x − min_db)/(−min_db)) − max, −max, max)
        //   = clamp(2·(x + 115)/115 − 1, −1, 1)
        let max_value = 1.0f64;
        let scale = 2.0 * max_value / (-MIN_DB);
        let bias = -2.0 * max_value * MIN_DB / (-MIN_DB) - max_value;
        // = scale·x + bias ; with min_db=−115, max=1 → 2/115·x + (1)
        let normed = db.affine(scale, bias)?;
        normed.clamp(-max_value, max_value)
    }

    /// Forward inference. Input `mel`: `(B, 80, T)`. Output waveform:
    /// `(B, 1, 240·T)` clamped to `[-1, 1]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let processed = self.process_mel_spectrogram(mel)?;
        let mut hidden = self.conv_pre.forward(&processed)?;

        for layer_index in 0..self.num_upsample_layers {
            // Upsample.
            hidden = self.ups[layer_index].forward(&hidden)?;

            // Sum the 3 resblock outputs for this level, then average.
            let base = layer_index * self.num_residual_blocks;
            let mut acc: Option<Tensor> = None;
            for block_index in 0..self.num_residual_blocks {
                let out = self.resblocks[base + block_index].forward(&hidden)?;
                acc = Some(match acc {
                    None => out,
                    Some(a) => (a + out)?,
                });
            }
            let summed = acc.expect("num_residual_blocks must be >= 1");
            // hidden = sum / num_residual_blocks  (AVG-sum, NOT residual add)
            hidden = (summed / self.num_residual_blocks as f64)?;
        }

        hidden = self.activation_post.forward(&hidden)?;
        let waveform = self.conv_post.forward(&hidden)?;
        waveform.clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::BigVganConfig;
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// Real released BigVGAN config (3B). Used by both the shape and the
    /// key-tree audit tests so the structure is faithful — only the time
    /// length of the test input is tiny.
    fn real_cfg() -> BigVganConfig {
        BigVganConfig {
            mel_dim: 80,
            upsample_initial_channel: 1536,
            upsample_rates: vec![5, 3, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![11, 7, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_bias_at_final: false,
        }
    }

    /// A trivially-small BigVGAN config used by the fast CPU shape test.
    /// Keeps the structural invariants (6 levels, 3 resblocks/level,
    /// kernels/strides) but shrinks the initial channel count so the
    /// random forward is laptop-cheap. Initial channel must stay
    /// divisible by 2^6 = 64.
    fn tiny_cfg() -> BigVganConfig {
        BigVganConfig {
            mel_dim: 8,
            upsample_initial_channel: 64, // → 32,16,8,4,2,1 down the levels
            upsample_rates: vec![5, 3, 2, 2, 2, 2],
            upsample_kernel_sizes: vec![11, 7, 4, 4, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            use_bias_at_final: false,
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(
                0f32,
                0.02,
                v.as_tensor().shape().clone(),
                device,
            )?)?;
        }
        Ok(())
    }

    /// **Phase 5 acceptance #1** — Kaiser-sinc filter matches the
    /// golden taps generated by `torch.kaiser_window` + `torch.sinc`
    /// (`kaiser_sinc_filter1d(0.25, 0.3, 12)`) within 1e-6. These taps
    /// were produced offline via PyTorch and hard-coded here. Also
    /// asserts Σ = 1, even-symmetry, and finiteness.
    #[test]
    fn kaiser_sinc_filter_matches_pytorch() -> Result<()> {
        let device = Device::Cpu;
        let f = kaiser_sinc_filter1d(0.25, 0.3, 12, DType::F32, &device)?;
        assert_eq!(f.dims(), &[1, 1, 12]);
        let taps = f.flatten_all()?.to_vec1::<f32>()?;

        // Golden taps from:
        //   torch.kaiser_window(12, beta, periodic=False) etc.
        let golden: [f32; 12] = [
            0.0020289647,
            0.0093894657,
            -0.0255434588,
            -0.0576573834,
            0.1285725832,
            0.4432097971,
            0.4432097971,
            0.1285725832,
            -0.0576573834,
            -0.0255434588,
            0.0093894657,
            0.0020289647,
        ];
        for (i, (g, e)) in taps.iter().zip(golden.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-6,
                "kaiser tap[{i}] = {g}, expected {e} (Δ={})",
                (g - e).abs()
            );
        }
        // Σ ≈ 1 and even-symmetric.
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "filter sum = {sum}");
        for i in 0..6 {
            assert!(
                (taps[i] - taps[11 - i]).abs() < 1e-6,
                "asymmetry at {i}: {} vs {}",
                taps[i],
                taps[11 - i]
            );
        }
        assert!(taps.iter().all(|x| x.is_finite()));

        // I0 sanity: I0(0) = 1.
        assert!((bessel_i0(0.0) - 1.0).abs() < 1e-12);
        Ok(())
    }

    /// **Phase 5 acceptance #2** — SnakeBeta at α=β=0 reduces to
    /// `y = x + sin²(x)` (since exp(0)=1, 1/(1+1e-9)≈1) within 1e-6.
    #[test]
    fn snakebeta_identity_params() -> Result<()> {
        let device = Device::Cpu;
        let channels = 4usize;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        // VarMap initializes alpha/beta to zeros on first `get`.
        let snake = SnakeBeta::new(channels, vb.pp("act"))?;
        // alpha/beta default-init to zero in VarMap → leave as-is.

        let x = Tensor::randn(0f32, 1f32, (1, channels, 7), &device)?;
        let y = snake.forward(&x)?;
        // expected = x + sin(x)²
        let expected = (&x + x.sin()?.sqr()?)?;
        let diff = (y - expected)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(diff < 1e-6, "SnakeBeta identity diff = {diff}");
        Ok(())
    }

    /// **Phase 5 acceptance #3** — anti-aliased TorchActivation1d is
    /// shape-preserving: `(B, C, T)` in → `(B, C, T)` out. The Kaiser
    /// up(×2)/down(÷2) round-trip must restore the time length exactly.
    #[test]
    fn torch_activation1d_shape_preserving() -> Result<()> {
        let device = Device::Cpu;
        let channels = 4usize;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let filter = kaiser_sinc_filter1d(0.25, 0.3, AA_KERNEL_SIZE, DType::F32, &device)?;
        let act = TorchActivation1d::new(channels, filter, vb.pp("0"))?;
        randomize(&vm, &device)?;

        for t in [16usize, 17, 64, 100] {
            let x = Tensor::randn(0f32, 1f32, (2, channels, t), &device)?;
            let y = act.forward(&x)?;
            assert_eq!(y.dims(), &[2, channels, t], "T={t} not preserved");
            assert!(y.flatten_all()?.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
        }
        Ok(())
    }

    /// **Phase 5 acceptance #4** — full BigVGAN forward on the tiny
    /// config produces `(B, 1, 240·T)` with all samples in `[-1, 1]`.
    /// 240 = 5·3·2·2·2·2.
    #[test]
    fn bigvgan_forward_shape_and_range() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = BigVgan::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let t = 10usize;
        let mel = Tensor::randn(0f32, 1f32, (1, cfg.mel_dim, t), &device)?;
        let wav = model.forward(&mel)?;
        assert_eq!(wav.dims(), &[1, 1, 240 * t]);
        let v = wav.flatten_all()?.to_vec1::<f32>()?;
        assert!(v.iter().all(|x| x.is_finite()), "non-finite samples");
        let (min, max) = v
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| {
                (mn.min(x), mx.max(x))
            });
        assert!(min >= -1.0 && max <= 1.0, "range [{min}, {max}] escapes [-1,1]");
        Ok(())
    }

    /// **Phase 5 acceptance #5** — the VarBuilder key tree we request
    /// matches the released `token2wav.code2wav_bigvgan_model.*` layout
    /// exactly: 449 keys, no extras (the recomputed Kaiser-sinc filters
    /// must NOT appear), no missing. This is the test that catches the
    /// `ups.{i}.0` extra-ModuleList trap and the `conv_post` no-bias trap.
    #[test]
    fn bigvgan_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = real_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _model = BigVgan::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let names: std::collections::BTreeSet<String> =
            vm.data().lock().unwrap().keys().cloned().collect();

        let mut want: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        // conv_pre (bias) + conv_post (NO bias).
        want.insert("conv_pre.weight".into());
        want.insert("conv_pre.bias".into());
        want.insert("conv_post.weight".into());
        // activation_post SnakeBeta.
        want.insert("activation_post.act.alpha".into());
        want.insert("activation_post.act.beta".into());
        // ups.{i}.0.{weight,bias} — extra ModuleList index 0.
        for i in 0..cfg.upsample_rates.len() {
            want.insert(format!("ups.{i}.0.weight"));
            want.insert(format!("ups.{i}.0.bias"));
        }
        // resblocks.{i}.{convs1,convs2}.{j}.{weight,bias} +
        // resblocks.{i}.activations.{k}.act.{alpha,beta}.
        let num_blocks = cfg.upsample_rates.len() * cfg.resblock_kernel_sizes.len();
        for i in 0..num_blocks {
            for j in 0..3 {
                want.insert(format!("resblocks.{i}.convs1.{j}.weight"));
                want.insert(format!("resblocks.{i}.convs1.{j}.bias"));
                want.insert(format!("resblocks.{i}.convs2.{j}.weight"));
                want.insert(format!("resblocks.{i}.convs2.{j}.bias"));
            }
            for k in 0..6 {
                want.insert(format!("resblocks.{i}.activations.{k}.act.alpha"));
                want.insert(format!("resblocks.{i}.activations.{k}.act.beta"));
            }
        }

        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );
        // Lock the documented total: 449 keys.
        assert_eq!(want.len(), 449, "expected 449 BigVGAN keys, got {}", want.len());
        Ok(())
    }

    /// **Phase 5 acceptance #6** — real-weight smoke against
    /// `Qwen/Qwen2.5-Omni-3B`'s `token2wav.code2wav_bigvgan_model.*`
    /// shard. CPU F32 (candle has no bf16 CPU kernel). `#[ignore]`'d —
    /// invoked from the Lambda smoke runner with `QWEN_OMNI_3B_DIR` set.
    /// Input mel `(1, 80, 100)` → waveform `(1, 1, 24000)` in `[-1, 1]`.
    #[test]
    #[ignore]
    fn real_weight_bigvgan_loads() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
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

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let model = BigVgan::new(
            &cfg.token2wav_config.bigvgan_config,
            vb.pp("token2wav").pp("code2wav_bigvgan_model"),
        )
        .expect("construct BigVgan from real weights");

        // 100 mel frames → 240·100 = 24000 samples = 1 s @ 24 kHz.
        let mel = Tensor::randn(0f32, 1f32, (1usize, 80usize, 100usize), &device).unwrap();
        let wav = model.forward(&mel).expect("forward on real weights");
        assert_eq!(wav.dims(), &[1, 1, 24000]);
        let v = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (min, max) = v
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &x| {
                (mn.min(x), mx.max(x))
            });
        eprintln!("real_weight_bigvgan_loads: range = [{min}, {max}], len = {}", v.len());
        assert!(min >= -1.0, "min {min} < -1");
        assert!(max <= 1.0, "max {max} > 1");
    }

    /// **Phase 5 GPU acceptance** — same as the CPU smoke but native
    /// BF16 on CUDA. Lambda + `--features cuda` only.
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn real_weight_bigvgan_loads_cuda_bf16() {
        use crate::models::qwen2_5_omni::config::OmniConfig;
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

        let device = Device::new_cuda(0).expect("cuda device 0 (run on a GPU box)");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::BF16, &device)
                .expect("mmap safetensors")
        };
        let model = BigVgan::new(
            &cfg.token2wav_config.bigvgan_config,
            vb.pp("token2wav").pp("code2wav_bigvgan_model"),
        )
        .expect("construct BigVgan from real weights");

        let mel = Tensor::randn(0f32, 1f32, (1usize, 80usize, 100usize), &device)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let wav = model.forward(&mel).expect("forward on real weights (cuda bf16)");
        assert_eq!(wav.dims(), &[1, 1, 24000]);
        assert_eq!(wav.dtype(), DType::BF16);
    }
}

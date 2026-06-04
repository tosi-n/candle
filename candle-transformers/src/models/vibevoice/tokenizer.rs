//! Acoustic and semantic audio tokenizers for VibeVoice.
//!
//! Pure-candle port of `vibevoice/modular/modular_vibevoice_tokenizer.py`.
//! σ-VAE-style continuous tokenizer operating at 7.5 Hz against a 24 kHz
//! waveform (3200× downsample for the 1.5B variant).
//!
//! ## Inference scope
//!
//! At inference VibeVoice's σ-VAE collapses to a deterministic
//! encoder–decoder: the encoder returns the mean only (no `logvar`
//! head), and `sample()` with `dist_type='fix'` reduces to
//! `mean + fix_std * eps`. Since this is a Rust *inference* port, we
//! drop the noise injection and return the mean directly. Callers can
//! add Gaussian noise externally if they need it. Same for KL —
//! training-only and dropped.
//!
//! Streaming caches, weight-norm, group-norm, drop-path, and all
//! reflect-pad branches are also out of scope. We support **only** the
//! configuration the released checkpoint ships:
//! `pad_mode='constant'`, `causal=True`, `conv_norm='none'`,
//! `disable_last_norm=True`, `layernorm='RMSNorm'`,
//! `mixer_layer='depthwise_conv'`, `bias=True`.
//!
//! ## Weight-key map
//!
//! Trace each module level so the `vb.pp(...)` chain is unambiguous.
//! Multi-level paths come from PyTorch wrapping `nn.Sequential` /
//! `NormConv1d` / `Convlayer` around the bare conv:
//!
//! ```text
//! TokenizerEncoder
//! ├── downsample_layers
//! │   └── {i}.0                         ← Sequential index .0 (single SConv1d)
//! │       └── conv.conv                 ← SConv1d→NormConv1d→Conv1d
//! │           ├── weight
//! │           └── bias
//! ├── stages
//! │   └── {i}.{j}                       ← Sequential index .{j} per Block1D
//! │       ├── norm.weight               (ConvRMSNorm, shape (C,))
//! │       ├── ffn_norm.weight
//! │       ├── mixer.conv.conv.conv.{weight,bias}   ← four levels
//! │       ├── ffn.linear1.{weight,bias}
//! │       ├── ffn.linear2.{weight,bias}
//! │       ├── gamma                      (shape (C,))
//! │       └── ffn_gamma                  (shape (C,))
//! ├── norm                               (Identity — no params loaded)
//! └── head.conv.conv.{weight,bias}       ← three levels (no Sequential wrapper)
//!
//! TokenizerDecoder mirrors this but transpose layers use
//!   upsample_layers.{i}.0.convtr.convtr.{weight,bias}
//! ```

use candle::{DType, Result, Tensor, D};
use candle_nn::{
    conv1d, conv_transpose1d, linear, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig,
    Linear, Module, VarBuilder,
};

use super::config::VibeVoiceTokenizerConfig;

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// `ConvRMSNorm` — root-mean-square norm over the channel dimension of
/// a `(B, C, T)` conv-style tensor.
///
/// Mirrors the Python `ConvRMSNorm`:
///
/// ```python
/// x = x.transpose(1, 2)   # (B, C, T) → (B, T, C)
/// x = x * rsqrt(mean(x², -1, keepdim=True) + eps)
/// x = x * weight
/// x = x.transpose(1, 2)   # (B, T, C) → (B, C, T)
/// ```
///
/// Naively running `mean(-1)` over a `(B, C, T)` tensor would normalize
/// over time — wrong axis. The transpose is mandatory.
#[derive(Debug, Clone)]
struct ConvRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ConvRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x is (B, C, T). Move channel to last, normalise, move back.
        let in_dtype = x.dtype();
        let xt = x.transpose(1, 2)?.to_dtype(DType::F32)?;
        let variance = xt.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_rms = (variance + self.eps)?.sqrt()?.recip()?;
        let normalized = xt.broadcast_mul(&inv_rms)?;
        let scaled = normalized.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?;
        scaled.to_dtype(in_dtype)?.transpose(1, 2)
    }
}

// ---------------------------------------------------------------------------
// Padding helpers
// ---------------------------------------------------------------------------

/// Compute the extra zero padding needed so that a stride-`s` conv
/// covers the full input. Matches Python's `get_extra_padding_for_conv1d`:
///
/// ```text
///   n_frames = (T - k + p_total) / s + 1
///   ideal_T  = (ceil(n_frames) - 1) * s + (k - p_total)
///   extra    = ideal_T - T
/// ```
///
/// For input lengths divisible by `s` (the case we exercise in tests),
/// `extra` is always 0.
fn extra_padding_for_conv1d(
    t_in: usize,
    kernel_size: usize,
    stride: usize,
    padding_total: usize,
) -> usize {
    // We use floats here only because the Python original does, and the
    // domain is tiny — conversion is exact.
    let length = t_in as f64;
    let n_frames = (length - kernel_size as f64 + padding_total as f64) / stride as f64 + 1.0;
    let ideal =
        (n_frames.ceil() - 1.0) * stride as f64 + (kernel_size as f64 - padding_total as f64);
    let ideal = ideal as usize;
    ideal.saturating_sub(t_in)
}

/// `pad1d` with constant=0 mode only. Pads a `(B, C, T)` tensor on the
/// left and right of the time axis. Reflect mode is intentionally
/// unsupported — the released VibeVoice checkpoint ships
/// `pad_mode='constant'`.
fn pad1d_constant(x: &Tensor, padding_left: usize, padding_right: usize) -> Result<Tensor> {
    if padding_left == 0 && padding_right == 0 {
        return Ok(x.clone());
    }
    x.pad_with_zeros(D::Minus1, padding_left, padding_right)
}

/// Trim left/right pad from the time axis. Mirrors `unpad1d`.
fn unpad1d(x: &Tensor, padding_left: usize, padding_right: usize) -> Result<Tensor> {
    if padding_left == 0 && padding_right == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(D::Minus1)?;
    let end = t - padding_right;
    x.narrow(D::Minus1, padding_left, end - padding_left)
}

// ---------------------------------------------------------------------------
// SConv1d: causal-padding-aware Conv1d
// ---------------------------------------------------------------------------

/// `SConv1d` = padding-aware Conv1d.
///
/// For causal mode, padding is `(padding_total + extra, 0)` (left only),
/// where `padding_total = (k - 1) * d - (s - 1)`. Off-by-one here
/// accumulates into garbage waveforms — the encoder/decoder shape unit
/// tests guard against that.
///
/// Weight path under VarBuilder: `conv.conv.{weight,bias}`
/// (NormConv1d → Conv1d).
#[derive(Debug, Clone)]
struct SConv1d {
    conv: Conv1d,
    causal: bool,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    padding_total: usize,
}

impl SConv1d {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        causal: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        // Weight nesting: SConv1d → NormConv1d (`conv`) → Conv1d (`conv`).
        let inner_vb = vb.pp("conv").pp("conv");
        let conv = if bias {
            conv1d(in_channels, out_channels, kernel_size, cfg, inner_vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, cfg, inner_vb)?
        };
        let padding_total = (kernel_size - 1) * dilation - (stride - 1);
        Ok(Self {
            conv,
            causal,
            kernel_size,
            stride,
            dilation,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t_in = x.dim(D::Minus1)?;
        let extra =
            extra_padding_for_conv1d(t_in, self.kernel_size, self.stride, self.padding_total);
        let _ = self.dilation; // documented; conv layer handles it
        let padded = if self.causal {
            // Causal: all padding on the left, plus extra on the right
            // for stride alignment.
            pad1d_constant(x, self.padding_total, extra)?
        } else {
            // Symmetric padding for non-causal: split padding_total
            // ⌈/2⌉ on left and ⌊/2⌋ + extra on the right.
            let pad_right = self.padding_total / 2;
            let pad_left = self.padding_total - pad_right;
            pad1d_constant(x, pad_left, pad_right + extra)?
        };
        self.conv.forward(&padded)
    }
}

// ---------------------------------------------------------------------------
// SConvTranspose1d: causal-aware ConvTranspose1d with right-trim
// ---------------------------------------------------------------------------

/// `SConvTranspose1d` = transposed conv with output trimming.
///
/// Forward applies the bare `ConvTranspose1d` then trims
/// `padding_total = k - s` samples off the right (causal) or
/// symmetrically (non-causal). `trim_right_ratio` defaults to 1.0 in
/// VibeVoice, meaning the entire trim is taken from the right edge.
///
/// Weight path: `convtr.convtr.{weight,bias}`.
#[derive(Debug, Clone)]
struct SConvTranspose1d {
    convtr: ConvTranspose1d,
    causal: bool,
    trim_right_ratio: f64,
    padding_total: usize,
}

impl SConvTranspose1d {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        causal: bool,
        trim_right_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&trim_right_ratio) {
            candle::bail!("trim_right_ratio must lie in [0, 1], got {trim_right_ratio}");
        }
        if !causal && (trim_right_ratio - 1.0).abs() > 1e-12 {
            candle::bail!("trim_right_ratio != 1.0 only makes sense for causal convolutions");
        }
        let cfg = ConvTranspose1dConfig {
            padding: 0,
            output_padding: 0,
            stride,
            dilation: 1,
            groups: 1,
        };
        let inner_vb = vb.pp("convtr").pp("convtr");
        let convtr = if bias {
            conv_transpose1d(in_channels, out_channels, kernel_size, cfg, inner_vb)?
        } else {
            candle_nn::conv_transpose1d_no_bias(
                in_channels,
                out_channels,
                kernel_size,
                cfg,
                inner_vb,
            )?
        };
        let padding_total = kernel_size - stride;
        Ok(Self {
            convtr,
            causal,
            trim_right_ratio,
            padding_total,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.convtr.forward(x)?;
        let (pad_left, pad_right) = if self.causal {
            // ceil(padding_total * trim_right_ratio) — but trim_right_ratio
            // is 1.0 in VibeVoice, so this is just padding_total on the
            // right.
            let pad_right = (self.padding_total as f64 * self.trim_right_ratio).ceil() as usize;
            let pad_left = self.padding_total - pad_right;
            (pad_left, pad_right)
        } else {
            let pad_right = self.padding_total / 2;
            let pad_left = self.padding_total - pad_right;
            (pad_left, pad_right)
        };
        unpad1d(&y, pad_left, pad_right)
    }
}

// ---------------------------------------------------------------------------
// Block1D: ConvNeXt-style residual block
// ---------------------------------------------------------------------------

/// FFN: Linear → exact GELU → Linear. Bias defaults to True per the
/// outer Block1D config.
#[derive(Debug, Clone)]
struct Ffn {
    linear1: Linear,
    linear2: Linear,
}

impl Ffn {
    fn new(embed_dim: usize, ffn_dim: usize, bias: bool, vb: VarBuilder) -> Result<Self> {
        let linear1 = if bias {
            linear(embed_dim, ffn_dim, vb.pp("linear1"))?
        } else {
            candle_nn::linear_no_bias(embed_dim, ffn_dim, vb.pp("linear1"))?
        };
        let linear2 = if bias {
            linear(ffn_dim, embed_dim, vb.pp("linear2"))?
        } else {
            candle_nn::linear_no_bias(ffn_dim, embed_dim, vb.pp("linear2"))?
        };
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.linear1.forward(x)?;
        // ACT2FN["gelu"] in HF is exact erf-based GELU (approximate='none').
        let h = h.gelu_erf()?;
        self.linear2.forward(&h)
    }
}

/// `Block1D` — ConvNeXt-style block.
///
/// 1. `r = x; x = norm(x); x = mixer(x); x = x * gamma; x = r + x`
/// 2. `r = x; x = ffn_norm(x); x = ffn(x.permute); x = x * ffn_gamma; x = r + x`
///
/// Mixer is a depthwise conv (`groups = dim`) wrapped in a `Convlayer`
/// → `SConv1d`. Per-channel scaling `gamma` / `ffn_gamma` is loaded
/// from the checkpoint (init `1e-6`).
#[derive(Debug, Clone)]
struct Block1D {
    norm: ConvRmsNorm,
    ffn_norm: ConvRmsNorm,
    mixer: SConv1d,
    ffn: Ffn,
    gamma: Option<Tensor>,
    ffn_gamma: Option<Tensor>,
}

impl Block1D {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        kernel_size: usize,
        layernorm_eps: f64,
        layer_scale_init_value: f64,
        causal: bool,
        bias: bool,
        ffn_expansion: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm = ConvRmsNorm::new(dim, layernorm_eps, vb.pp("norm"))?;
        let ffn_norm = ConvRmsNorm::new(dim, layernorm_eps, vb.pp("ffn_norm"))?;
        // Block1D's mixer is a Convlayer wrapping SConv1d:
        // mixer.conv → SConv1d. SConv1d itself adds two more levels.
        let mixer = SConv1d::new(
            dim,
            dim,
            kernel_size,
            /* stride */ 1,
            /* dilation */ 1,
            /* groups */ dim, // depthwise
            bias,
            causal,
            vb.pp("mixer").pp("conv"),
        )?;
        let ffn = Ffn::new(dim, ffn_expansion * dim, bias, vb.pp("ffn"))?;
        let (gamma, ffn_gamma) = if layer_scale_init_value > 0.0 {
            (Some(vb.get(dim, "gamma")?), Some(vb.get(dim, "ffn_gamma")?))
        } else {
            (None, None)
        };
        Ok(Self {
            norm,
            ffn_norm,
            mixer,
            ffn,
            gamma,
            ffn_gamma,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mixer branch (operates on (B, C, T)).
        let r = x.clone();
        let mut h = self.norm.forward(x)?;
        h = self.mixer.forward(&h)?;
        if let Some(gamma) = &self.gamma {
            // gamma is (C,); broadcast onto (B, C, T) by inserting axes
            // for batch and time.
            let g = gamma.reshape((1, gamma.dim(0)?, 1))?;
            h = h.broadcast_mul(&g)?;
        }
        let x = (r + h)?;

        // FFN branch — operates on (B, T, C); permute in and out.
        let r = x.clone();
        let h = self.ffn_norm.forward(&x)?;
        let h = h.permute((0, 2, 1))?; // (B, C, T) → (B, T, C)
        let h = self.ffn.forward(&h)?;
        let mut h = h.permute((0, 2, 1))?; // back to (B, C, T)
        if let Some(gamma) = &self.ffn_gamma {
            let g = gamma.reshape((1, gamma.dim(0)?, 1))?;
            h = h.broadcast_mul(&g)?;
        }
        r + h
    }
}

// ---------------------------------------------------------------------------
// TokenizerEncoder
// ---------------------------------------------------------------------------

/// Parse the dash-separated depths string ("3-3-3-3-3-3-8" → vec).
fn parse_depths(s: &str) -> Result<Vec<usize>> {
    s.split('-')
        .map(|p| {
            p.parse::<usize>()
                .map_err(|e| candle::Error::Msg(format!("invalid depth string {s:?}: {e}")))
        })
        .collect()
}

/// Encoder: `(B, channels, T)` → `(B, vae_dim, T / prod(ratios))`.
///
/// Layout per stage `i ∈ 0..len(depths)`:
/// 1. `downsample_layers[i]` — stem at i=0 (stride-1, k=7), then
///    downsample SConv1d(k=2r, s=r) for each ratio.
/// 2. `stages[i]` — `depths[i]` Block1D blocks.
///
/// Final `head` SConv1d(k=last_kernel_size=7, s=1) projects to `vae_dim`.
#[derive(Debug, Clone)]
struct TokenizerEncoder {
    downsample_layers: Vec<SConv1d>,
    stages: Vec<Vec<Block1D>>,
    head: SConv1d,
}

impl TokenizerEncoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        channels: usize,
        dimension: usize, // = vae_dim
        n_filters: usize,
        ratios_input: &[usize],
        depths: &[usize],
        causal: bool,
        bias: bool,
        layernorm_eps: f64,
        layer_scale_init_value: f64,
        kernel_size: usize,
        last_kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if depths.len() != ratios_input.len() + 1 {
            candle::bail!(
                "depths length ({}) must equal ratios length + 1 ({})",
                depths.len(),
                ratios_input.len() + 1
            );
        }
        // The Python encoder reverses ratios internally.
        let ratios: Vec<usize> = ratios_input.iter().rev().copied().collect();

        let mut downsample_layers: Vec<SConv1d> = Vec::with_capacity(depths.len());
        // Stem: (channels → n_filters), kernel=7, stride=1 — wrapped in
        // an `nn.Sequential` whose `.0` child is the SConv1d.
        let stem_vb = vb.pp("downsample_layers").pp("0").pp("0");
        let stem = SConv1d::new(
            channels,
            n_filters,
            kernel_size,
            1,
            1,
            1,
            bias,
            causal,
            stem_vb,
        )?;
        downsample_layers.push(stem);
        for (i, &r) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << i);
            let out_ch = n_filters * (1 << (i + 1));
            // downsample_layers[i + 1].0 (Sequential index 0)
            let dvb = vb.pp("downsample_layers").pp(format!("{}", i + 1)).pp("0");
            let layer = SConv1d::new(in_ch, out_ch, r * 2, r, 1, 1, bias, causal, dvb)?;
            downsample_layers.push(layer);
        }

        let mut stages: Vec<Vec<Block1D>> = Vec::with_capacity(depths.len());
        for (i, &depth) in depths.iter().enumerate() {
            let dim = n_filters * (1 << i);
            let mut blocks = Vec::with_capacity(depth);
            for j in 0..depth {
                let bvb = vb.pp("stages").pp(format!("{i}")).pp(format!("{j}"));
                let block = Block1D::new(
                    dim,
                    /* kernel_size */ 7,
                    layernorm_eps,
                    layer_scale_init_value,
                    causal,
                    bias,
                    /* ffn_expansion */ 4,
                    bvb,
                )?;
                blocks.push(block);
            }
            stages.push(blocks);
        }

        // disable_last_norm = true → norm is Identity; no params.
        let final_dim = n_filters * (1 << (depths.len() - 1));
        let head = SConv1d::new(
            final_dim,
            dimension,
            last_kernel_size,
            1,
            1,
            1,
            bias,
            causal,
            vb.pp("head"),
        )?;

        Ok(Self {
            downsample_layers,
            stages,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, ds) in self.downsample_layers.iter().enumerate() {
            h = ds.forward(&h)?;
            for block in &self.stages[i] {
                h = block.forward(&h)?;
            }
        }
        // Identity norm — skip.
        self.head.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// TokenizerDecoder
// ---------------------------------------------------------------------------

/// Decoder: `(B, vae_dim, T_tokens)` → `(B, channels, T_tokens · prod(ratios))`.
///
/// Layout per stage `i ∈ 0..len(depths)`:
/// 1. `upsample_layers[0]` — stem SConv1d (vae_dim → widest hidden,
///    `n_filters * 2^(L-1)` where L = len(depths)), kernel=7, stride=1.
/// 2. For `i ∈ 1..L`: `upsample_layers[i]` is an SConvTranspose1d that
///    upsamples by `ratios[i-1]`.
/// 3. `stages[i]` — `depths[i]` Block1D blocks at width
///    `n_filters * 2^(L-1-i)` (so stage 0 is widest = 2048 for the 1.5B variant).
///
/// **Decoder depths come in pre-reversed**: the wrapper above passes
/// `reversed(encoder_depths)` so encoder `[3,3,3,3,3,3,8]` → decoder
/// `[8,3,3,3,3,3,3]`. The reversal is the caller's job, not ours; we
/// consume `depths` in the order given.
#[derive(Debug, Clone)]
struct TokenizerDecoder {
    stem: SConv1d,
    upsample_layers: Vec<SConvTranspose1d>,
    stages: Vec<Vec<Block1D>>,
    head: SConv1d,
}

impl TokenizerDecoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        channels: usize,
        dimension: usize, // = vae_dim
        n_filters: usize,
        ratios: &[usize],
        depths: &[usize],
        causal: bool,
        bias: bool,
        layernorm_eps: f64,
        layer_scale_init_value: f64,
        kernel_size: usize,
        last_kernel_size: usize,
        trim_right_ratio: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        if depths.len() != ratios.len() + 1 {
            candle::bail!(
                "decoder depths length ({}) must equal ratios length + 1 ({})",
                depths.len(),
                ratios.len() + 1
            );
        }
        let l = depths.len();
        let widest = n_filters * (1 << (l - 1));

        // Stem SConv1d at upsample_layers.0.0
        let stem_vb = vb.pp("upsample_layers").pp("0").pp("0");
        let stem = SConv1d::new(
            dimension,
            widest,
            kernel_size,
            1,
            1,
            1,
            bias,
            causal,
            stem_vb,
        )?;

        // Upsample layers at upsample_layers.{1..L}.0
        let mut upsample_layers: Vec<SConvTranspose1d> = Vec::with_capacity(ratios.len());
        for (i, &r) in ratios.iter().enumerate() {
            let in_ch = n_filters * (1 << (l - 1 - i));
            let out_ch = n_filters * (1 << (l - 1 - i - 1));
            let uvb = vb.pp("upsample_layers").pp(format!("{}", i + 1)).pp("0");
            let layer = SConvTranspose1d::new(
                in_ch,
                out_ch,
                r * 2,
                r,
                bias,
                causal,
                trim_right_ratio,
                uvb,
            )?;
            upsample_layers.push(layer);
        }

        // Stages — stage i has channel width `n_filters * 2^(L-1-i)`.
        let mut stages: Vec<Vec<Block1D>> = Vec::with_capacity(l);
        for (i, &depth) in depths.iter().enumerate() {
            let dim = n_filters * (1 << (l - 1 - i));
            let mut blocks = Vec::with_capacity(depth);
            for j in 0..depth {
                let bvb = vb.pp("stages").pp(format!("{i}")).pp(format!("{j}"));
                let block = Block1D::new(
                    dim,
                    7,
                    layernorm_eps,
                    layer_scale_init_value,
                    causal,
                    bias,
                    4,
                    bvb,
                )?;
                blocks.push(block);
            }
            stages.push(blocks);
        }

        // disable_last_norm = true → Identity, no params.
        let head = SConv1d::new(
            n_filters,
            channels,
            last_kernel_size,
            1,
            1,
            1,
            bias,
            causal,
            vb.pp("head"),
        )?;

        Ok(Self {
            stem,
            upsample_layers,
            stages,
            head,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Stage 0: stem + first stage of blocks.
        let mut h = self.stem.forward(x)?;
        for block in &self.stages[0] {
            h = block.forward(&h)?;
        }
        // Stages 1..L: each upsample_layers[i-1] then stages[i].
        for (i, up) in self.upsample_layers.iter().enumerate() {
            h = up.forward(&h)?;
            for block in &self.stages[i + 1] {
                h = block.forward(&h)?;
            }
        }
        // Identity norm — skip.
        self.head.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Public types: AcousticTokenizer / SemanticTokenizer
// ---------------------------------------------------------------------------

const STEM_KERNEL: usize = 7;
const HEAD_KERNEL: usize = 7;
const TRIM_RIGHT_RATIO: f64 = 1.0;

/// σ-VAE acoustic tokenizer: encodes 24 kHz waveform → 7.5 Hz continuous
/// tokens; decodes back to 24 kHz.
///
/// At inference we treat the σ-VAE as deterministic: `encode` returns
/// the mean only. The encoder ratios are reversed inside the encoder
/// per the upstream Python; decoder uses the original order.
///
/// Decoder depths are derived as `reversed(encoder_depths)`, matching
/// the wrapper logic in `VibeVoiceAcousticTokenizerModel.__init__`.
#[derive(Debug, Clone)]
pub struct AcousticTokenizer {
    config: VibeVoiceTokenizerConfig,
    encoder: TokenizerEncoder,
    decoder: TokenizerDecoder,
}

impl AcousticTokenizer {
    pub fn new(config: VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        Self::validate_config(&config)?;
        let depths = parse_depths(&config.depths)?;
        let dec_depths: Vec<usize> = depths.iter().rev().copied().collect();
        let encoder = TokenizerEncoder::new(
            config.channels,
            config.vae_dim,
            config.n_filters,
            &config.ratios,
            &depths,
            config.causal,
            /* bias */ true,
            config.layernorm_eps,
            config.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            vb.pp("encoder"),
        )?;
        let decoder = TokenizerDecoder::new(
            config.channels,
            config.vae_dim,
            config.n_filters,
            &config.ratios,
            &dec_depths,
            config.causal,
            /* bias */ true,
            config.layernorm_eps,
            config.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            TRIM_RIGHT_RATIO,
            vb.pp("decoder"),
        )?;
        Ok(Self {
            config,
            encoder,
            decoder,
        })
    }

    fn validate_config(c: &VibeVoiceTokenizerConfig) -> Result<()> {
        if c.pad_mode != "constant" {
            candle::bail!(
                "VibeVoice tokenizer Rust port only supports pad_mode='constant', got {:?}",
                c.pad_mode
            );
        }
        if !c.causal {
            candle::bail!("VibeVoice tokenizer Rust port only supports causal=true");
        }
        if c.conv_norm != "none" {
            candle::bail!(
                "VibeVoice tokenizer Rust port only supports conv_norm='none', got {:?}",
                c.conv_norm
            );
        }
        if !c.disable_last_norm {
            candle::bail!("VibeVoice tokenizer Rust port only supports disable_last_norm=true");
        }
        if c.layernorm != "RMSNorm" {
            candle::bail!(
                "VibeVoice tokenizer Rust port only supports layernorm='RMSNorm', got {:?}",
                c.layernorm
            );
        }
        Ok(())
    }

    /// Encode raw 24 kHz audio `(B, channels, T_samples)` → continuous
    /// tokens `(B, T_tokens, vae_dim)`. Returns the mean of the σ-VAE
    /// (deterministic at inference).
    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        // Encoder works in (B, C, T); output is (B, vae_dim, T/prod).
        let latents = self.encoder.forward(audio)?;
        // Public API contract: (B, T_tokens, vae_dim).
        latents.permute((0, 2, 1))
    }

    /// Decode continuous tokens `(B, T_tokens, vae_dim)` → 24 kHz audio
    /// `(B, channels, T_samples)`.
    ///
    /// Accepts either layout: if `T_tokens` axis (dim 2) already equals
    /// `vae_dim`, we treat the input as `(B, vae_dim, T_tokens)` and
    /// skip the permutation. Mirrors the Python decoder's behaviour.
    pub fn decode(&self, tokens: &Tensor) -> Result<Tensor> {
        let dim1 = tokens.dim(1)?;
        let latents = if dim1 == self.config.vae_dim {
            tokens.clone()
        } else {
            tokens.permute((0, 2, 1))?
        };
        self.decoder.forward(&latents)
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }
}

/// Semantic tokenizer — encoder-only at inference. Produces the speaker
/// conditioning embedding used to bias decoding for voice cloning /
/// multi-speaker dialogue.
#[derive(Debug, Clone)]
pub struct SemanticTokenizer {
    config: VibeVoiceTokenizerConfig,
    encoder: TokenizerEncoder,
}

impl SemanticTokenizer {
    pub fn new(config: VibeVoiceTokenizerConfig, vb: VarBuilder) -> Result<Self> {
        AcousticTokenizer::validate_config(&config)?;
        let depths = parse_depths(&config.depths)?;
        let encoder = TokenizerEncoder::new(
            config.channels,
            config.vae_dim,
            config.n_filters,
            &config.ratios,
            &depths,
            config.causal,
            /* bias */ true,
            config.layernorm_eps,
            config.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            vb.pp("encoder"),
        )?;
        Ok(Self { config, encoder })
    }

    pub fn encode(&self, audio: &Tensor) -> Result<Tensor> {
        let latents = self.encoder.forward(audio)?;
        latents.permute((0, 2, 1))
    }

    pub fn config(&self) -> &VibeVoiceTokenizerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};
    use candle_nn::VarMap;

    /// Tiny tokenizer config for shape tests.
    fn tiny_cfg(ratios: Vec<usize>, depths: &str) -> VibeVoiceTokenizerConfig {
        VibeVoiceTokenizerConfig {
            channels: 1,
            vae_dim: 8,
            n_filters: 4,
            ratios,
            depths: depths.to_string(),
            causal: true,
            pad_mode: "constant".to_string(),
            conv_norm: "none".to_string(),
            layernorm: "RMSNorm".to_string(),
            layernorm_eps: 1e-6,
            disable_last_norm: true,
            layer_scale_init_value: 1e-6,
        }
    }

    /// `SConv1d` causal padding produces the expected output length.
    ///
    /// For `(k=7, s=1, d=1, causal=True)` — same-padding behaviour:
    ///   `padding_total = 6`, all on the left. Output length = T_in.
    ///
    /// For `(k=4, s=2, d=1, causal=True)` — strided downsampler:
    ///   `padding_total = 2`. With T_in = 8 (multiple of 2),
    ///   `extra_padding = 0`. Output length = (8 + 2 - 4) / 2 + 1 = 4.
    #[test]
    fn sconv1d_causal_lengths() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let conv1 = SConv1d::new(2, 2, 7, 1, 1, 1, true, true, vb.pp("c1"))?;
        let x = Tensor::zeros((1, 2, 16), DType::F32, &device)?;
        let y = conv1.forward(&x)?;
        assert_eq!(y.dims(), &[1, 2, 16], "k=7,s=1 same-length");

        let conv2 = SConv1d::new(2, 2, 4, 2, 1, 1, true, true, vb.pp("c2"))?;
        let x = Tensor::zeros((1, 2, 8), DType::F32, &device)?;
        let y = conv2.forward(&x)?;
        assert_eq!(y.dims(), &[1, 2, 4], "k=4,s=2 halves length");
        Ok(())
    }

    /// `Block1D` smoke — random init, forward a small tensor, output shape
    /// must equal input shape (residual block is shape-preserving).
    #[test]
    fn block1d_preserves_shape() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let dim = 8;
        let block = Block1D::new(dim, 7, 1e-6, 1e-6, true, true, 4, vb.pp("blk"))?;
        let x = Tensor::randn(0.0_f32, 1.0, (2, dim, 24), &device)?;
        let y = block.forward(&x)?;
        assert_eq!(y.dims(), &[2, dim, 24]);
        Ok(())
    }

    /// Encoder forward: time axis must shrink by `prod(ratios)`.
    #[test]
    fn encoder_downsamples_by_prod_ratios() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        // ratios = [2, 2] ⇒ depths must have length 3.
        let cfg = tiny_cfg(vec![2, 2], "1-1-1");
        let prod_ratios: usize = cfg.ratios.iter().product();
        // Pick T_in divisible by prod_ratios so the ceil-padding logic
        // returns 0 extra and we get an exact factor-of-prod downsample.
        let t_in = 64;
        assert_eq!(t_in % prod_ratios, 0);

        let depths = parse_depths(&cfg.depths)?;
        let enc = TokenizerEncoder::new(
            cfg.channels,
            cfg.vae_dim,
            cfg.n_filters,
            &cfg.ratios,
            &depths,
            cfg.causal,
            true,
            cfg.layernorm_eps,
            cfg.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            vb.pp("encoder"),
        )?;
        let x = Tensor::randn(0.0_f32, 1.0, (1, cfg.channels, t_in), &device)?;
        let y = enc.forward(&x)?;
        assert_eq!(y.dims(), &[1, cfg.vae_dim, t_in / prod_ratios]);
        Ok(())
    }

    /// Decoder forward: round-trip — feed encoder output through
    /// decoder, time axis must scale back up to `T_in`.
    #[test]
    fn decoder_round_trip_shape() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let cfg = tiny_cfg(vec![2, 2], "1-1-1");
        let prod_ratios: usize = cfg.ratios.iter().product();
        let t_in = 64;

        let depths = parse_depths(&cfg.depths)?;
        let dec_depths: Vec<usize> = depths.iter().rev().copied().collect();
        let enc = TokenizerEncoder::new(
            cfg.channels,
            cfg.vae_dim,
            cfg.n_filters,
            &cfg.ratios,
            &depths,
            cfg.causal,
            true,
            cfg.layernorm_eps,
            cfg.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            vb.pp("encoder"),
        )?;
        let dec = TokenizerDecoder::new(
            cfg.channels,
            cfg.vae_dim,
            cfg.n_filters,
            &cfg.ratios,
            &dec_depths,
            cfg.causal,
            true,
            cfg.layernorm_eps,
            cfg.layer_scale_init_value,
            STEM_KERNEL,
            HEAD_KERNEL,
            TRIM_RIGHT_RATIO,
            vb.pp("decoder"),
        )?;
        let x = Tensor::randn(0.0_f32, 1.0, (1, cfg.channels, t_in), &device)?;
        let z = enc.forward(&x)?;
        assert_eq!(z.dims(), &[1, cfg.vae_dim, t_in / prod_ratios]);
        let y = dec.forward(&z)?;
        assert_eq!(y.dims(), &[1, cfg.channels, t_in]);
        Ok(())
    }

    /// AcousticTokenizer end-to-end public API smoke test:
    /// encode returns (B, T_tokens, vae_dim); decode back to (B, C, T_in).
    #[test]
    fn acoustic_tokenizer_public_api_smoke() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let cfg = tiny_cfg(vec![2, 2], "1-1-1");
        let prod_ratios: usize = cfg.ratios.iter().product();
        let t_in = 64;
        let tok = AcousticTokenizer::new(cfg.clone(), vb)?;
        let audio = Tensor::randn(0.0_f32, 1.0, (1, cfg.channels, t_in), &device)?;
        let toks = tok.encode(&audio)?;
        assert_eq!(toks.dims(), &[1, t_in / prod_ratios, cfg.vae_dim]);
        let recon = tok.decode(&toks)?;
        assert_eq!(recon.dims(), &[1, cfg.channels, t_in]);
        Ok(())
    }

    /// SemanticTokenizer encode-only API.
    #[test]
    fn semantic_tokenizer_encode_only() -> Result<()> {
        let device = Device::Cpu;
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);

        let cfg = tiny_cfg(vec![2, 2], "1-1-1");
        let prod_ratios: usize = cfg.ratios.iter().product();
        let t_in = 64;
        let tok = SemanticTokenizer::new(cfg.clone(), vb)?;
        let audio = Tensor::randn(0.0_f32, 1.0, (1, cfg.channels, t_in), &device)?;
        let toks = tok.encode(&audio)?;
        assert_eq!(toks.dims(), &[1, t_in / prod_ratios, cfg.vae_dim]);
        Ok(())
    }
}

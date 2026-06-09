//! ECAPA-TDNN speaker encoder for the Token2Wav DiT vocoder.
//!
//! Mirrors `ECAPA_TimeDelayNet` and its sub-blocks in the upstream
//! `modeling_qwen2_5_omni.py` (lines 2507–2806). The encoder turns a
//! reference mel spectrogram `(B, T, mel_dim=80)` into a fixed-width
//! speaker embedding `(B, enc_dim=128)` that conditions the DiT's
//! `DiTInputEmbedding`.
//!
//! ## Architecture (3B config — see `hybrie/docs/qwen2_5_omni-phases-2-6-spec.md` §"Phase 5 — ECAPA")
//!
//! ```text
//! blocks[0] : TDNN(80, 256, k=5, d=1)                      # initial conv
//! blocks[1] : SE-Res2Net(256, 256, scale=2, se=64, k=3, d=2)
//! blocks[2] : SE-Res2Net(256, 256, scale=2, se=64, k=3, d=3)
//! blocks[3] : SE-Res2Net(256, 256, scale=2, se=64, k=3, d=4)   # 3 SE-Res2Net, NOT 4
//! mfa : TDNN(768=256·3, 768, k=1, d=1)                     # over cat(blocks[1..], dim=1)
//! asp : AttentiveStatPool(768, attention_channels=64)      # → (B, 1536, 1)
//! fc  : Conv1d(1536, 128, k=1)                             # NOT Linear; weight is 3-D, no act
//! ```
//!
//! ## Critical gotchas (each silently breaks the port)
//!
//! - **`enc_dim=128` is the OUTPUT dim** (NOT `enc_emb_dim=192`, which is
//!   a separate precomputed speaker embed not produced here, and NOT
//!   `enc_lin_neurons=192`, declared-but-unused in this repo).
//! - **`fc` is `Conv1d(1536→128, k=1)`, not `Linear`** — its weight is
//!   3-D `[128, 1536, 1]`. Loading it as a `Linear` silently fails. **No
//!   activation after `fc`** — raw output, then `squeeze(-1)`.
//! - **Only 3 SE-Res2Net blocks**: the upstream loop is
//!   `range(1, len(enc_channels) - 1)` with `len == 5`.
//! - **MFA aggregates `cat(hidden_states[1:], dim=1)`** — it drops
//!   `blocks[0]`'s output (the initial TDNN), keeping the 3 SE-Res2Net
//!   outputs (3·256 = 768 channels).
//! - **Res2Net `scale=2`**: `torch.chunk(x, 2, dim=1)` → first half is a
//!   passthrough, second half goes through one inner TDNN, then concat.
//!   `range(scale-1) == 1` → exactly one inner conv
//!   (`res2net_block.blocks.0.conv`), which is what the released index
//!   carries.
//! - **Dilation / kernel size > 1 live ONLY on the res2net inner conv.**
//!   `tdnn1` / `tdnn2` are always `k=1, d=1`; `se_block` convs are `k=1`.
//!   The block's `(k, d)` flows solely into the `Res2NetBlock` inner TDNN.
//! - **ASP runs uniform stats FIRST** (mean+std over time with uniform
//!   `1/T` weights), then builds the attention, then weighted stats —
//!   *both* statistic passes happen. The attention chain is
//!   `tdnn(conv k=1 → ReLU) → tanh → conv → softmax(over T)` — both ReLU
//!   *and* tanh are present. The length mask upstream is all-ones
//!   (`lengths = ones · seq_length`), so its `masked_fill(-inf)` is a
//!   no-op; we skip the mask machinery entirely.
//! - **`padding="same"` + `padding_mode="reflect"`**: candle's `Conv1d`
//!   only zero-pads, so reflect padding is applied manually (narrow +
//!   flip + cat). For odd kernels (all ECAPA convs are odd) the "same"
//!   pad is symmetric: `p = (k-1)·d/2` each side. `k=1` convs need no pad.
//! - **`_compute_statistics` uses `clamp_min(eps)`** (single-arg upstream
//!   `.clamp(eps)`), i.e. a floor, NOT a symmetric clamp. Variance is the
//!   population variance (÷T via the weights). The attention softmax is
//!   computed in F32 then cast back (upstream-forced-fp32 numerics).

use candle::{DType, Module, Result, Tensor, D};
use candle_nn::{conv1d, Activation, Conv1d, Conv1dConfig, VarBuilder};

use crate::models::qwen2_5_omni::config::DitConfig;

/// `eps` floor for the std computed in attentive statistics pooling.
/// Matches upstream `AttentiveStatisticsPooling.eps = 1e-12`.
const ASP_EPS: f64 = 1e-12;

/// Reflect-pad a `(B, C, T)` tensor by `p` frames on each side of the
/// time axis (dim 2), matching PyTorch `nn.Conv1d(padding_mode="reflect")`
/// semantics (the border element is NOT duplicated):
///
/// `[a,b,c,d,e]` with `p=2` → `[c,b,a,b,c,d,e,d,c]`.
///
/// `p == 0` returns the input unchanged (the `k=1` fast path that every
/// pointwise conv hits). Requires `p < T`.
fn reflect_pad1d(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(x.clone());
    }
    let t = x.dim(2)?;
    if p >= t {
        candle::bail!("reflect_pad1d: pad {p} must be < time length {t}");
    }
    // Left: reflect frames [1..=p], reversed → [p, p-1, ..., 1].
    // `flip` uses index-select internally, which requires a contiguous
    // input, so materialise the narrowed view first.
    let left = x.narrow(2, 1, p)?.contiguous()?.flip(&[2])?;
    // Right: reflect frames [T-1-p .. T-1], reversed.
    let right = x.narrow(2, t - 1 - p, p)?.contiguous()?.flip(&[2])?;
    Tensor::cat(&[&left, x, &right], 2)
}

/// `TimeDelayNetBlock` — reflect-padded dilated `Conv1d` (`padding="same"`,
/// `padding_mode="reflect"`) followed by ReLU. Upstream lines 2507–2527.
#[derive(Debug, Clone)]
struct TimeDelayNetBlock {
    conv: Conv1d,
    pad: usize,
}

impl TimeDelayNetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // "same" reflect pad for an odd kernel is symmetric:
        // p = (k-1)·d / 2 each side.
        let pad = (kernel_size - 1) * dilation / 2;
        let cfg = Conv1dConfig {
            padding: 0, // reflect padding is applied manually
            stride: 1,
            dilation,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(in_channels, out_channels, kernel_size, cfg, vb.pp("conv"))?;
        Ok(Self { conv, pad })
    }

    /// `x`: `(B, C_in, T)` → `(B, C_out, T)` (length preserved by "same").
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = reflect_pad1d(x, self.pad)?;
        let x = self.conv.forward(&x)?;
        x.apply(&Activation::Relu)
    }
}

/// `Res2NetBlock` with `scale` chunks. Upstream lines 2530–2561. For
/// `scale=2`: split channels in two; the first half is a passthrough, the
/// second half goes through one inner `TimeDelayNetBlock`; concat. The
/// general accumulation (`output_part = blocks[i-1](hidden_part +
/// output_part)` for `i >= 2`) is implemented faithfully so the same code
/// holds at larger scales.
#[derive(Debug, Clone)]
struct Res2NetBlock {
    blocks: Vec<TimeDelayNetBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;
        let vb_blocks = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..(scale - 1) {
            blocks.push(TimeDelayNetBlock::new(
                in_channel,
                hidden_channel,
                kernel_size,
                dilation,
                vb_blocks.pp(i),
            )?);
        }
        Ok(Self { blocks, scale })
    }

    /// `x`: `(B, C, T)` → `(B, C, T)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = x.dim(1)?;
        let part = c / self.scale;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(self.scale);
        let mut prev: Option<Tensor> = None;
        for i in 0..self.scale {
            let hidden_part = x.narrow(1, i * part, part)?;
            let output_part = if i == 0 {
                hidden_part
            } else if i == 1 {
                self.blocks[i - 1].forward(&hidden_part)?
            } else {
                let summed = (&hidden_part + prev.as_ref().unwrap())?;
                self.blocks[i - 1].forward(&summed)?
            };
            prev = Some(output_part.clone());
            outputs.push(output_part);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 1)
    }
}

/// `SqueezeExcitationBlock` — global-avg-pool over time → `Conv1d(C→se,
/// k=1)` → ReLU → `Conv1d(se→C, k=1)` → sigmoid → per-channel rescale.
/// Upstream lines 2564–2591. The two convs are `k=1` so no reflect pad is
/// needed.
#[derive(Debug, Clone)]
struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn new(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv1 = conv1d(in_channels, se_channels, 1, cfg, vb.pp("conv1"))?;
        let conv2 = conv1d(se_channels, out_channels, 1, cfg, vb.pp("conv2"))?;
        Ok(Self { conv1, conv2 })
    }

    /// `x`: `(B, C, T)` → `(B, C, T)`, channels rescaled by a per-channel
    /// gate computed from the time-mean.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // mean over time, keepdim → (B, C, 1).
        let mean = x.mean_keepdim(2)?;
        let mean = self.conv1.forward(&mean)?.apply(&Activation::Relu)?;
        let gate = candle_nn::ops::sigmoid(&self.conv2.forward(&mean)?)?; // (B, C, 1)
        x.broadcast_mul(&gate)
    }
}

/// `SqueezeExcitationRes2NetBlock` — TDNN → Res2Net → TDNN → SE, with a
/// residual add. Upstream lines 2683–2722. `tdnn1` and `tdnn2` are always
/// `k=1, d=1`; the block's `(kernel_size, dilation)` flow ONLY into the
/// inner `Res2NetBlock` conv.
#[derive(Debug, Clone)]
struct SqueezeExcitationRes2NetBlock {
    tdnn1: TimeDelayNetBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TimeDelayNetBlock,
    se_block: SqueezeExcitationBlock,
}

impl SqueezeExcitationRes2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // tdnn1 / tdnn2 are pointwise (k=1, d=1) — see gotcha.
        let tdnn1 = TimeDelayNetBlock::new(in_channels, out_channels, 1, 1, vb.pp("tdnn1"))?;
        let res2net_block = Res2NetBlock::new(
            out_channels,
            out_channels,
            res2net_scale,
            kernel_size,
            dilation,
            vb.pp("res2net_block"),
        )?;
        let tdnn2 = TimeDelayNetBlock::new(out_channels, out_channels, 1, 1, vb.pp("tdnn2"))?;
        let se_block =
            SqueezeExcitationBlock::new(out_channels, se_channels, out_channels, vb.pp("se_block"))?;
        Ok(Self {
            tdnn1,
            res2net_block,
            tdnn2,
            se_block,
        })
    }

    /// `x`: `(B, C, T)` → `(B, C, T)`, residual-added.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let h = self.tdnn1.forward(x)?;
        let h = self.res2net_block.forward(&h)?;
        let h = self.tdnn2.forward(&h)?;
        let h = self.se_block.forward(&h)?;
        residual + &h
    }
}

/// `AttentiveStatisticsPooling` — upstream lines 2594–2680. Returns the
/// concatenated attention-weighted mean + std over time, as `(B, 2C, 1)`.
#[derive(Debug, Clone)]
struct AttentiveStatisticsPooling {
    tdnn: TimeDelayNetBlock, // TimeDelayNetBlock(channels*3, attention_channels, 1, 1)
    conv: Conv1d,            // Conv1d(attention_channels, channels, k=1)
}

impl AttentiveStatisticsPooling {
    fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        let tdnn = TimeDelayNetBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?;
        let cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let conv = conv1d(attention_channels, channels, 1, cfg, vb.pp("conv"))?;
        Ok(Self { tdnn, conv })
    }

    /// Weighted statistics over the time axis (dim 2):
    /// `mean = Σ_t w·x`, `std = sqrt(clamp_min(Σ_t w·(x-mean)², eps))`.
    /// `weights` broadcasts against `x`: `(B,1,T)` (uniform) or `(B,C,T)`
    /// (attention). Returns `(mean, std)` each `(B, C)`.
    fn compute_statistics(x: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        // mean over time → (B, C).
        let mean = x.broadcast_mul(weights)?.sum(2)?;
        // (x - mean)² weighted, summed over time → variance (population).
        let centered = x.broadcast_sub(&mean.unsqueeze(2)?)?;
        let var = centered.sqr()?.broadcast_mul(weights)?.sum(2)?;
        let std = var.maximum(ASP_EPS)?.sqrt()?;
        Ok((mean, std))
    }

    /// `x`: `(B, C, T)` → `(B, 2C, 1)`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t) = x.dims3()?;
        let dtype = x.dtype();

        // Uniform-weight statistics first: each frame weighted 1/T. The
        // upstream length mask is all-ones, so `mask / total` is the
        // uniform `1/T`; the later `masked_fill(-inf)` is a no-op.
        let uniform = Tensor::full(1.0f64 / t as f64, (b, 1, t), x.device())?.to_dtype(dtype)?;
        let (mean, std) = Self::compute_statistics(x, &uniform)?;
        // Broadcast mean/std back over time → (B, C, T).
        let mean_rep = mean.unsqueeze(2)?.broadcast_as((b, c, t))?;
        let std_rep = std.unsqueeze(2)?.broadcast_as((b, c, t))?;

        // attention input: cat([x, mean, std], dim=1) → (B, 3C, T).
        let attn_in = Tensor::cat(&[x, &mean_rep, &std_rep], 1)?;

        // tdnn (conv k=1 → ReLU) → tanh → conv → softmax over T.
        let attn = self.tdnn.forward(&attn_in)?; // includes ReLU
        let attn = attn.tanh()?;
        let attn = self.conv.forward(&attn)?; // (B, C, T)

        // Softmax over the time axis (last dim), in F32 then cast back —
        // mirrors upstream fp32-forced softmax numerics.
        let attn = candle_nn::ops::softmax(&attn.to_dtype(DType::F32)?, D::Minus1)?
            .to_dtype(dtype)?; // (B, C, T)

        // Attention-weighted statistics.
        let (mean, std) = Self::compute_statistics(x, &attn)?;
        // cat((mean, std), dim=1) → (B, 2C), then unsqueeze(2) → (B, 2C, 1).
        let pooled = Tensor::cat(&[&mean, &std], 1)?;
        pooled.unsqueeze(2)
    }
}

/// ECAPA-TDNN speaker encoder. Construct with a VarBuilder pre-scoped to
/// the released checkpoint's
/// `token2wav.code2wav_dit_model.input_embed.spk_encoder.*` slice.
#[derive(Debug, Clone)]
pub struct EcapaTdnn {
    blocks: Vec<Block>,
    mfa: TimeDelayNetBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
    enc_dim: usize,
}

/// The `blocks` ModuleList is heterogeneous: index 0 is the initial TDNN
/// conv, indices 1.. are SE-Res2Net blocks. The SE-Res2Net variant is
/// boxed so the enum stays small (avoids `clippy::large_enum_variant`).
#[derive(Debug, Clone)]
enum Block {
    Tdnn(TimeDelayNetBlock),
    SeRes2Net(Box<SqueezeExcitationRes2NetBlock>),
}

impl Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::Tdnn(b) => b.forward(x),
            Block::SeRes2Net(b) => b.forward(x),
        }
    }
}

impl EcapaTdnn {
    /// Build from the DiT config + a VarBuilder scoped to `spk_encoder`.
    pub fn new(cfg: &DitConfig, vb: VarBuilder) -> Result<Self> {
        let channels = &cfg.enc_channels;
        let kernels = &cfg.enc_kernel_sizes;
        let dilations = &cfg.enc_dilations;
        if channels.len() != kernels.len() || channels.len() != dilations.len() {
            candle::bail!(
                "enc_channels ({}), enc_kernel_sizes ({}) and enc_dilations ({}) must share length",
                channels.len(),
                kernels.len(),
                dilations.len()
            );
        }
        let n = channels.len();
        if n < 2 {
            candle::bail!("ECAPA requires at least 2 enc_channels entries, got {n}");
        }

        let vb_blocks = vb.pp("blocks");
        let mut blocks: Vec<Block> = Vec::with_capacity(n - 1);

        // blocks[0]: initial TDNN(mel_dim → enc_channels[0], k[0], d[0]).
        blocks.push(Block::Tdnn(TimeDelayNetBlock::new(
            cfg.mel_dim,
            channels[0],
            kernels[0],
            dilations[0],
            vb_blocks.pp(0),
        )?));

        // blocks[1..len-1]: SE-Res2Net layers. Upstream uses
        // `enc_channels[i-1]` as in_channels (always == enc_channels[i]
        // here since all hidden channels are equal at 256).
        for i in 1..(n - 1) {
            blocks.push(Block::SeRes2Net(Box::new(
                SqueezeExcitationRes2NetBlock::new(
                    channels[i - 1],
                    channels[i],
                    cfg.enc_res2net_scale,
                    cfg.enc_se_channels,
                    kernels[i],
                    dilations[i],
                    vb_blocks.pp(i),
                )?,
            )));
        }

        // MFA: TDNN(enc_channels[-1], enc_channels[-1], k[-1], d[-1]). Its
        // input at forward time is cat(blocks[1..]) which equals
        // enc_channels[-1] (= 3·256 = 768).
        let mfa = TimeDelayNetBlock::new(
            channels[n - 1],
            channels[n - 1],
            kernels[n - 1],
            dilations[n - 1],
            vb.pp("mfa"),
        )?;

        // ASP over enc_channels[-1] channels → (B, 2·enc_channels[-1], 1).
        let asp =
            AttentiveStatisticsPooling::new(channels[n - 1], cfg.enc_attention_channels, vb.pp("asp"))?;

        // Final fc is a Conv1d(enc_channels[-1]·2 → enc_dim, k=1) — NOT a
        // Linear; weight is 3-D `[enc_dim, enc_channels[-1]·2, 1]`. No
        // activation follows.
        let fc_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let fc = conv1d(channels[n - 1] * 2, cfg.enc_dim, 1, fc_cfg, vb.pp("fc"))?;

        Ok(Self {
            blocks,
            mfa,
            asp,
            fc,
            enc_dim: cfg.enc_dim,
        })
    }

    /// Output speaker-embedding dim (`enc_dim` = 128 for the 3B model).
    pub fn output_dim(&self) -> usize {
        self.enc_dim
    }

    /// Speaker-embedding forward.
    ///
    /// Input `mel`: `(B, T, mel_dim=80)` — a reference utterance's mel
    /// features. Output: `(B, enc_dim=128)`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // (B, T, mel_dim) → (B, mel_dim, T). Upstream line 2788.
        let mut h = mel.transpose(1, 2)?.contiguous()?;

        // Run all blocks, collecting per-block outputs.
        let mut hidden_states: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            h = block.forward(&h)?;
            hidden_states.push(h.clone());
        }

        // Multi-layer feature aggregation: cat(hidden_states[1:], dim=1)
        // — drops blocks[0] (the initial TDNN).
        let mfa_in_refs: Vec<&Tensor> = hidden_states[1..].iter().collect();
        let mfa_in = Tensor::cat(&mfa_in_refs, 1)?;
        let h = self.mfa.forward(&mfa_in)?;

        // Attentive statistical pooling → (B, 2C, 1).
        let h = self.asp.forward(&h)?;

        // Final Conv1d → (B, enc_dim, 1), then squeeze(-1) → (B, enc_dim).
        let h = self.fc.forward(&h)?;
        h.squeeze(D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::qwen2_5_omni::config::DitConfig;
    use candle::Device;
    use candle_nn::{VarBuilder, VarMap};

    /// A trivially-small DiT config used by the CPU shape + key-tree
    /// tests. Constraints baked in (else shapes drift):
    /// - `enc_channels` length 5: `[c0, c1, c2, c3, c_last]`.
    /// - `c_last == 3 · c1` (MFA in-channels == cat of 3 SE-Res2Net outs).
    /// - all channels even (Res2Net `scale=2` chunk).
    /// - `mfa`/`asp` are built with `enc_channels[-1]` as in-channels.
    fn tiny_cfg() -> DitConfig {
        DitConfig {
            hidden_size: 32,
            num_hidden_layers: 1,
            heads: 2,
            head_dim: 16,
            ff_mult: 2,
            mel_dim: 12,
            emb_dim: 8,
            num_embeds: 17,
            repeats: 2,
            enc_dim: 24, // OUTPUT dim (mirrors the real enc_dim=128)
            enc_emb_dim: 16,
            enc_channels: vec![16, 16, 16, 16, 48], // c_last = 48 = 3·16
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 2,
            enc_se_channels: 8,
            enc_lin_neurons: 16,
            dropout: 0.0,
            block_size: 24,
            look_ahead_layers: vec![10],
            look_backward_layers: vec![0, 20],
        }
    }

    fn randomize(vm: &VarMap, device: &Device) -> Result<()> {
        for v in vm.all_vars() {
            v.set(&Tensor::randn(0f32, 0.02, v.as_tensor().shape().clone(), device)?)?;
        }
        Ok(())
    }

    /// Reflect-pad helper matches PyTorch `padding_mode="reflect"`
    /// element-wise: `[0,1,2,3,4]` pad 2 → `[2,1,0,1,2,3,4,3,2]`, and
    /// `p=0` is the identity.
    #[test]
    fn reflect_pad_matches_pytorch() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![0f32, 1., 2., 3., 4.], (1, 1, 5), &device)?;
        let padded = reflect_pad1d(&x, 2)?;
        assert_eq!(padded.dims(), &[1, 1, 9]);
        let got: Vec<f32> = padded.flatten_all()?.to_vec1()?;
        assert_eq!(got, vec![2., 1., 0., 1., 2., 3., 4., 3., 2.]);

        // p=0 is identity.
        let same = reflect_pad1d(&x, 0)?;
        assert_eq!(same.flatten_all()?.to_vec1::<f32>()?, vec![0., 1., 2., 3., 4.]);

        // p=1: [0,1,2,3,4] → [1,0,1,2,3,4,3].
        let p1 = reflect_pad1d(&x, 1)?;
        assert_eq!(p1.flatten_all()?.to_vec1::<f32>()?, vec![1., 0., 1., 2., 3., 4., 3.]);
        Ok(())
    }

    /// **Phase 5 ECAPA acceptance #1** — forward on a tiny random config
    /// turns `(B, T, mel_dim)` into `(B, enc_dim)` and the output is
    /// finite. `enc_dim` (NOT enc_emb_dim) is the output width.
    #[test]
    fn ecapa_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let enc = EcapaTdnn::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        assert_eq!(enc.output_dim(), cfg.enc_dim);

        // (B=2, T=40, mel_dim).
        let mel = Tensor::randn(0f32, 1f32, (2, 40, cfg.mel_dim), &device)?;
        let out = enc.forward(&mel)?;
        assert_eq!(out.dims(), &[2, cfg.enc_dim]);
        assert!(out
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()));

        // A single-batch utterance of a different length also works.
        let mel = Tensor::randn(0f32, 1f32, (1, 23, cfg.mel_dim), &device)?;
        let out = enc.forward(&mel)?;
        assert_eq!(out.dims(), &[1, cfg.enc_dim]);
        Ok(())
    }

    /// **Phase 5 ECAPA acceptance #2** — the VarBuilder key tree exactly
    /// matches the released
    /// `token2wav.code2wav_dit_model.input_embed.spk_encoder.*` layout
    /// (40 tensors per the released `index_3b.json` — the spec's "42" is
    /// stale; the index is ground truth). No extra, no missing keys. This
    /// is the test that catches: `fc` accidentally built as a 2-D Linear,
    /// a 4th SE-Res2Net block, dilation leaking onto tdnn1/tdnn2 (would
    /// not change keys but would change shapes elsewhere), and the
    /// scale=2 → single `res2net_block.blocks.0` invariant.
    #[test]
    fn ecapa_safetensors_keys_match_upstream() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_cfg();
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let _enc = EcapaTdnn::new(&cfg, vb)?;
        randomize(&vm, &device)?;

        let names: std::collections::BTreeSet<String> =
            vm.data().lock().unwrap().keys().cloned().collect();

        let mut want: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        // blocks.0 — initial TDNN conv.
        want.insert("blocks.0.conv.weight".to_string());
        want.insert("blocks.0.conv.bias".to_string());
        // blocks.{1,2,3} — SE-Res2Net (scale=2 → one inner res2net conv).
        for i in 1..4 {
            for sub in [
                "tdnn1.conv",
                "res2net_block.blocks.0.conv",
                "tdnn2.conv",
                "se_block.conv1",
                "se_block.conv2",
            ] {
                want.insert(format!("blocks.{i}.{sub}.weight"));
                want.insert(format!("blocks.{i}.{sub}.bias"));
            }
        }
        // mfa, asp, fc.
        for key in [
            "mfa.conv.weight",
            "mfa.conv.bias",
            "asp.tdnn.conv.weight",
            "asp.tdnn.conv.bias",
            "asp.conv.weight",
            "asp.conv.bias",
            "fc.weight",
            "fc.bias",
        ] {
            want.insert(key.to_string());
        }

        // Sanity: the released index carries exactly 40 spk_encoder keys.
        assert_eq!(want.len(), 40, "expected 40 ECAPA keys, modelled {}", want.len());

        let extra: Vec<_> = names.difference(&want).collect();
        let missing: Vec<_> = want.difference(&names).collect();
        assert!(
            extra.is_empty() && missing.is_empty(),
            "key tree drift — extra: {extra:?}\nmissing: {missing:?}"
        );

        // fc is a Conv1d: its weight must be 3-D `[enc_dim, 2C, 1]`, NOT
        // a 2-D Linear weight. Asserting the rank locks gotcha #1.
        let fc_w = vm.data().lock().unwrap().get("fc.weight").unwrap().clone();
        assert_eq!(fc_w.dims().len(), 3, "fc.weight must be a 3-D Conv1d kernel");
        assert_eq!(
            fc_w.dims(),
            &[cfg.enc_dim, cfg.enc_channels[4] * 2, 1],
            "fc.weight shape mismatch"
        );
        Ok(())
    }

    /// **Phase 5 ECAPA acceptance #3** — real-weight smoke against
    /// `Qwen/Qwen2.5-Omni-3B`. CPU/F32 (candle CPU has no bf16 matmul
    /// kernel). `#[ignore]` keeps it out of the default matrix; invoked
    /// from the Lambda smoke runner. `(1, 200, 80)` ref mel → `(1, 128)`.
    #[test]
    #[ignore]
    fn real_weight_ecapa_loads() {
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
        // Released weights are bf16 on disk; cast to F32 on read so the
        // smoke runs on CPU (candle CPU matmul has no bf16 kernel — the
        // GPU bf16 variant lives in the cuda test below).
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, DType::F32, &device)
                .expect("mmap safetensors")
        };
        let enc = EcapaTdnn::new(
            &cfg.token2wav_config.dit_config,
            vb.pp("token2wav")
                .pp("code2wav_dit_model")
                .pp("input_embed")
                .pp("spk_encoder"),
        )
        .expect("construct EcapaTdnn from real weights");

        // (1, 200, 80) reference mel → (1, 128) speaker embedding.
        let mel = Tensor::randn(
            0f32,
            1f32,
            (1usize, 200usize, cfg.token2wav_config.dit_config.mel_dim),
            &device,
        )
        .unwrap();
        let out = enc.forward(&mel).expect("forward on real weights");
        eprintln!("real_weight_ecapa_loads: output shape = {:?}", out.dims());
        assert_eq!(out.dims(), &[1, enc.output_dim()]);
        assert_eq!(enc.output_dim(), 128);
        assert!(out
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
    }

    /// **Phase 5 ECAPA GPU acceptance** — same smoke at native BF16 on
    /// CUDA. Lambda + `--features cuda` only.
    #[test]
    #[ignore]
    #[cfg(feature = "cuda")]
    fn real_weight_ecapa_loads_cuda_bf16() {
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
        let enc = EcapaTdnn::new(
            &cfg.token2wav_config.dit_config,
            vb.pp("token2wav")
                .pp("code2wav_dit_model")
                .pp("input_embed")
                .pp("spk_encoder"),
        )
        .expect("construct EcapaTdnn from real weights");

        let mel = Tensor::randn(
            0f32,
            1f32,
            (1usize, 200usize, cfg.token2wav_config.dit_config.mel_dim),
            &device,
        )
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let out = enc.forward(&mel).expect("forward on real weights (cuda bf16)");
        eprintln!(
            "real_weight_ecapa_loads_cuda_bf16: output shape = {:?}",
            out.dims()
        );
        assert_eq!(out.dims(), &[1, enc.output_dim()]);
        assert_eq!(enc.output_dim(), 128);
        assert_eq!(out.dtype(), DType::BF16);
    }
}

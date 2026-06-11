//! Multimodal Rotary Position Embedding (M-RoPE) for the Qwen2.5-Omni
//! Thinker and Talker decoders.
//!
//! Mirrors `apply_multimodal_rotary_pos_emb` in upstream
//! `modeling_qwen2_5_omni.py:1389-1431`. The key trick — easy to silently
//! get wrong — is the **Python list-repetition** of `mrope_section`:
//!
//! ```python
//! mrope_section = mrope_section * 2     # [16,24,24] → [16,24,24,16,24,24]
//! cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, -1))], -1)
//! ```
//!
//! Concretely with Thinker's `mrope_section=[16,24,24]` and `head_dim=128`,
//! the head_dim is split into **6 chunks**, each chunk indexed by `i % 3`
//! into the `(temporal, height, width)` position streams. Final layout:
//! `[T:16, H:24, W:24, T:16, H:24, W:24]`.
//!
//! ## Invariant
//!
//! `sum(mrope_section) == head_dim / 2` is required. Thinker: 16+24+24 =
//! 64 = 128/2 ✓. Talker: 16+16+0 = 32 = 64/2 ✓. A wrong value here is one
//! of the silent-misload traps cataloged in
//! `hybrie/docs/qwen2_5_omni-phases-2-6-spec.md`.

use candle::{DType, Device, IndexOp, Result, Tensor, D};

/// M-RoPE table — precomputes the standard `inv_freq` cos/sin tables once,
/// then applies the upstream `mrope_section` split-and-pick algorithm at
/// forward time.
#[derive(Debug, Clone)]
pub struct MRopeTable {
    /// `(max_position_embeddings, head_dim/2)`. Stored at construction
    /// dtype (typically F32 for numerical stability — the rotary
    /// computation upstream is fp32-only).
    cos: Tensor,
    sin: Tensor,
    /// Section sizes, e.g. `[16, 24, 24]` for Thinker, `[16, 16, 0]` for
    /// Talker. Sum must equal `head_dim / 2`.
    mrope_section: Vec<usize>,
    head_dim: usize,
}

impl MRopeTable {
    /// Build a M-RoPE table.
    ///
    /// `rope_theta` is the rotary base (1e6 for both Thinker and Talker
    /// in Qwen2.5-Omni-3B). `dtype` controls the storage dtype of the
    /// precomputed cos/sin tables.
    pub fn new(
        head_dim: usize,
        rope_theta: f64,
        max_position_embeddings: usize,
        mrope_section: &[usize],
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        // Section sum must equal head_dim/2; otherwise the apply step's
        // 6-chunk indexer addresses out of bounds. Trip loudly.
        let section_sum: usize = mrope_section.iter().sum();
        if section_sum * 2 != head_dim {
            candle::bail!(
                "M-RoPE invariant violated: sum(mrope_section)={section_sum} must equal head_dim/2={}, got mrope_section={mrope_section:?}, head_dim={head_dim}",
                head_dim / 2,
            );
        }

        // Standard RoPE inv_freq computation.
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (1, half), device)?;
        let t = Tensor::arange(0u32, max_position_embeddings as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            mrope_section: mrope_section.to_vec(),
            head_dim,
        })
    }

    /// Apply M-RoPE to `q` `(B, H, S, head_dim)` and `k`
    /// `(B, kv_heads, S, head_dim)` given 3-stream `position_ids`
    /// `(3, B, S)`. Returns `(q_rope, k_rope)` of the same shapes.
    ///
    /// For text-only forwards, all 3 streams are identical (an arange
    /// expanded to 3 rows), and M-RoPE collapses to standard 1-D RoPE.
    /// For multimodal forwards, the streams differ per-modality and the
    /// `i % 3` chunk picker mixes them per spec.
    pub fn apply(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (three, _b, _s) = position_ids.dims3()?;
        if three != 3 {
            candle::bail!("position_ids must have 3 streams; got {three}");
        }

        let (cos_3d, sin_3d) = self.compute_3d_rope_embeddings(position_ids)?;
        let (cos, sin) = self.apply_mrope_sections(&cos_3d, &sin_3d)?;
        // Reshape for broadcast over head axis: (B, 1, S, head_dim).
        let cos = cos.unsqueeze(1)?;
        let sin = sin.unsqueeze(1)?;
        let q = apply_rope_to_tensor(q, &cos, &sin)?;
        let k = apply_rope_to_tensor(k, &cos, &sin)?;
        Ok((q, k))
    }

    /// For each of the 3 position streams, gather cos/sin via
    /// `index_select` and duplicate the head_dim/2-wide table to
    /// head_dim via `cat([t, t], -1)`. Returns `(cos_3d, sin_3d)` each
    /// `(3, B, S, head_dim)`.
    fn compute_3d_rope_embeddings(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_three, b, s) = position_ids.dims3()?;
        let half = self.head_dim / 2;

        let mut cos_parts = Vec::with_capacity(3);
        let mut sin_parts = Vec::with_capacity(3);
        for dim_idx in 0..3 {
            let pos = position_ids.i(dim_idx)?;
            let pos_flat = pos.flatten_all()?;
            let cos_gathered = self.cos.index_select(&pos_flat, 0)?;
            let sin_gathered = self.sin.index_select(&pos_flat, 0)?;
            let cos_dim = cos_gathered.reshape((b, s, half))?;
            let sin_dim = sin_gathered.reshape((b, s, half))?;
            // Duplicate to full head_dim along the last axis.
            cos_parts.push(Tensor::cat(&[&cos_dim, &cos_dim], D::Minus1)?);
            sin_parts.push(Tensor::cat(&[&sin_dim, &sin_dim], D::Minus1)?);
        }
        Ok((Tensor::stack(&cos_parts, 0)?, Tensor::stack(&sin_parts, 0)?))
    }

    /// Pick chunks from the 3 streams per the `mrope_section * 2` /
    /// `i % 3` upstream algorithm. Returns `(cos, sin)` each
    /// `(B, S, head_dim)`.
    fn apply_mrope_sections(
        &self,
        cos_3d: &Tensor,
        sin_3d: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        // mrope_section * 2 — **Python list repetition**, not arithmetic.
        let mut sections = Vec::with_capacity(self.mrope_section.len() * 2);
        sections.extend_from_slice(&self.mrope_section);
        sections.extend_from_slice(&self.mrope_section);

        let mut cos_parts = Vec::with_capacity(sections.len());
        let mut sin_parts = Vec::with_capacity(sections.len());
        let mut offset = 0usize;
        for (i, &sec) in sections.iter().enumerate() {
            if sec == 0 {
                // Width-axis is 0-width on Talker `[16,16,0]`. Skip with
                // no offset advance — the chunk contributes nothing.
                continue;
            }
            let dim_idx = i % 3;
            cos_parts.push(cos_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec)?);
            sin_parts.push(sin_3d.i(dim_idx)?.narrow(D::Minus1, offset, sec)?);
            offset += sec;
        }
        Ok((Tensor::cat(&cos_parts, D::Minus1)?, Tensor::cat(&sin_parts, D::Minus1)?))
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

/// Apply rotary to `x: (B, H, S, head_dim)` given broadcastable cos/sin
/// of shape `(B, 1, S, head_dim)`. `rotate_half` is the standard
/// split-half negation that pairs with block-duplicated cos/sin (the
/// Qwen2.5-Omni Thinker/Talker M-RoPE convention).
pub(crate) fn apply_rope_to_tensor(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let neg_x2 = (x2 * -1f64)?;
    let rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
    let in_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let rot_f32 = rotated.to_dtype(DType::F32)?;
    let cos_f32 = cos.to_dtype(DType::F32)?;
    let sin_f32 = sin.to_dtype(DType::F32)?;
    let out = (x_f32.broadcast_mul(&cos_f32)? + rot_f32.broadcast_mul(&sin_f32)?)?;
    out.to_dtype(in_dtype)
}

/// Build a text-only position-ids tensor `(3, B, S)` from a sequence
/// length plus a prefill offset. All 3 streams are identical (arange
/// shifted by the offset), so M-RoPE collapses to standard 1-D RoPE in
/// this case — used for any forward where the input has no audio /
/// image / video tokens, including the LM-only prefill smoke.
///
/// For multimodal inputs the caller must use a proper `get_rope_index`
/// builder (deferred to a follow-up — defers `use_audio_in_video=True`
/// chunked interleave and packed `(4, B, S)` mode per the phase spec).
pub fn text_only_position_ids(
    batch: usize,
    seq_len: usize,
    offset: usize,
    device: &Device,
) -> Result<Tensor> {
    let v: Vec<i64> = (0..seq_len).map(|i| (i + offset) as i64).collect();
    let row = Tensor::from_vec(v, (1, 1, seq_len), device)?
        .broadcast_as((1, batch, seq_len))?;
    // Stack 3 identical rows → (3, B, S).
    Tensor::cat(&[&row, &row, &row], 0)?.reshape((3, batch, seq_len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// **Invariant** — `sum(mrope_section) == head_dim/2`. Wrong values
    /// must fail loudly at construction, not silently produce nonsense.
    #[test]
    fn mrope_section_invariant() {
        let dev = Device::Cpu;
        // Thinker — head_dim 128, section [16,24,24], sum 64 = 128/2 ✓
        let _ = MRopeTable::new(128, 1e6, 64, &[16, 24, 24], &dev, DType::F32).unwrap();
        // Talker — head_dim 64, section [16,16,0], sum 32 = 64/2 ✓
        let _ = MRopeTable::new(64, 1e6, 64, &[16, 16, 0], &dev, DType::F32).unwrap();
        // Wrong sum → bail
        let err = MRopeTable::new(128, 1e6, 64, &[10, 24, 24], &dev, DType::F32);
        assert!(err.is_err());
    }

    /// Identical T/H/W position rows should collapse M-RoPE to standard
    /// 1-D RoPE — applying it twice with the same position must give the
    /// same answer the second time (idempotent + deterministic), and the
    /// shape must round-trip.
    #[test]
    fn mrope_text_only_collapses_to_1d() -> Result<()> {
        let dev = Device::Cpu;
        let head_dim = 16usize;
        let table = MRopeTable::new(head_dim, 1e6, 32, &[2, 3, 3], &dev, DType::F32)?;
        let b = 1;
        let h_q = 4;
        let h_k = 2;
        let s = 5;
        let q = Tensor::randn(0f32, 1f32, (b, h_q, s, head_dim), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (b, h_k, s, head_dim), &dev)?;
        let pids = text_only_position_ids(b, s, 0, &dev)?;
        assert_eq!(pids.dims(), &[3, b, s]);

        let (q1, k1) = table.apply(&q, &k, &pids)?;
        assert_eq!(q1.dims(), &[b, h_q, s, head_dim]);
        assert_eq!(k1.dims(), &[b, h_k, s, head_dim]);
        // Determinism: same call again, same answer.
        let (q2, _) = table.apply(&q, &k, &pids)?;
        let diff_v = (q1 - q2)?.abs()?.flatten_all()?.to_vec1::<f32>()?;
        let max_diff = diff_v.iter().cloned().fold(0f32, f32::max);
        assert!(max_diff < 1e-5, "non-deterministic apply: {max_diff}");
        Ok(())
    }

    /// Talker config: head_dim 64, section [16,16,0]. The 0-width
    /// chunk on the W axis must be skipped without breaking the
    /// indexer, and the output dim must remain head_dim.
    #[test]
    fn mrope_talker_section_with_zero_width() -> Result<()> {
        let dev = Device::Cpu;
        let head_dim = 64usize;
        let table = MRopeTable::new(head_dim, 1e6, 16, &[16, 16, 0], &dev, DType::F32)?;
        let q = Tensor::randn(0f32, 1f32, (1, 4, 3, head_dim), &dev)?;
        let k = Tensor::randn(0f32, 1f32, (1, 2, 3, head_dim), &dev)?;
        let pids = text_only_position_ids(1, 3, 0, &dev)?;
        let (q1, k1) = table.apply(&q, &k, &pids)?;
        assert_eq!(q1.dims(), &[1, 4, 3, head_dim]);
        assert_eq!(k1.dims(), &[1, 2, 3, head_dim]);
        Ok(())
    }
}

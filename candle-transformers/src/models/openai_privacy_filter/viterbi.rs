//! Constrained Viterbi decoder for BIOES label sequences.
//!
//! Given per-token logits over `1 + 4*N_classes` labels (1 outside + B/I/E/S
//! per class), returns the highest-scoring label sequence that respects
//! BIOES transition rules: `B-X` must be followed by `I-X` or `E-X`,
//! `S-X` and `E-X` end the span, `O` cannot be followed by `I-X`/`E-X`.
//!
//! Label encoding contract (matches `OpenMed/privacy-filter-nemotron`):
//! - Index 0 is `O`
//! - Indices `[1 + 4*c]` is `B-class[c]`, `+1` is `I`, `+2` is `E`, `+3` is `S`
//! - Total = `1 + 4 * num_classes`
//!
//! Returned spans are tuples `(start_token, end_token_exclusive, class_idx)`.

use candle::{Result, Tensor, D};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tag {
    O,
    B,
    I,
    E,
    S,
}

#[derive(Debug, Clone, Copy)]
struct Decoded {
    tag: Tag,
    class_idx: Option<usize>, // None for O
}

/// Decode BIOES spans from logits of shape `(seq_len, num_labels)`.
/// Returns `(start, end, class_idx)` triplets — `end` is exclusive.
pub fn decode(logits: &Tensor, num_classes: usize) -> Result<Vec<(usize, usize, usize)>> {
    let (seq_len, num_labels) = logits.dims2()?;
    debug_assert_eq!(num_labels, 1 + 4 * num_classes);

    // Gather logits as f32 for the DP. seq_len × num_labels rows.
    let scores = logits.to_dtype(candle::DType::F32)?.to_vec2::<f32>()?;

    // dp[t][label] = best path score ending at (t, label).
    // bp[t][label] = the prev label that produced it.
    let mut dp = vec![vec![f32::NEG_INFINITY; num_labels]; seq_len];
    let mut bp = vec![vec![0usize; num_labels]; seq_len];

    // t = 0: only O, B-X, S-X are valid starts.
    for label in 0..num_labels {
        let dec = decode_label(label, num_classes);
        if matches!(dec.tag, Tag::O | Tag::B | Tag::S) {
            dp[0][label] = scores[0][label];
        }
    }

    // DP forward.
    for t in 1..seq_len {
        for label in 0..num_labels {
            let cur = decode_label(label, num_classes);
            for prev_label in 0..num_labels {
                if dp[t - 1][prev_label] == f32::NEG_INFINITY {
                    continue;
                }
                let prev = decode_label(prev_label, num_classes);
                if !is_valid_transition(prev, cur) {
                    continue;
                }
                let s = dp[t - 1][prev_label] + scores[t][label];
                if s > dp[t][label] {
                    dp[t][label] = s;
                    bp[t][label] = prev_label;
                }
            }
        }
    }

    // Last token must end a span: O, E-X, S-X.
    let mut best_last = 0;
    let mut best_score = f32::NEG_INFINITY;
    for label in 0..num_labels {
        let dec = decode_label(label, num_classes);
        if matches!(dec.tag, Tag::O | Tag::E | Tag::S) && dp[seq_len - 1][label] > best_score {
            best_score = dp[seq_len - 1][label];
            best_last = label;
        }
    }

    // Backtrack the best label sequence.
    let mut path = vec![0usize; seq_len];
    path[seq_len - 1] = best_last;
    for t in (1..seq_len).rev() {
        path[t - 1] = bp[t][path[t]];
    }

    // Convert label sequence → spans.
    let mut spans = Vec::new();
    let mut span_start: Option<usize> = None;
    let mut span_class: Option<usize> = None;
    for (t, &label) in path.iter().enumerate() {
        let dec = decode_label(label, num_classes);
        match dec.tag {
            Tag::B => {
                span_start = Some(t);
                span_class = dec.class_idx;
            }
            Tag::I => {
                // Continue (no-op) — invariant maintained by transitions.
            }
            Tag::E => {
                if let (Some(start), Some(c)) = (span_start, span_class) {
                    spans.push((start, t + 1, c));
                }
                span_start = None;
                span_class = None;
            }
            Tag::S => {
                if let Some(c) = dec.class_idx {
                    spans.push((t, t + 1, c));
                }
                span_start = None;
                span_class = None;
            }
            Tag::O => {
                span_start = None;
                span_class = None;
            }
        }
    }
    Ok(spans)
}

fn decode_label(label: usize, num_classes: usize) -> Decoded {
    if label == 0 {
        return Decoded {
            tag: Tag::O,
            class_idx: None,
        };
    }
    let zero_based = label - 1;
    let class_idx = zero_based / 4;
    let tag = match zero_based % 4 {
        0 => Tag::B,
        1 => Tag::I,
        2 => Tag::E,
        3 => Tag::S,
        _ => unreachable!(),
    };
    debug_assert!(class_idx < num_classes);
    Decoded {
        tag,
        class_idx: Some(class_idx),
    }
}

fn is_valid_transition(prev: Decoded, cur: Decoded) -> bool {
    match (prev.tag, cur.tag) {
        // O → O / B-* / S-*
        (Tag::O, Tag::O) | (Tag::O, Tag::B) | (Tag::O, Tag::S) => true,
        // B-X → I-X / E-X
        (Tag::B, Tag::I) | (Tag::B, Tag::E) => prev.class_idx == cur.class_idx,
        // I-X → I-X / E-X
        (Tag::I, Tag::I) | (Tag::I, Tag::E) => prev.class_idx == cur.class_idx,
        // E-X → O / B-* / S-*
        (Tag::E, Tag::O) | (Tag::E, Tag::B) | (Tag::E, Tag::S) => true,
        // S-X → O / B-* / S-*
        (Tag::S, Tag::O) | (Tag::S, Tag::B) | (Tag::S, Tag::S) => true,
        _ => false,
    }
}

/// Greedy argmax decoder for parity comparison and quick checks. No
/// transition constraints applied; the BIOES contract may be violated.
pub fn decode_argmax(logits: &Tensor, num_classes: usize) -> Result<Vec<(usize, usize, usize)>> {
    let (_seq_len, _num_labels) = logits.dims2()?;
    let labels: Vec<u32> = logits.argmax(D::Minus1)?.to_vec1()?;
    let mut spans = Vec::new();
    let mut start: Option<usize> = None;
    let mut class: Option<usize> = None;
    for (t, &lab) in labels.iter().enumerate() {
        let dec = decode_label(lab as usize, num_classes);
        match dec.tag {
            Tag::B => {
                start = Some(t);
                class = dec.class_idx;
            }
            Tag::E => {
                if let (Some(s), Some(c)) = (start, class) {
                    spans.push((s, t + 1, c));
                }
                start = None;
                class = None;
            }
            Tag::S => {
                if let Some(c) = dec.class_idx {
                    spans.push((t, t + 1, c));
                }
            }
            _ => {}
        }
    }
    Ok(spans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    fn one_hot(seq_len: usize, num_labels: usize, labels: &[usize]) -> Tensor {
        let mut data = vec![-10.0f32; seq_len * num_labels];
        for (t, &l) in labels.iter().enumerate() {
            data[t * num_labels + l] = 5.0;
        }
        Tensor::from_vec(data, (seq_len, num_labels), &Device::Cpu).unwrap()
    }

    #[test]
    fn decodes_single_span() {
        // 2 classes ⇒ 9 labels. Sequence: O B-0 I-0 E-0 O
        let logits = one_hot(5, 9, &[0, 1, 2, 3, 0]);
        let spans = decode(&logits, 2).unwrap();
        assert_eq!(spans, vec![(1, 4, 0)]);
    }

    #[test]
    fn decodes_singleton_span() {
        let logits = one_hot(3, 9, &[0, 4, 0]); // S-0 at t=1
        let spans = decode(&logits, 2).unwrap();
        assert_eq!(spans, vec![(1, 2, 0)]);
    }

    #[test]
    fn forbids_class_switch_mid_span() {
        // O B-0 I-1(invalid) → Viterbi must reroute.
        // We verify it doesn't return a span crossing classes.
        let logits = one_hot(3, 9, &[0, 1, 6]); // I-1 = label 6
        let spans = decode(&logits, 2).unwrap();
        for (_, _, class) in &spans {
            assert!(*class < 2);
        }
    }
}

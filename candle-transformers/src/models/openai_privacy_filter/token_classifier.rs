//! Convenience wrapper that runs the full forward pass and returns
//! per-token argmax labels — for callers that don't need Viterbi
//! constrained decoding (regression / quick-check use).

use candle::{Result, Tensor, D};

use super::{Config, OpenAIPrivacyFilterForTokenClassification};
use candle_nn::VarBuilder;

pub struct TokenClassifier {
    inner: OpenAIPrivacyFilterForTokenClassification,
}

impl TokenClassifier {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            inner: OpenAIPrivacyFilterForTokenClassification::new(cfg, vb)?,
        })
    }

    pub fn forward_logits(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.inner.forward(input_ids)
    }

    /// Per-token argmax labels — shape `(batch, seq_len)`.
    /// Use [`super::viterbi::decode`] for constrained BIOES decoding.
    pub fn forward_labels(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(input_ids)?;
        logits.argmax(D::Minus1)
    }
}

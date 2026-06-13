//! Shared Hugging Face `tokenizer.json` model-type inference.

use alloc::vec::Vec;

#[cfg(feature = "std")]
use anyhow::{bail, Context, Result};

#[cfg(feature = "std")]
pub(crate) fn infer_model_type(model: &serde_json::Value) -> Result<&str> {
    if let Some(model_type) = model.get("type").and_then(|value| value.as_str()) {
        return Ok(model_type);
    }

    if model
        .get("merges")
        .and_then(|value| value.as_array())
        .is_some()
    {
        return Ok("BPE");
    }
    if has_wordpiece_shape(model) {
        return Ok("WordPiece");
    }
    if model
        .get("vocab")
        .and_then(|value| value.as_array())
        .is_some()
    {
        return Ok("Unigram");
    }

    let keys = model
        .as_object()
        .context("tokenizer model must be a JSON object")?
        .keys()
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    bail!("unable to infer tokenizer model type from model keys: [{keys}]")
}

#[cfg(feature = "std")]
fn has_wordpiece_shape(model: &serde_json::Value) -> bool {
    model
        .get("vocab")
        .and_then(|value| value.as_object())
        .is_some()
        && (model.get("continuing_subword_prefix").is_some()
            || model.get("max_input_chars_per_word").is_some()
            || model.get("unk_token").is_some())
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::infer_model_type;

    #[test]
    fn infers_wordpiece_without_model_type() {
        let json = serde_json::json!({
            "vocab": {
                "[PAD]": 0,
                "[UNK]": 1,
                "hello": 2,
                "##s": 3
            },
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "unk_token": "[UNK]"
        });

        assert_eq!(infer_model_type(&json).unwrap(), "WordPiece");
    }

    #[test]
    fn infers_bpe_before_wordpiece_when_merges_are_present() {
        let json = serde_json::json!({
            "vocab": {
                "h": 0,
                "i": 1
            },
            "merges": ["h i"],
            "continuing_subword_prefix": "##"
        });

        assert_eq!(infer_model_type(&json).unwrap(), "BPE");
    }
}

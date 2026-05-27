//! Model addressing (V&V class MA, architecture §8).
//!
//! A model file folds to a `uor-addr` **κ-label** — a typed, σ-projection-
//! grounded, replayable content address (`<axis>:<hex>`) — via the uor-addr
//! realization that matches its format. The κ-label is the model's canonical
//! identity: structurally-equivalent models collapse to one label, so it is
//! the dedup / warm-start cache key. Byte-level parsing stays inside uor-addr's
//! canonical-form pipeline; hologram-ai only hands it the raw bytes.
//!
//! The default σ-axis is SHA-256 — byte-identical to the labels pinned in
//! uor-addr's authoritative `tests/external_models.rs`, against which the
//! conformance harness validates ([`crate::address::model_kappa`]).
//!
//! Multi-component models compose with the order-independent E₈ product
//! (`hologram_archive::address::compose_model`); a single model file addresses
//! directly here.

use anyhow::{bail, Result};

/// Model file format, selecting the uor-addr canonical realization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// ONNX `ModelProto` (protobuf).
    Onnx,
    /// GGUF v3 (`GGUF` magic).
    Gguf,
    /// JSON (JCS-RFC8785 canonical form) — e.g. `tokenizer.json`, `config.json`.
    Json,
}

impl ModelFormat {
    /// Detect the format from a file's leading bytes. GGUF carries the
    /// `GGUF` magic; JSON begins (after optional whitespace) with `{` or `[`;
    /// everything else is treated as ONNX protobuf. Returns `None` when the
    /// bytes match no known model format.
    pub fn detect(bytes: &[u8]) -> Option<Self> {
        if bytes.len() >= 4 && &bytes[..4] == b"GGUF" {
            return Some(ModelFormat::Gguf);
        }
        match bytes.iter().find(|b| !b.is_ascii_whitespace()) {
            Some(b'{') | Some(b'[') => Some(ModelFormat::Json),
            // A protobuf `ModelProto` starts with a field tag; ONNX models in
            // practice begin with field 1 (`ir_version`, tag 0x08) or field 7
            // (`graph`, tag 0x3a). Anything non-empty that is not JSON/GGUF is
            // addressed as ONNX.
            Some(_) => Some(ModelFormat::Onnx),
            None => None,
        }
    }
}

/// The κ-label minted for a model on the SHA-256 σ-axis (71 bytes,
/// `sha256:<64 hex>`). Carries the replayable TC-05 witness.
pub type ModelOutcome = uor_addr::onnx::AddressOutcome<71>;

/// Address `bytes` (a model file of the given `format`) to its uor-addr
/// κ-label outcome on the SHA-256 axis. The returned [`ModelOutcome`] holds
/// both the κ-label (`outcome.address`) and a replayable witness
/// (`outcome.witness.verify()`).
pub fn model_kappa(format: ModelFormat, bytes: &[u8]) -> Result<ModelOutcome> {
    match format {
        ModelFormat::Onnx => {
            uor_addr::onnx::address(bytes).map_err(|e| anyhow::anyhow!("onnx address: {e:?}"))
        }
        ModelFormat::Gguf => {
            uor_addr::gguf::address(bytes).map_err(|e| anyhow::anyhow!("gguf address: {e:?}"))
        }
        ModelFormat::Json => {
            uor_addr::json::address(bytes).map_err(|e| anyhow::anyhow!("json address: {e:?}"))
        }
    }
}

/// Mint a model's κ-label string (`<axis>:<hex>`), auto-detecting the format
/// from the bytes. This is the model identity used for dedup / warm-start.
pub fn model_kappa_label(bytes: &[u8]) -> Result<String> {
    let Some(format) = ModelFormat::detect(bytes) else {
        bail!("unrecognized model format (not GGUF, JSON, or ONNX protobuf)");
    };
    Ok(model_kappa(format, bytes)?.address.as_str().to_string())
}

// ── Multi-component composition (MA-2, architecture §8) ──────────────────────
//
// A model assembled from several files (e.g. a diffusion pipeline's text
// encoder + UNet + VAE, or a sharded LLM) has one identity: the E₈ categorical
// composition of its components' κ-labels. Composition runs on hologram's
// canonical BLAKE3 σ-axis (ADR-052), so component labels for composition are
// minted on that axis ([`component_kappa`]) — distinct from the SHA-256
// single-model identity that matches uor-addr's pinned corpus.

/// hologram's content-address type (a BLAKE3-σ-axis κ-label, 71 bytes).
pub use hologram_archive::ContentLabel;

/// Address one component file (auto-detecting its format) on the BLAKE3 axis,
/// yielding the κ-label used as a [`compose_model`] operand.
pub fn component_kappa(bytes: &[u8]) -> Result<ContentLabel> {
    let Some(format) = ModelFormat::detect(bytes) else {
        bail!("unrecognized component format (not GGUF, JSON, or ONNX protobuf)");
    };
    let outcome = match format {
        ModelFormat::Onnx => {
            uor_addr::onnx::address_blake3(bytes).map_err(|e| anyhow::anyhow!("onnx: {e:?}"))?
        }
        ModelFormat::Gguf => {
            uor_addr::gguf::address_blake3(bytes).map_err(|e| anyhow::anyhow!("gguf: {e:?}"))?
        }
        ModelFormat::Json => {
            uor_addr::json::address_blake3(bytes).map_err(|e| anyhow::anyhow!("json: {e:?}"))?
        }
    };
    Ok(outcome.address)
}

/// Compose component κ-labels into one model identity via the E₈ (CS-G2)
/// product. The composed label is a deterministic, axis-homogeneous function of
/// the component labels and is **independent of their order** (architecture §8,
/// MA-2): a model assembled from the same parts addresses identically however
/// it was put together.
///
/// The underlying CS-G2 product is pairwise-commutative but the fold over 3+
/// operands is order-sensitive, so the parts are first put into a canonical
/// (label-byte) order — that is what makes the composed identity a pure
/// function of the *set* of components.
pub fn compose_model(parts: &[ContentLabel]) -> Result<ContentLabel> {
    let mut ordered: Vec<ContentLabel> = parts.to_vec();
    ordered.sort_unstable_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    hologram_archive::compose_model(&ordered).map_err(|e| anyhow::anyhow!("compose_model: {e:?}"))
}

/// Address each component file (BLAKE3 axis) and compose them into a single
/// multi-component model identity.
pub fn compose_models(components: &[&[u8]]) -> Result<ContentLabel> {
    let parts = components
        .iter()
        .map(|b| component_kappa(b))
        .collect::<Result<Vec<_>>>()?;
    compose_model(&parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_gguf_magic() {
        assert_eq!(
            ModelFormat::detect(b"GGUF\x03\0\0\0"),
            Some(ModelFormat::Gguf)
        );
    }

    #[test]
    fn detect_json_object_and_array() {
        assert_eq!(ModelFormat::detect(b"  {\"a\":1}"), Some(ModelFormat::Json));
        assert_eq!(ModelFormat::detect(b"[1,2,3]"), Some(ModelFormat::Json));
    }

    #[test]
    fn detect_onnx_protobuf_tag() {
        // ONNX ModelProto begins with field 1 (ir_version), wire tag 0x08.
        assert_eq!(ModelFormat::detect(&[0x08, 0x07]), Some(ModelFormat::Onnx));
    }

    #[test]
    fn detect_empty_is_none() {
        assert_eq!(ModelFormat::detect(&[]), None);
        assert_eq!(ModelFormat::detect(b"   "), None);
    }

    #[test]
    fn json_label_is_sha256_kappa() {
        // A tiny canonical JSON addresses to a well-formed sha256 κ-label.
        let label = model_kappa_label(b"{\"k\":1}").expect("json kappa");
        assert!(label.starts_with("sha256:"), "got {label}");
        // sha256 κ-label width: "sha256:" (7) + 64 hex = 71 bytes.
        assert_eq!(label.len(), 71);
    }
}

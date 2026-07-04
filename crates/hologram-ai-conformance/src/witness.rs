//! Shared witness helpers bound by the BDD steps and reused by test targets.
//!
//! The safetensors streaming-header walk lives here so the acquisition test
//! helper (`tests/fetch_helper.rs`) and the `safetensors-header-streaming`
//! BDD witness exercise the *same* parse — the row diffs one code path
//! against the reference `safetensors` crate, never a per-test reimplementation.

use anyhow::{Context, Result};
use hologram_ai_common::DType;

/// One tensor entry from a streamed safetensors JSON header.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamedTensorMeta {
    /// Tensor name (the header key).
    pub name: String,
    /// Storage dtype, mapped to the IR vocabulary.
    pub dtype: DType,
    /// Row-major shape.
    pub shape: Vec<u64>,
    /// `[begin, end)` byte range within the data section (after the header).
    pub data_offsets: (u64, u64),
}

/// Map a safetensors header dtype code to the IR dtype. Codes outside the
/// mapped set fall back to the F32 compute type — the same posture the
/// streamed acquisition path takes for exotic storage types.
pub fn safetensors_dtype(code: &str) -> DType {
    match code {
        "F32" => DType::F32,
        "F16" => DType::F16,
        "BF16" => DType::BF16,
        "I64" => DType::INT64,
        "I32" => DType::INT32,
        "I8" => DType::INT8,
        "U8" => DType::U8,
        "BOOL" => DType::BOOL,
        _ => DType::F32,
    }
}

/// Parse a safetensors JSON header (the bytes *after* the 8-byte
/// little-endian length prefix) into its tensor entries, skipping the
/// `__metadata__` block. This is the streaming header walk: it consumes only
/// the header, never tensor data.
pub fn parse_streamed_header(header: &[u8]) -> Result<Vec<StreamedTensorMeta>> {
    let json: serde_json::Value =
        serde_json::from_slice(header).context("safetensors header is not valid JSON")?;
    let obj = json
        .as_object()
        .context("safetensors header is not a JSON object")?;

    let mut entries = Vec::new();
    for (name, meta) in obj {
        if name == "__metadata__" {
            continue;
        }
        let meta = meta
            .as_object()
            .with_context(|| format!("header entry `{name}` is not a JSON object"))?;
        let dtype = safetensors_dtype(meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32"));
        let shape: Vec<u64> = meta
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|a| a.iter().map(|d| d.as_u64().unwrap_or(1)).collect())
            .unwrap_or_default();
        let offsets = meta
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .with_context(|| format!("header entry `{name}` carries no data_offsets"))?;
        let begin = offsets
            .first()
            .and_then(|v| v.as_u64())
            .with_context(|| format!("header entry `{name}` has a malformed range begin"))?;
        let end = offsets
            .get(1)
            .and_then(|v| v.as_u64())
            .with_context(|| format!("header entry `{name}` has a malformed range end"))?;
        entries.push(StreamedTensorMeta {
            name: name.clone(),
            dtype,
            shape,
            data_offsets: (begin, end),
        });
    }
    Ok(entries)
}

/// Split a whole in-memory safetensors file into its 8-byte length prefix's
/// declared header slice and the trailing data section.
pub fn split_safetensors(bytes: &[u8]) -> Result<(&[u8], &[u8])> {
    let len_bytes: [u8; 8] = bytes
        .get(..8)
        .context("file shorter than the 8-byte header length")?
        .try_into()
        .expect("an 8-byte slice converts to [u8; 8]");
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let header = bytes
        .get(8..8 + header_len)
        .context("declared header length exceeds the file")?;
    let data = &bytes[8 + header_len..];
    Ok((header, data))
}

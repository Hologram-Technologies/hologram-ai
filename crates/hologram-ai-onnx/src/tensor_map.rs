//! Convert `TensorProto` initializers to `AiParam`.

use crate::{dtype_map::onnx_dtype, onnx_pb::TensorProto};
use anyhow::Context;
use hologram_ai_common::{shape_from_concrete, AiParam, DType, QuantDescriptor, TensorInfo};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Convert an ONNX `TensorProto` to `(AiParam, TensorInfo)`.
///
/// `model_dir` is the directory containing the `.onnx` file, used to resolve
/// relative paths for tensors stored in external data files.
pub fn tensor_to_param(
    t: &TensorProto,
    model_dir: Option<&Path>,
) -> anyhow::Result<(AiParam, TensorInfo)> {
    let dtype = onnx_dtype(t.data_type).with_context(|| {
        format!(
            "unsupported ONNX data_type {} in tensor '{}'",
            t.data_type, t.name
        )
    })?;

    let shape = shape_from_concrete(&t.dims.iter().map(|&d| d as u64).collect::<Vec<_>>());

    let info = TensorInfo {
        logical_dtype: dtype,
        storage_dtype: dtype,
        shape,
        quant: QuantDescriptor::none(),
        known_i64_values: None,
    };

    // Use Mmap for external data to avoid loading multi-GB weight files into memory.
    if t.data_location == crate::onnx_pb::tensor_proto::DataLocation::External as i32 {
        let param = mmap_external_data(t, model_dir, info.clone())?;
        return Ok((param, info));
    }

    let data = extract_raw_bytes(t, dtype, model_dir)?;
    let param = AiParam::inline(data, info.clone());

    Ok((param, info))
}

/// Extract raw bytes from a `TensorProto` (handles external data, `raw_data`, and typed fields).
fn extract_raw_bytes(
    t: &TensorProto,
    dtype: DType,
    model_dir: Option<&Path>,
) -> anyhow::Result<Vec<u8>> {
    // External data: weights stored in a separate file.
    if t.data_location == crate::onnx_pb::tensor_proto::DataLocation::External as i32 {
        return load_external_data(t, model_dir);
    }

    if !t.raw_data.is_empty() {
        return Ok(t.raw_data.clone());
    }

    // Typed fields — convert to bytes.
    match dtype {
        DType::F32 => {
            let bytes: Vec<u8> = t.float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            Ok(bytes)
        }
        DType::F16 => {
            let bytes: Vec<u8> = t
                .float_data
                .iter()
                .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DType::INT8 => {
            let bytes: Vec<u8> = t.int32_data.iter().map(|&i| i as i8 as u8).collect();
            Ok(bytes)
        }
        DType::U8 | DType::BOOL => {
            let bytes: Vec<u8> = t.int32_data.iter().map(|&i| i as u8).collect();
            Ok(bytes)
        }
        DType::INT32 => {
            let bytes: Vec<u8> = t.int32_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            Ok(bytes)
        }
        DType::INT64 => {
            let bytes: Vec<u8> = t.int64_data.iter().flat_map(|i| i.to_le_bytes()).collect();
            Ok(bytes)
        }
        _ => {
            anyhow::bail!(
                "cannot extract bytes from TensorProto '{}' with dtype {:?}",
                t.name,
                dtype
            )
        }
    }
}

/// Create an `AiParam::Mmap` for a tensor stored in an external data file.
fn mmap_external_data(
    t: &TensorProto,
    model_dir: Option<&Path>,
    info: TensorInfo,
) -> anyhow::Result<AiParam> {
    let model_dir = model_dir.ok_or_else(|| {
        anyhow::anyhow!(
            "tensor '{}' uses external data but no model directory was provided",
            t.name,
        )
    })?;

    let mut location: Option<&str> = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for entry in &t.external_data {
        match entry.key.as_str() {
            "location" => location = Some(&entry.value),
            "offset" => {
                offset = entry.value.parse().with_context(|| {
                    format!(
                        "tensor '{}': invalid external_data offset '{}'",
                        t.name, entry.value
                    )
                })?
            }
            "length" => {
                length = Some(entry.value.parse().with_context(|| {
                    format!(
                        "tensor '{}': invalid external_data length '{}'",
                        t.name, entry.value
                    )
                })?)
            }
            _ => {}
        }
    }

    let rel_path = location.ok_or_else(|| {
        anyhow::anyhow!("tensor '{}': external_data missing 'location' key", t.name)
    })?;

    let data_path = model_dir.join(rel_path);
    let file_len = std::fs::metadata(&data_path)
        .with_context(|| format!("stat external data file {data_path:?}"))?
        .len();

    let len = length.unwrap_or(file_len - offset);

    Ok(AiParam::Mmap {
        path: data_path,
        offset,
        len,
        info,
    })
}

/// Load tensor data from an external file referenced by `TensorProto.external_data`.
fn load_external_data(t: &TensorProto, model_dir: Option<&Path>) -> anyhow::Result<Vec<u8>> {
    let model_dir = model_dir.ok_or_else(|| {
        anyhow::anyhow!(
            "tensor '{}' uses external data but no model directory was provided \
             (use import_onnx_path instead of import_onnx)",
            t.name,
        )
    })?;

    let mut location: Option<&str> = None;
    let mut offset: u64 = 0;
    let mut length: Option<u64> = None;

    for entry in &t.external_data {
        match entry.key.as_str() {
            "location" => location = Some(&entry.value),
            "offset" => {
                offset = entry.value.parse().with_context(|| {
                    format!(
                        "tensor '{}': invalid external_data offset '{}'",
                        t.name, entry.value
                    )
                })?
            }
            "length" => {
                length = Some(entry.value.parse().with_context(|| {
                    format!(
                        "tensor '{}': invalid external_data length '{}'",
                        t.name, entry.value
                    )
                })?)
            }
            _ => {}
        }
    }

    let rel_path = location.ok_or_else(|| {
        anyhow::anyhow!("tensor '{}': external_data missing 'location' key", t.name)
    })?;

    let data_path = model_dir.join(rel_path);
    let mut file = std::fs::File::open(&data_path).with_context(|| {
        format!(
            "opening external data file {data_path:?} for tensor '{}'",
            t.name
        )
    })?;

    let read_len = match length {
        Some(len) => len,
        None => {
            let file_len = file.metadata()?.len();
            file_len - offset
        }
    };

    if offset > 0 {
        file.seek(SeekFrom::Start(offset))?;
    }

    let mut buf = vec![0u8; read_len as usize];
    file.read_exact(&mut buf).with_context(|| {
        format!(
            "reading {} bytes at offset {} from {data_path:?}",
            read_len, offset
        )
    })?;

    Ok(buf)
}

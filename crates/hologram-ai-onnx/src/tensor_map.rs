//! Convert `TensorProto` initializers to `AiParam`.

use anyhow::Context;
use hologram_ai_common::{AiParam, DType, TensorInfo, shape_from_concrete, QuantDescriptor};
use crate::{dtype_map::onnx_dtype, onnx_pb::TensorProto};

/// Convert an ONNX `TensorProto` to `(AiParam, TensorInfo)`.
pub fn tensor_to_param(t: &TensorProto) -> anyhow::Result<(AiParam, TensorInfo)> {
    let dtype = onnx_dtype(t.data_type)
        .with_context(|| format!("unsupported ONNX data_type {} in tensor '{}'", t.data_type, t.name))?;

    let shape = shape_from_concrete(
        &t.dims.iter().map(|&d| d as u64).collect::<Vec<_>>()
    );

    let info = TensorInfo {
        logical_dtype: dtype,
        storage_dtype: dtype,
        shape,
        quant: QuantDescriptor::none(),
    };

    let data = extract_raw_bytes(t, dtype)?;
    let param = AiParam::inline(data, info.clone());

    Ok((param, info))
}

/// Extract raw bytes from a `TensorProto` (handles `raw_data` and typed fields).
fn extract_raw_bytes(t: &TensorProto, dtype: DType) -> anyhow::Result<Vec<u8>> {
    if !t.raw_data.is_empty() {
        return Ok(t.raw_data.clone());
    }

    // Typed fields — convert to bytes.
    match dtype {
        DType::F32 => {
            let bytes: Vec<u8> = t.float_data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DType::F16 => {
            let bytes: Vec<u8> = t.float_data.iter()
                .flat_map(|f| half::f16::from_f32(*f).to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DType::INT8 => {
            let bytes: Vec<u8> = t.int32_data.iter()
                .map(|&i| i as i8 as u8)
                .collect();
            Ok(bytes)
        }
        DType::U8 => {
            let bytes: Vec<u8> = t.int32_data.iter()
                .map(|&i| i as u8)
                .collect();
            Ok(bytes)
        }
        _ => {
            if !t.int64_data.is_empty() {
                let bytes: Vec<u8> = t.int64_data.iter()
                    .flat_map(|i| i.to_le_bytes())
                    .collect();
                Ok(bytes)
            } else {
                anyhow::bail!("cannot extract bytes from TensorProto '{}' with dtype {:?}", t.name, dtype)
            }
        }
    }
}

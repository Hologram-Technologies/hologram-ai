//! Map ONNX `TensorProto::DataType` integers to `DType`.

use hologram_ai_common::DType;

/// Map an ONNX data_type integer to `DType`.
///
/// Returns `None` for types not supported by hologram-ai (e.g. complex, string).
pub fn onnx_dtype(data_type: i32) -> Option<DType> {
    match data_type {
        1  => Some(DType::F32),
        10 => Some(DType::F16),
        16 => Some(DType::BF16),
        3  => Some(DType::INT8),
        2  => Some(DType::U8),
        // UINT4 / INT4 → INT4 (hologram packs nibbles)
        21 | 22 => Some(DType::INT4),
        _ => None,
    }
}

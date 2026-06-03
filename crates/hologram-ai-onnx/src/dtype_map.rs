//! Map ONNX `TensorProto::DataType` integers to `DType`.

use hologram_ai_common::DType;

/// Map an ONNX data_type integer to `DType`.
///
/// Returns `None` for types not supported by hologram-ai (e.g. complex, string).
///
/// Widening casts for uncommon types:
/// - UINT16 (4) → INT32 (no native u16)
/// - INT16 (5) → INT16
/// - UINT32 (12) → INT64 (no native u32)
/// - UINT64 (13) → INT64 (no native u64)
/// - FLOAT64 (11) → F64
pub fn onnx_dtype(data_type: i32) -> Option<DType> {
    match data_type {
        1 => Some(DType::F32),
        11 => Some(DType::F64),
        10 => Some(DType::F16),
        16 => Some(DType::BF16),
        3 => Some(DType::INT8),
        2 => Some(DType::U8),
        5 => Some(DType::INT16),
        6 => Some(DType::INT32),
        7 => Some(DType::INT64),
        9 => Some(DType::BOOL),
        // UINT4 / INT4 → INT4 (hologram packs nibbles)
        21 | 22 => Some(DType::INT4),
        // Widening casts for types without native representation.
        4 => Some(DType::INT32),  // UINT16 → INT32
        12 => Some(DType::INT64), // UINT32 → INT64
        13 => Some(DType::INT64), // UINT64 → INT64
        _ => None,
    }
}

/// Returns true if the ONNX dtype was widened during import.
#[allow(dead_code)]
pub fn is_widened_dtype(data_type: i32) -> bool {
    matches!(data_type, 4 | 12 | 13)
}

/// Human-readable name for an ONNX data_type integer.
pub fn onnx_dtype_name(data_type: i32) -> &'static str {
    match data_type {
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        16 => "BFLOAT16",
        21 => "UINT4",
        22 => "INT4",
        _ => "UNKNOWN",
    }
}

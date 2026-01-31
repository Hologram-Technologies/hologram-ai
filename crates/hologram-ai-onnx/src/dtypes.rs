//! Data type conversions between ONNX and hologram.

use anyhow::{Result, bail};
use hologram::compiler::DType;

/// Convert ONNX tensor data type to hologram DType.
///
/// Maps ONNX protobuf type enum to hologram's type system.
pub fn from_onnx(onnx_dtype: i32) -> Result<DType> {
    match onnx_dtype {
        1 => Ok(DType::F32),  // FLOAT
        2 => Ok(DType::U8),   // UINT8
        3 => Ok(DType::I8),   // INT8
        5 => Ok(DType::I16),  // INT16
        6 => Ok(DType::I32),  // INT32
        7 => Ok(DType::I64),  // INT64
        9 => Ok(DType::Bool), // BOOL
        10 => Ok(DType::F64), // DOUBLE
        _ => bail!("Unsupported ONNX dtype: {}", onnx_dtype),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(from_onnx(1).unwrap(), DType::F32);
        assert_eq!(from_onnx(6).unwrap(), DType::I32);
        assert_eq!(from_onnx(9).unwrap(), DType::Bool);
        assert_eq!(from_onnx(10).unwrap(), DType::F64);
    }

    #[test]
    fn test_unsupported_dtype() {
        assert!(from_onnx(999).is_err());
    }
}

//! ONNX advanced operations.

use hologram_ir::{GraphBuilder, NodeIndex, DType};
use crate::core::{OnnxError, Result};
use crate::proto::AttributeProto;
use crate::ops::utils::parse_attr_int;

/// Translate ONNX Cast to IR.
pub fn translate_cast(
    inputs: &[NodeIndex],
    attrs: &[AttributeProto],
    builder: &mut GraphBuilder,
) -> Result<Vec<NodeIndex>> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel("Cast requires 1 input".into()));
    }

    let to_type = parse_attr_int(attrs, "to", 1)?;

    // Convert ONNX type to DType
    let dtype = match to_type {
        1 => DType::F32,
        2 => DType::U8,
        3 => DType::I8,
        6 => DType::I32,
        7 => DType::I64,
        10 => DType::F16,
        11 => DType::F64,
        _ => DType::F32,
    };

    let result = builder.cast(inputs[0], dtype)?;

    Ok(vec![result])
}

//! Cast operation translator.

use crate::proto::NodeProto;
use crate::translators::{InputRequirement, OnnxAttributes, OnnxTranslator, TranslationError};
use hologram::ir::{ConstantData, DType, GraphBuilder, NodeIndex, NodeOp};

/// ONNX type to DType lookup table.
/// Index is ONNX type enum, value is Option<DType>.
/// None means unsupported/unknown type (defaults to F32).
const ONNX_TYPE_TO_DTYPE: [(i64, DType); 7] = [
    (1, DType::F32),  // FLOAT
    (2, DType::U8),   // UINT8
    (3, DType::I8),   // INT8
    (6, DType::I32),  // INT32
    (7, DType::I64),  // INT64
    (10, DType::F16), // FLOAT16
    (11, DType::F64), // DOUBLE
];

/// Translator for ONNX Cast operation.
///
/// Cast(input, to) converts the input tensor to the specified data type.
///
/// # ONNX Specification
///
/// - Inputs: input
/// - Attributes:
///   - to (required): Target data type (ONNX type enum)
/// - Output: output tensor with same shape, converted type
///
/// # Supported ONNX Types
///
/// - 1: FLOAT (f32)
/// - 2: UINT8 (u8)
/// - 3: INT8 (i8)
/// - 6: INT32 (i32)
/// - 7: INT64 (i64)
/// - 10: FLOAT16 (f16)
/// - 11: DOUBLE (f64)
#[derive(Debug, Default)]
pub struct CastTranslator;

impl OnnxTranslator for CastTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Cast"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(1)
    }

    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        let to_type = node.get_int_or("to", 1);

        // Convert ONNX type to DType
        let dtype = Self::onnx_type_to_dtype(to_type);

        // Check for constant folding opportunity
        let input_node = builder
            .graph()
            .node(inputs[0])
            .ok_or_else(|| TranslationError::IrBuilder("Cast: input not found".to_string()))?;

        if let NodeOp::Constant { data } = &input_node.op.op {
            // Attempt constant folding for common type casts
            if let Some(folded_data) = Self::constant_fold_cast(data, dtype) {
                let output_shape = input_node.op.shape.clone();
                let result = builder.constant(folded_data, output_shape);
                return Ok(vec![result]);
            }
        }

        // No constant folding, create regular cast node
        let result = builder
            .cast(inputs[0], dtype)
            .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;

        Ok(vec![result])
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

impl CastTranslator {
    /// Convert ONNX type enum to hologram DType using lookup table.
    fn onnx_type_to_dtype(onnx_type: i64) -> DType {
        ONNX_TYPE_TO_DTYPE
            .iter()
            .find(|(t, _)| *t == onnx_type)
            .map(|(_, dtype)| *dtype)
            .unwrap_or(DType::F32) // Default to F32 for unknown types
    }

    /// Attempt constant folding for common type conversions.
    ///
    /// Supports conversions between I32, I64, and F32 types.
    fn constant_fold_cast(data: &ConstantData, dtype: DType) -> Option<ConstantData> {
        match data {
            ConstantData::I64(values) => Self::cast_i64(values, dtype),
            ConstantData::I32(values) => Self::cast_i32(values, dtype),
            ConstantData::F32(values) => Self::cast_f32(values, dtype),
            _ => None, // Unsupported source types
        }
    }

    /// Cast I64 values to target dtype.
    fn cast_i64(values: &[i64], dtype: DType) -> Option<ConstantData> {
        match dtype {
            DType::I64 => Some(ConstantData::I64(values.to_vec())),
            DType::I32 => Some(ConstantData::I32(
                values.iter().map(|&v| v as i32).collect(),
            )),
            DType::F32 => Some(ConstantData::F32(
                values.iter().map(|&v| v as f32).collect(),
            )),
            _ => None,
        }
    }

    /// Cast I32 values to target dtype.
    fn cast_i32(values: &[i32], dtype: DType) -> Option<ConstantData> {
        match dtype {
            DType::I32 => Some(ConstantData::I32(values.to_vec())),
            DType::I64 => Some(ConstantData::I64(
                values.iter().map(|&v| v as i64).collect(),
            )),
            DType::F32 => Some(ConstantData::F32(
                values.iter().map(|&v| v as f32).collect(),
            )),
            _ => None,
        }
    }

    /// Cast F32 values to target dtype.
    fn cast_f32(values: &[f32], dtype: DType) -> Option<ConstantData> {
        match dtype {
            DType::F32 => Some(ConstantData::F32(values.to_vec())),
            DType::I64 => Some(ConstantData::I64(
                values.iter().map(|&v| v as i64).collect(),
            )),
            DType::I32 => Some(ConstantData::I32(
                values.iter().map(|&v| v as i32).collect(),
            )),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;
    use hologram::ir::Shape;

    fn make_node_with_to(to_type: i64) -> NodeProto {
        NodeProto {
            name: "cast_test".to_string(),
            op_type: "Cast".to_string(),
            attribute: vec![AttributeProto {
                name: "to".to_string(),
                i: to_type,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_cast_f32_to_i32() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let node = make_node_with_to(6); // INT32
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_cast_f32_to_f64() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[2, 3]), DType::F32);

        let node = make_node_with_to(11); // DOUBLE
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cast_i64_to_f32() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[10]), DType::I64);

        let node = make_node_with_to(1); // FLOAT
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cast_constant_folding_i64_to_f32() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let constant = builder.constant(
            ConstantData::I64(vec![1, 2, 3, 4]),
            Shape::static_shape(&[4]),
        );

        let node = make_node_with_to(1); // FLOAT
        let result = translator.translate(&node, &[constant], &mut builder);
        assert!(result.is_ok());

        // Verify constant folding occurred
        let output = result.unwrap();
        let output_node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values, &[1.0, 2.0, 3.0, 4.0]);
            } else {
                panic!("Expected F32 constant data");
            }
        } else {
            panic!("Expected Constant node after folding");
        }
    }

    #[test]
    fn test_cast_constant_folding_i32_to_i64() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let constant = builder.constant(
            ConstantData::I32(vec![10, 20, 30]),
            Shape::static_shape(&[3]),
        );

        let node = make_node_with_to(7); // INT64
        let result = translator.translate(&node, &[constant], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap();
        let output_node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &output_node.op.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values, &[10i64, 20, 30]);
            } else {
                panic!("Expected I64 constant data");
            }
        }
    }

    #[test]
    fn test_cast_unknown_type_defaults_to_f32() {
        let translator = CastTranslator;
        let mut builder = GraphBuilder::new();
        let input = builder.input("input", Shape::static_shape(&[5]), DType::I32);

        let node = make_node_with_to(999); // Unknown type
        let result = translator.translate(&node, &[input], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cast_input_validation() {
        let translator = CastTranslator;

        // 0 inputs should fail
        let err = translator.input_requirement().validate(0, "Cast");
        assert!(err.is_err());

        // 1 input should pass
        assert!(translator.input_requirement().validate(1, "Cast").is_ok());

        // 2 inputs should fail
        let err = translator.input_requirement().validate(2, "Cast");
        assert!(err.is_err());
    }

    #[test]
    fn test_onnx_type_to_dtype() {
        assert!(matches!(CastTranslator::onnx_type_to_dtype(1), DType::F32));
        assert!(matches!(CastTranslator::onnx_type_to_dtype(6), DType::I32));
        assert!(matches!(CastTranslator::onnx_type_to_dtype(7), DType::I64));
        assert!(matches!(CastTranslator::onnx_type_to_dtype(11), DType::F64));
    }
}

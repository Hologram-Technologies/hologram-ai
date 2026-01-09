//! Range operation translator.

use hologram::ir::{GraphBuilder, NodeIndex, NodeOp, ConstantData, Shape};
use crate::proto::NodeProto;
use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};

/// Translator for ONNX Range operation.
///
/// Range(start, limit, delta) generates a sequence of numbers from start
/// to limit (exclusive) with step delta.
///
/// # ONNX Specification
///
/// - Inputs: start, limit, delta (all scalars of same type)
/// - Output: 1D tensor containing the sequence
///
/// Formula: output[i] = start + (i * delta)
/// Number of elements: max(ceil((limit - start) / delta), 0)
#[derive(Debug, Default)]
pub struct RangeTranslator;

impl OnnxTranslator for RangeTranslator {
    fn onnx_op_type(&self) -> &'static str {
        "Range"
    }

    fn input_requirement(&self) -> InputRequirement {
        InputRequirement::Exact(3)
    }

    fn translate(
        &self,
        _node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError> {
        // Extract constant data if all inputs are constants, otherwise use dynamic range
        let constant_data = {
            let start_node = builder.graph().node(inputs[0])
                .ok_or_else(|| TranslationError::IrBuilder("Range: start input not found".to_string()))?;
            let limit_node = builder.graph().node(inputs[1])
                .ok_or_else(|| TranslationError::IrBuilder("Range: limit input not found".to_string()))?;
            let delta_node = builder.graph().node(inputs[2])
                .ok_or_else(|| TranslationError::IrBuilder("Range: delta input not found".to_string()))?;

            // Check if all inputs are constants for constant folding
            match (&start_node.op, &limit_node.op, &delta_node.op) {
                (
                    NodeOp::Constant { data: start_data },
                    NodeOp::Constant { data: limit_data },
                    NodeOp::Constant { data: delta_data },
                ) => Some((start_data.clone(), limit_data.clone(), delta_data.clone())),
                _ => None,
            }
        };

        match constant_data {
            Some((start_data, limit_data, delta_data)) => {
                // Compute range based on data type
                let result = Self::compute_range(&start_data, &limit_data, &delta_data, builder)?;
                Ok(vec![result])
            }
            None => {
                // Dynamic range - use runtime Range operation
                let result = builder
                    .range(inputs[0], inputs[1], inputs[2])
                    .map_err(|e| TranslationError::IrBuilder(e.to_string()))?;
                Ok(vec![result])
            }
        }
    }

    fn supports_constant_folding(&self) -> bool {
        true
    }
}

impl RangeTranslator {
    /// Compute range from constant inputs.
    fn compute_range(
        start_data: &ConstantData,
        limit_data: &ConstantData,
        delta_data: &ConstantData,
        builder: &mut GraphBuilder,
    ) -> Result<NodeIndex, TranslationError> {
        match (start_data, limit_data, delta_data) {
            (ConstantData::I64(start_vec), ConstantData::I64(limit_vec), ConstantData::I64(delta_vec)) => {
                Self::compute_range_i64(start_vec, limit_vec, delta_vec, builder)
            }
            (ConstantData::I32(start_vec), ConstantData::I32(limit_vec), ConstantData::I32(delta_vec)) => {
                Self::compute_range_i32(start_vec, limit_vec, delta_vec, builder)
            }
            (ConstantData::F32(start_vec), ConstantData::F32(limit_vec), ConstantData::F32(delta_vec)) => {
                Self::compute_range_f32(start_vec, limit_vec, delta_vec, builder)
            }
            (ConstantData::F64(start_vec), ConstantData::F64(limit_vec), ConstantData::F64(delta_vec)) => {
                Self::compute_range_f64(start_vec, limit_vec, delta_vec, builder)
            }
            _ => Err(TranslationError::IrBuilder(
                "Range: type mismatch or unsupported data types (all inputs must have same type)".to_string()
            )),
        }
    }

    fn compute_range_i64(
        start_vec: &[i64],
        limit_vec: &[i64],
        delta_vec: &[i64],
        builder: &mut GraphBuilder,
    ) -> Result<NodeIndex, TranslationError> {
        if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
            return Err(TranslationError::IrBuilder("Range: inputs must be non-empty".to_string()));
        }

        let start = start_vec[0];
        let limit = limit_vec[0];
        let delta = delta_vec[0];

        if delta == 0 {
            return Err(TranslationError::IrBuilder("Range: delta cannot be zero".to_string()));
        }

        let num_elements = if (delta > 0 && start >= limit) || (delta < 0 && start <= limit) {
            0
        } else {
            ((limit - start + delta - delta.signum()) / delta).max(0) as usize
        };

        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            values.push(start + (i as i64) * delta);
        }

        let output_shape = Shape::static_shape(&[values.len()]);
        Ok(builder.constant(ConstantData::I64(values), output_shape))
    }

    fn compute_range_i32(
        start_vec: &[i32],
        limit_vec: &[i32],
        delta_vec: &[i32],
        builder: &mut GraphBuilder,
    ) -> Result<NodeIndex, TranslationError> {
        if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
            return Err(TranslationError::IrBuilder("Range: inputs must be non-empty".to_string()));
        }

        let start = start_vec[0];
        let limit = limit_vec[0];
        let delta = delta_vec[0];

        if delta == 0 {
            return Err(TranslationError::IrBuilder("Range: delta cannot be zero".to_string()));
        }

        let num_elements = if (delta > 0 && start >= limit) || (delta < 0 && start <= limit) {
            0
        } else {
            ((limit - start + delta - delta.signum()) / delta).max(0) as usize
        };

        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            values.push(start + (i as i32) * delta);
        }

        let output_shape = Shape::static_shape(&[values.len()]);
        Ok(builder.constant(ConstantData::I32(values), output_shape))
    }

    fn compute_range_f32(
        start_vec: &[f32],
        limit_vec: &[f32],
        delta_vec: &[f32],
        builder: &mut GraphBuilder,
    ) -> Result<NodeIndex, TranslationError> {
        if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
            return Err(TranslationError::IrBuilder("Range: inputs must be non-empty".to_string()));
        }

        let start = start_vec[0];
        let limit = limit_vec[0];
        let delta = delta_vec[0];

        if delta == 0.0 {
            return Err(TranslationError::IrBuilder("Range: delta cannot be zero".to_string()));
        }

        let num_elements = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
            0
        } else {
            ((limit - start) / delta).ceil().max(0.0) as usize
        };

        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            values.push(start + (i as f32) * delta);
        }

        let output_shape = Shape::static_shape(&[values.len()]);
        Ok(builder.constant(ConstantData::F32(values), output_shape))
    }

    fn compute_range_f64(
        start_vec: &[f64],
        limit_vec: &[f64],
        delta_vec: &[f64],
        builder: &mut GraphBuilder,
    ) -> Result<NodeIndex, TranslationError> {
        if start_vec.is_empty() || limit_vec.is_empty() || delta_vec.is_empty() {
            return Err(TranslationError::IrBuilder("Range: inputs must be non-empty".to_string()));
        }

        let start = start_vec[0];
        let limit = limit_vec[0];
        let delta = delta_vec[0];

        if delta == 0.0 {
            return Err(TranslationError::IrBuilder("Range: delta cannot be zero".to_string()));
        }

        let num_elements = if (delta > 0.0 && start >= limit) || (delta < 0.0 && start <= limit) {
            0
        } else {
            ((limit - start) / delta).ceil().max(0.0) as usize
        };

        let mut values = Vec::with_capacity(num_elements);
        for i in 0..num_elements {
            values.push(start + (i as f64) * delta);
        }

        let output_shape = Shape::static_shape(&[values.len()]);
        Ok(builder.constant(ConstantData::F64(values), output_shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node() -> NodeProto {
        NodeProto {
            name: "range_test".to_string(),
            op_type: "Range".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn test_range_i64_ascending() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![9]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![3]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap();
        let node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values, &vec![3, 6]);
            } else {
                panic!("Expected I64 data");
            }
        }
    }

    #[test]
    fn test_range_i64_descending() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![4]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![-2]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap();
        let node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values, &vec![10, 8, 6]);
            }
        }
    }

    #[test]
    fn test_range_f32() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::F32(vec![0.0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::F32(vec![1.0]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::F32(vec![0.25]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap();
        let node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::F32(values) = data {
                assert_eq!(values.len(), 4);
                assert!((values[0] - 0.0).abs() < 1e-6);
                assert!((values[1] - 0.25).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_range_empty_sequence() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![5]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![1]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_ok());

        let output = result.unwrap();
        let node = builder.graph().node(output[0]).unwrap();
        if let NodeOp::Constant { data } = &node.op {
            if let ConstantData::I64(values) = data {
                assert_eq!(values.len(), 0);
            }
        }
    }

    #[test]
    fn test_range_zero_delta_error() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));
        let limit = builder.constant(ConstantData::I64(vec![10]), Shape::static_shape(&[]));
        let delta = builder.constant(ConstantData::I64(vec![0]), Shape::static_shape(&[]));

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("delta cannot be zero"));
    }

    #[test]
    fn test_range_dynamic_inputs() {
        let translator = RangeTranslator;
        let mut builder = GraphBuilder::new();
        let start = builder.input("start", Shape::static_shape(&[]), hologram::ir::DType::I64);
        let limit = builder.input("limit", Shape::static_shape(&[]), hologram::ir::DType::I64);
        let delta = builder.input("delta", Shape::static_shape(&[]), hologram::ir::DType::I64);

        let result = translator.translate(&make_node(), &[start, limit, delta], &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_range_input_validation() {
        let translator = RangeTranslator;

        let err = translator.input_requirement().validate(2, "Range");
        assert!(err.is_err());

        assert!(translator.input_requirement().validate(3, "Range").is_ok());

        let err = translator.input_requirement().validate(4, "Range");
        assert!(err.is_err());
    }
}

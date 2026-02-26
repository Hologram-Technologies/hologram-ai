//! Constant operation - creates a constant tensor from attributes.

use anyhow::{Context, Result, bail};
use hologram::compiler::ConstantData;

use super::{OpTranslator, TranslateContext, TranslateResult};
use crate::{dtypes, proto};

/// ONNX Constant operation.
///
/// Unlike most ops that consume input tensors, Constant extracts its value
/// from a node attribute (TensorProto). This is always constant-folded.
pub struct ConstantOp;

impl OpTranslator for ConstantOp {
    fn op_type(&self) -> &'static str {
        "Constant"
    }

    fn try_fold(
        &self,
        node: &proto::NodeProto,
        _ctx: &TranslateContext,
    ) -> Option<TranslateResult> {
        // Constant always folds - extract from attribute
        self.extract_constant(node).ok()
    }

    fn translate(
        &self,
        node: &proto::NodeProto,
        _ctx: &TranslateContext,
    ) -> Result<TranslateResult> {
        // Constant should always fold, but implement translate for completeness
        self.extract_constant(node)
    }
}

impl ConstantOp {
    fn extract_constant(&self, node: &proto::NodeProto) -> Result<TranslateResult> {
        // Find the value attribute (TensorProto)
        let value_attr = node
            .attribute
            .iter()
            .find(|attr| attr.name == "value")
            .context("Constant has no value attribute")?;

        let tensor = value_attr
            .t
            .as_ref()
            .context("Constant value is not a tensor")?;

        let shape: Vec<usize> = tensor.dims.iter().map(|&d| d as usize).collect();
        let dtype = dtypes::from_onnx(tensor.data_type)?;

        // Extract constant data
        let const_data = extract_constant_data(tensor)?;

        Ok(TranslateResult::constant(shape, dtype, const_data))
    }
}

/// Extract constant data from ONNX TensorProto.
pub fn extract_constant_data(tensor: &proto::TensorProto) -> Result<ConstantData> {
    match tensor.data_type {
        1 => {
            // F32
            if !tensor.float_data.is_empty() {
                Ok(ConstantData::F32(tensor.float_data.clone()))
            } else if !tensor.raw_data.is_empty() {
                let floats: Vec<f32> = bytemuck::cast_slice(&tensor.raw_data).to_vec();
                Ok(ConstantData::F32(floats))
            } else {
                bail!("F32 tensor has no data")
            }
        }
        6 => {
            // I32
            if !tensor.int32_data.is_empty() {
                Ok(ConstantData::I32(tensor.int32_data.clone()))
            } else if !tensor.raw_data.is_empty() {
                let ints: Vec<i32> = bytemuck::cast_slice(&tensor.raw_data).to_vec();
                Ok(ConstantData::I32(ints))
            } else {
                bail!("I32 tensor has no data")
            }
        }
        7 => {
            // I64
            if !tensor.int64_data.is_empty() {
                Ok(ConstantData::I64(tensor.int64_data.clone()))
            } else if !tensor.raw_data.is_empty() {
                let ints: Vec<i64> = bytemuck::cast_slice(&tensor.raw_data).to_vec();
                Ok(ConstantData::I64(ints))
            } else {
                bail!("I64 tensor has no data")
            }
        }
        _ => bail!("Unsupported constant dtype: {}", tensor.data_type),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::{DType, OperationGraph};
    use std::collections::HashMap;

    #[test]
    fn test_constant_f32() {
        let graph = OperationGraph::default();
        let value_to_node = HashMap::new();
        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let tensor = proto::TensorProto {
            dims: vec![2, 3],
            data_type: 1, // F32
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ..Default::default()
        };

        let node = proto::NodeProto {
            output: vec!["const_out".to_string()],
            op_type: "Constant".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "value".to_string(),
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = ConstantOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![2, 3]);
        assert_eq!(result.dtype, DType::F32);
        assert!(result.constant_data.is_some());

        if let Some(ConstantData::F32(data)) = result.constant_data {
            assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        } else {
            panic!("Expected F32 constant data");
        }
    }

    #[test]
    fn test_constant_i64() {
        let graph = OperationGraph::default();
        let value_to_node = HashMap::new();
        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let tensor = proto::TensorProto {
            dims: vec![3],
            data_type: 7, // I64
            int64_data: vec![10, 20, 30],
            ..Default::default()
        };

        let node = proto::NodeProto {
            output: vec!["const_out".to_string()],
            op_type: "Constant".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "value".to_string(),
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        };

        let result = ConstantOp.translate(&node, &ctx).unwrap();
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.dtype, DType::I64);

        if let Some(ConstantData::I64(data)) = result.constant_data {
            assert_eq!(data, vec![10, 20, 30]);
        } else {
            panic!("Expected I64 constant data");
        }
    }

    #[test]
    fn test_constant_try_fold() {
        let graph = OperationGraph::default();
        let value_to_node = HashMap::new();
        let ctx = TranslateContext::new(&graph, &value_to_node, &[]);

        let tensor = proto::TensorProto {
            dims: vec![2],
            data_type: 1,
            float_data: vec![1.0, 2.0],
            ..Default::default()
        };

        let node = proto::NodeProto {
            output: vec!["const_out".to_string()],
            op_type: "Constant".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "value".to_string(),
                t: Some(tensor),
                ..Default::default()
            }],
            ..Default::default()
        };

        // Constant should always fold
        let result = ConstantOp.try_fold(&node, &ctx);
        assert!(result.is_some());
    }
}

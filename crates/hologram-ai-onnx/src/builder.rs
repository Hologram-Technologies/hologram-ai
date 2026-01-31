//! Building hologram OperationGraph from ONNX models.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use hologram::compiler::{ConstantData, OpKind, OpNode, OperationGraph};

use crate::{dtypes, ops, parser, proto};

/// Build a hologram OperationGraph from an ONNX ModelProto.
pub fn build_graph(model: &proto::ModelProto) -> Result<OperationGraph> {
    let graph_proto = model.graph.as_ref().context("ONNX model has no graph")?;

    let mut graph = OperationGraph::default();
    let mut node_id_counter = 0u32;
    let mut value_to_node: HashMap<String, u32> = HashMap::new();

    // Process inputs
    for input in &graph_proto.input {
        let name = input.name.clone();
        let shape = parser::extract_shape(input)?;
        let dtype = parser::extract_dtype(input)?;

        let node =
            OpNode::new(node_id_counter, OpKind::Input, shape, dtype).with_name(name.clone());

        value_to_node.insert(name.clone(), node_id_counter);
        graph.nodes.push(node);
        graph.inputs.push((name, node_id_counter));

        node_id_counter += 1;
    }

    // Process initializers (constants/weights)
    for initializer in &graph_proto.initializer {
        let name = initializer.name.clone();
        let shape = initializer.dims.iter().map(|&d| d as usize).collect();
        let dtype = dtypes::from_onnx(initializer.data_type)?;

        // Create constant node
        let node =
            OpNode::new(node_id_counter, OpKind::Constant, shape, dtype).with_name(name.clone());

        value_to_node.insert(name, node_id_counter);
        graph.nodes.push(node);

        // Extract constant data
        let const_data = extract_constant_data(initializer)?;
        graph.constants.push(const_data);

        node_id_counter += 1;
    }

    // Process operations
    for node_proto in &graph_proto.node {
        let (op_kind, output_shape, output_dtype) =
            ops::translate_node(node_proto, &value_to_node, &graph)?;

        let output_name = node_proto
            .output
            .first()
            .context("Node has no output")?
            .clone();

        let node = OpNode::new(node_id_counter, op_kind, output_shape, output_dtype)
            .with_name(output_name.clone());

        // Add edges from inputs to this node
        for input_name in &node_proto.input {
            if let Some(&input_id) = value_to_node.get(input_name) {
                graph.edges.push((input_id, node_id_counter));
            }
        }

        value_to_node.insert(output_name, node_id_counter);
        graph.nodes.push(node);

        node_id_counter += 1;
    }

    // Process outputs
    for output in &graph_proto.output {
        let name = output.name.clone();

        if let Some(&source_id) = value_to_node.get(&name) {
            let source_node = &graph.nodes[source_id as usize];
            let shape = source_node.shape.clone();
            let dtype = source_node.dtype;

            let output_node =
                OpNode::new(node_id_counter, OpKind::Output, shape, dtype).with_name(name.clone());

            graph.edges.push((source_id, node_id_counter));
            graph.outputs.push((name, node_id_counter));
            graph.nodes.push(output_node);

            node_id_counter += 1;
        }
    }

    Ok(graph)
}

/// Extract constant data from ONNX TensorProto.
fn extract_constant_data(tensor: &proto::TensorProto) -> Result<ConstantData> {
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

    fn create_shape_proto(dims: &[i64]) -> proto::TensorShapeProto {
        proto::TensorShapeProto {
            dim: dims
                .iter()
                .map(|&d| proto::tensor_shape_proto::Dimension {
                    value: Some(proto::tensor_shape_proto::dimension::Value::DimValue(d)),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn create_value_info(name: &str, dims: &[i64], dtype: i32) -> proto::ValueInfoProto {
        proto::ValueInfoProto {
            name: name.to_string(),
            r#type: Some(proto::TypeProto {
                value: Some(proto::type_proto::Value::TensorType(
                    proto::type_proto::Tensor {
                        elem_type: dtype,
                        shape: Some(create_shape_proto(dims)),
                    },
                )),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn test_simple_graph_builds() {
        // Create a simple graph: Input -> ReLU -> Output
        let mut model = proto::ModelProto {
            graph: Some(proto::GraphProto {
                node: vec![],
                input: vec![],
                output: vec![],
                initializer: vec![],
                ..Default::default()
            }),
            ..Default::default()
        };

        let graph_proto = model.graph.as_mut().unwrap();

        // Add input
        graph_proto
            .input
            .push(create_value_info("input", &[1, 10], 1));

        // Add ReLU node
        graph_proto.node.push(proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["relu_out".to_string()],
            op_type: "Relu".to_string(),
            ..Default::default()
        });

        // Add output
        graph_proto
            .output
            .push(create_value_info("relu_out", &[1, 10], 1));

        // Should build successfully
        let result = build_graph(&model);
        assert!(result.is_ok());

        let op_graph = result.unwrap();
        assert_eq!(op_graph.nodes.len(), 3); // Input, ReLU, Output
        assert_eq!(op_graph.inputs.len(), 1);
        assert_eq!(op_graph.outputs.len(), 1);
    }
}

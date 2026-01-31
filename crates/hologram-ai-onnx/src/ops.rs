//! ONNX operation translation to hologram OpKind.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use hologram::compiler::{DType, OpKind, OperationGraph};

use crate::proto;

/// Translate a single ONNX node to hologram OpKind with shape inference.
///
/// Returns (OpKind, output_shape, output_dtype)
pub fn translate_node(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let op_type = node.op_type.as_str();

    match op_type {
        // Activations
        "Relu" => translate_relu(node, value_to_node, graph),
        "Sigmoid" => translate_sigmoid(node, value_to_node, graph),
        "Tanh" => translate_tanh(node, value_to_node, graph),
        "Gelu" => translate_gelu(node, value_to_node, graph),
        "Silu" => translate_silu(node, value_to_node, graph),
        "Softmax" => translate_softmax(node, value_to_node, graph),

        // Element-wise arithmetic
        "Add" => translate_add(node, value_to_node, graph),
        "Sub" => translate_sub(node, value_to_node, graph),
        "Mul" => translate_mul(node, value_to_node, graph),
        "Div" => translate_div(node, value_to_node, graph),

        // Linear algebra
        "MatMul" | "Gemm" => translate_matmul(node, value_to_node, graph),

        // Reduction operations
        "ReduceSum" => translate_reduce_sum(node, value_to_node, graph),
        "ReduceMean" => translate_reduce_mean(node, value_to_node, graph),
        "ReduceMax" => translate_reduce_max(node, value_to_node, graph),
        "ReduceMin" => translate_reduce_min(node, value_to_node, graph),

        // Shape manipulation
        "Reshape" => translate_reshape(node, value_to_node, graph),
        "Transpose" => translate_transpose(node, value_to_node, graph),
        "Concat" => translate_concat(node, value_to_node, graph),
        "Gather" => translate_gather(node, value_to_node, graph),
        "Unsqueeze" => translate_unsqueeze(node, value_to_node, graph),
        "Squeeze" => translate_squeeze(node, value_to_node, graph),
        "Slice" => translate_slice(node, value_to_node, graph),
        "Cast" => translate_cast(node, value_to_node, graph),

        _ => bail!("Unsupported ONNX operation: {}", op_type),
    }
}

fn translate_relu(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Relu has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Relu, input_node.shape.clone(), input_node.dtype))
}

fn translate_add(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Add has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Add, input_node.shape.clone(), input_node.dtype))
}

fn translate_matmul(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_a = node.input.first().context("MatMul has no inputs")?;
    let input_b = node.input.get(1).context("MatMul missing second input")?;

    let a_id = value_to_node
        .get(input_a)
        .context("MatMul input A not found")?;
    let b_id = value_to_node
        .get(input_b)
        .context("MatMul input B not found")?;

    let a_shape = &graph.nodes[*a_id as usize].shape;
    let b_shape = &graph.nodes[*b_id as usize].shape;
    let dtype = graph.nodes[*a_id as usize].dtype;

    // For 2D matmul: A[m,k] × B[k,n] = C[m,n]
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let op = OpKind::MatMul { m, k, n };
        let output_shape = vec![m, n];

        Ok((op, output_shape, dtype))
    } else {
        bail!("Unsupported MatMul shape: {:?} × {:?}", a_shape, b_shape)
    }
}

fn translate_softmax(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Softmax has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Softmax, input_node.shape.clone(), input_node.dtype))
}

fn translate_sigmoid(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Sigmoid has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Sigmoid, input_node.shape.clone(), input_node.dtype))
}

fn translate_tanh(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Tanh has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Tanh, input_node.shape.clone(), input_node.dtype))
}

fn translate_gelu(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Gelu has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Gelu, input_node.shape.clone(), input_node.dtype))
}

fn translate_silu(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Silu has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Silu, input_node.shape.clone(), input_node.dtype))
}

fn translate_sub(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Sub has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Sub, input_node.shape.clone(), input_node.dtype))
}

fn translate_mul(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Mul has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Mul, input_node.shape.clone(), input_node.dtype))
}

fn translate_div(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Div has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Div, input_node.shape.clone(), input_node.dtype))
}

// Reduction operations
fn translate_reduce_sum(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("ReduceSum has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // For now, assume reduction produces a scalar
    // TODO: Handle axes and keepdims attributes for partial reduction
    Ok((OpKind::Sum, vec![1], input_node.dtype))
}

fn translate_reduce_mean(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("ReduceMean has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Mean, vec![1], input_node.dtype))
}

fn translate_reduce_max(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("ReduceMax has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Max, vec![1], input_node.dtype))
}

fn translate_reduce_min(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("ReduceMin has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    Ok((OpKind::Min, vec![1], input_node.dtype))
}

// Shape manipulation operations
fn translate_reshape(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Reshape has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Get shape from second input (constant)
    let shape_input = node.input.get(1).context("Reshape missing shape input")?;
    let shape_id = value_to_node
        .get(shape_input)
        .context("Shape input not found")?;
    let shape_node = &graph.nodes[*shape_id as usize];

    // Extract shape from constant
    if !matches!(shape_node.op, OpKind::Constant) {
        bail!("Reshape shape must be a constant");
    }

    // For now, use a placeholder shape - proper implementation needs constant folding
    let target_shape = input_node.shape.clone(); // TODO: Extract from constant

    Ok((
        OpKind::Reshape {
            shape: target_shape.clone(),
        },
        target_shape,
        input_node.dtype,
    ))
}

fn translate_transpose(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Transpose has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract perm attribute
    let perm = get_ints_attr(node, "perm")?;
    let perm_usize: Vec<usize> = perm.iter().map(|&x| x as usize).collect();

    // Compute output shape
    let mut output_shape = vec![0; input_node.shape.len()];
    for (i, &p) in perm_usize.iter().enumerate() {
        output_shape[i] = input_node.shape[p];
    }

    Ok((
        OpKind::Transpose { perm: perm_usize },
        output_shape,
        input_node.dtype,
    ))
}

fn translate_concat(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let axis = get_int_attr(node, "axis")? as usize;
    let num_inputs = node.input.len();

    // Get first input for shape inference
    let first_input = node.input.first().context("Concat has no inputs")?;
    let first_id = value_to_node.get(first_input).context("Input not found")?;
    let first_node = &graph.nodes[*first_id as usize];

    // Compute output shape
    let mut output_shape = first_node.shape.clone();
    for input_name in &node.input[1..] {
        let input_id = value_to_node.get(input_name).context("Input not found")?;
        let input_node = &graph.nodes[*input_id as usize];
        output_shape[axis] += input_node.shape[axis];
    }

    Ok((
        OpKind::Concat { axis, num_inputs },
        output_shape,
        first_node.dtype,
    ))
}

fn translate_gather(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let axis = get_int_attr(node, "axis").unwrap_or(0) as usize;

    let input_name = node.input.first().context("Gather has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Get indices input for shape
    let indices_name = node.input.get(1).context("Gather missing indices")?;
    let indices_id = value_to_node
        .get(indices_name)
        .context("Indices not found")?;
    let indices_node = &graph.nodes[*indices_id as usize];

    // Output shape combines input and indices shapes
    let mut output_shape = input_node.shape.clone();
    output_shape[axis] = indices_node.shape.iter().product();

    Ok((OpKind::Gather { axis }, output_shape, input_node.dtype))
}

fn translate_unsqueeze(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Unsqueeze has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    let axes = get_ints_attr(node, "axes")?;
    // Hologram only supports single axis, use first one
    let axis = axes.first().copied().context("No axes provided")? as usize;

    // Compute output shape
    let mut output_shape = input_node.shape.clone();
    output_shape.insert(axis, 1);

    Ok((OpKind::Unsqueeze { axis }, output_shape, input_node.dtype))
}

fn translate_squeeze(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Squeeze has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    let axes = get_ints_attr(node, "axes").unwrap_or_default();
    // Hologram only supports single axis, use first one
    let axis = axes.first().copied().unwrap_or(0) as usize;

    // Compute output shape
    let mut output_shape = input_node.shape.clone();
    output_shape.remove(axis);

    Ok((OpKind::Squeeze { axis }, output_shape, input_node.dtype))
}

fn translate_slice(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Slice has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract slice parameters from inputs (ONNX 11+ style)
    // starts, ends, axes, steps are inputs, not attributes
    // For simplicity, use placeholder values
    let starts = vec![0];
    let ends = vec![1];

    Ok((
        OpKind::Slice { starts, ends },
        input_node.shape.clone(),
        input_node.dtype,
    ))
}

fn translate_cast(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Cast has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract target dtype from attribute
    let to_onnx = get_int_attr(node, "to")?;
    let to_dtype = crate::dtypes::from_onnx(to_onnx as i32)?;

    Ok((
        OpKind::Cast { to: to_dtype },
        input_node.shape.clone(),
        to_dtype,
    ))
}

// Helper functions for extracting ONNX attributes
fn get_int_attr(node: &proto::NodeProto, name: &str) -> Result<i64> {
    for attr in &node.attribute {
        if attr.name == name {
            return Ok(attr.i);
        }
    }
    bail!("Attribute '{}' not found", name)
}

fn get_ints_attr(node: &proto::NodeProto, name: &str) -> Result<Vec<i64>> {
    for attr in &node.attribute {
        if attr.name == name {
            return Ok(attr.ints.clone());
        }
    }
    bail!("Attribute '{}' not found", name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::compiler::OpNode;

    #[test]
    fn test_matmul_translation() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create input nodes
        let a_node = OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32);
        let b_node = OpNode::new(1, OpKind::Input, vec![3, 4], DType::F32);

        graph.nodes.push(a_node);
        graph.nodes.push(b_node);

        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        // Create MatMul node
        let matmul_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "MatMul".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) =
            translate_node(&matmul_proto, &value_to_node, &graph).unwrap();

        match op_kind {
            OpKind::MatMul { m, k, n } => {
                assert_eq!(m, 2);
                assert_eq!(k, 3);
                assert_eq!(n, 4);
            }
            _ => panic!("Expected MatMul op"),
        }

        assert_eq!(shape, vec![2, 4]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_unsupported_operation() {
        let graph = OperationGraph::default();
        let value_to_node = HashMap::new();

        let node = proto::NodeProto {
            op_type: "UnsupportedOp".to_string(),
            ..Default::default()
        };

        let result = translate_node(&node, &value_to_node, &graph);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unsupported ONNX operation")
        );
    }

    // Helper function to create a test graph with a single input node
    fn create_test_graph_with_input(
        shape: Vec<usize>,
        dtype: DType,
    ) -> (OperationGraph, HashMap<String, u32>) {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let input_node = OpNode::new(0, OpKind::Input, shape, dtype);
        graph.nodes.push(input_node);
        value_to_node.insert("input".to_string(), 0);

        (graph, value_to_node)
    }

    #[test]
    fn test_relu_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 3], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Relu".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Relu));
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_sigmoid_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![4, 5], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Sigmoid".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Sigmoid));
        assert_eq!(shape, vec![4, 5]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_tanh_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![3, 3], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Tanh".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Tanh));
        assert_eq!(shape, vec![3, 3]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_gelu_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 768], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Gelu".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Gelu));
        assert_eq!(shape, vec![1, 768]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_silu_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 512], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Silu".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Silu));
        assert_eq!(shape, vec![2, 512]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_add_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![10, 20], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Add));
        assert_eq!(shape, vec![10, 20]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_sub_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![8, 16], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Sub".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Sub));
        assert_eq!(shape, vec![8, 16]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_mul_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![5, 5], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Mul".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Mul));
        assert_eq!(shape, vec![5, 5]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_div_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![3, 7], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Div".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Div));
        assert_eq!(shape, vec![3, 7]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_softmax_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 1000], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Softmax".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Softmax));
        assert_eq!(shape, vec![1, 1000]);
        assert_eq!(dtype, DType::F32);
    }

    // Reduction operation tests
    #[test]
    fn test_reduce_sum_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 3, 4], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "ReduceSum".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Sum));
        assert_eq!(shape, vec![1]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_reduce_mean_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![4, 8], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "ReduceMean".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Mean));
        assert_eq!(shape, vec![1]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_reduce_max_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![3, 5, 7], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "ReduceMax".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Max));
        assert_eq!(shape, vec![1]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_reduce_min_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![6, 6], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "ReduceMin".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Min));
        assert_eq!(shape, vec![1]);
        assert_eq!(dtype, DType::F32);
    }

    // Shape manipulation tests
    #[test]
    fn test_transpose_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 3, 4], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Transpose".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "perm".to_string(),
                ints: vec![0, 2, 1],
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Transpose { .. }));
        assert_eq!(shape, vec![2, 4, 3]); // Transposed shape
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_concat_translation() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create two input nodes
        let input1 = OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32);
        let input2 = OpNode::new(1, OpKind::Input, vec![2, 3], DType::F32);
        graph.nodes.push(input1);
        graph.nodes.push(input2);
        value_to_node.insert("input1".to_string(), 0);
        value_to_node.insert("input2".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input1".to_string(), "input2".to_string()],
            output: vec!["output".to_string()],
            op_type: "Concat".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axis".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        if let OpKind::Concat { axis, num_inputs } = op_kind {
            assert_eq!(axis, 1);
            assert_eq!(num_inputs, 2);
        } else {
            panic!("Expected Concat op");
        }
        assert_eq!(shape, vec![2, 6]); // Concatenated along axis 1
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_unsqueeze_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 3], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Unsqueeze".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![0], // Single axis
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, _shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        if let OpKind::Unsqueeze { axis } = op_kind {
            assert_eq!(axis, 0);
        } else {
            panic!("Expected Unsqueeze op");
        }
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_squeeze_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 2, 1, 3], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Squeeze".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axes".to_string(),
                ints: vec![0], // Single axis
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, _shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        if let OpKind::Squeeze { axis } = op_kind {
            assert_eq!(axis, 0);
        } else {
            panic!("Expected Squeeze op");
        }
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_cast_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![3, 4], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Cast".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "to".to_string(),
                i: 6, // INT32
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        if let OpKind::Cast { to } = op_kind {
            assert_eq!(to, DType::I32);
        } else {
            panic!("Expected Cast op");
        }
        assert_eq!(shape, vec![3, 4]);
        assert_eq!(dtype, DType::I32);
    }

    #[test]
    fn test_reshape_translation() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create input node
        let input = OpNode::new(0, OpKind::Input, vec![2, 6], DType::F32);
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        // Create constant shape node
        let shape_const = OpNode::new(1, OpKind::Constant, vec![3], DType::I64);
        graph.nodes.push(shape_const);
        value_to_node.insert("shape".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "shape".to_string()],
            output: vec!["output".to_string()],
            op_type: "Reshape".to_string(),
            ..Default::default()
        };

        let (op_kind, _shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Reshape { .. }));
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_gather_translation() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create input node (embedding table)
        let input = OpNode::new(0, OpKind::Input, vec![1000, 128], DType::F32);
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        // Create indices node
        let indices = OpNode::new(1, OpKind::Input, vec![10], DType::I64);
        graph.nodes.push(indices);
        value_to_node.insert("indices".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "indices".to_string()],
            output: vec!["output".to_string()],
            op_type: "Gather".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, _shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        if let OpKind::Gather { axis } = op_kind {
            assert_eq!(axis, 0);
        } else {
            panic!("Expected Gather op");
        }
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_slice_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![10, 20], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Slice".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Slice { .. }));
        assert_eq!(shape, vec![10, 20]);
        assert_eq!(dtype, DType::F32);
    }
}

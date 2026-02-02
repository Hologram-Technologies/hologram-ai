//! ONNX operation translation to hologram OpKind.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use hologram::compiler::shape::broadcast_shapes;
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

        // Convolutional operations
        "Conv" => translate_conv(node, value_to_node, graph),
        "BatchNormalization" => translate_batchnorm(node, value_to_node, graph),
        "MaxPool" => translate_maxpool(node, value_to_node, graph),
        "GlobalAveragePool" => translate_global_avg_pool(node, value_to_node, graph),

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
        "Flatten" => translate_flatten(node, value_to_node, graph),

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
    // Fetch BOTH inputs
    let input_a_name = node.input.first().context("Add has no first input")?;
    let input_b_name = node.input.get(1).context("Add has no second input")?;

    let a_id = value_to_node
        .get(input_a_name)
        .context("First input not found")?;
    let b_id = value_to_node
        .get(input_b_name)
        .context("Second input not found")?;

    let a_node = &graph.nodes[*a_id as usize];
    let b_node = &graph.nodes[*b_id as usize];

    // Compute broadcasted output shape using hologram's shape module
    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)
        .context("Incompatible shapes for Add operation")?;

    Ok((OpKind::Add, output_shape, a_node.dtype))
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
    // Note: transB is handled at the builder level by inserting Transpose nodes
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
    // Fetch BOTH inputs
    let input_a_name = node.input.first().context("Sub has no first input")?;
    let input_b_name = node.input.get(1).context("Sub has no second input")?;

    let a_id = value_to_node
        .get(input_a_name)
        .context("First input not found")?;
    let b_id = value_to_node
        .get(input_b_name)
        .context("Second input not found")?;

    let a_node = &graph.nodes[*a_id as usize];
    let b_node = &graph.nodes[*b_id as usize];

    // Compute broadcasted output shape using hologram's shape module
    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)
        .context("Incompatible shapes for Sub operation")?;

    Ok((OpKind::Sub, output_shape, a_node.dtype))
}

fn translate_mul(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    // Fetch BOTH inputs
    let input_a_name = node.input.first().context("Mul has no first input")?;
    let input_b_name = node.input.get(1).context("Mul has no second input")?;

    let a_id = value_to_node
        .get(input_a_name)
        .context("First input not found")?;
    let b_id = value_to_node
        .get(input_b_name)
        .context("Second input not found")?;

    let a_node = &graph.nodes[*a_id as usize];
    let b_node = &graph.nodes[*b_id as usize];

    // Compute broadcasted output shape using hologram's shape module
    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)
        .context("Incompatible shapes for Mul operation")?;

    Ok((OpKind::Mul, output_shape, a_node.dtype))
}

fn translate_div(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    // Fetch BOTH inputs
    let input_a_name = node.input.first().context("Div has no first input")?;
    let input_b_name = node.input.get(1).context("Div has no second input")?;

    let a_id = value_to_node
        .get(input_a_name)
        .context("First input not found")?;
    let b_id = value_to_node
        .get(input_b_name)
        .context("Second input not found")?;

    let a_node = &graph.nodes[*a_id as usize];
    let b_node = &graph.nodes[*b_id as usize];

    // Compute broadcasted output shape using hologram's shape module
    let output_shape = broadcast_shapes(&a_node.shape, &b_node.shape)
        .context("Incompatible shapes for Div operation")?;

    Ok((OpKind::Div, output_shape, a_node.dtype))
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

fn get_float_attr(node: &proto::NodeProto, name: &str) -> Result<f32> {
    for attr in &node.attribute {
        if attr.name == name {
            return Ok(attr.f);
        }
    }
    bail!("Attribute '{}' not found", name)
}

// CNN operations

fn translate_conv(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Conv has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Get weight input for output channels
    let weight_name = node.input.get(1).context("Conv missing weight input")?;
    let weight_id = value_to_node.get(weight_name).context("Weight not found")?;
    let weight_node = &graph.nodes[*weight_id as usize];

    // Extract ONNX attributes
    let kernel_shape = get_ints_attr(node, "kernel_shape")?;
    let strides = get_ints_attr(node, "strides").unwrap_or_else(|_| vec![1, 1]);
    let pads = get_ints_attr(node, "pads").unwrap_or_else(|_| vec![0, 0, 0, 0]);
    let dilations = get_ints_attr(node, "dilations").unwrap_or_else(|_| vec![1, 1]);
    let group = get_int_attr(node, "group").unwrap_or(1) as usize;

    // For 2D convolution: assume NCHW format
    // Input: [N, C_in, H, W]
    // Weight: [C_out, C_in/group, K_h, K_w]
    // Output: [N, C_out, H_out, W_out]

    let n = input_node.shape[0];
    let h = input_node.shape[2];
    let w = input_node.shape[3];
    let c_out = weight_node.shape[0];

    let kernel_h = kernel_shape[0] as usize;
    let kernel_w = kernel_shape[1] as usize;
    let stride_h = strides[0] as usize;
    let stride_w = strides[1] as usize;
    let pad_h = pads[0] as usize; // top padding
    let pad_w = pads[1] as usize; // left padding
    let dilation_h = dilations[0] as usize;
    let dilation_w = dilations[1] as usize;

    // Calculate output spatial dimensions
    let h_out = (h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    let w_out = (w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    let output_shape = vec![n, c_out, h_out, w_out];

    let op = OpKind::Conv2d {
        kernel: (kernel_h, kernel_w),
        stride: (stride_h, stride_w),
        padding: (pad_h, pad_w),
        dilation: (dilation_h, dilation_w),
        groups: group,
    };

    Ok((op, output_shape, input_node.dtype))
}

fn translate_batchnorm(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node
        .input
        .first()
        .context("BatchNormalization has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract epsilon attribute (default 1e-5)
    let epsilon = get_float_attr(node, "epsilon").unwrap_or(1e-5);

    // BatchNorm output has the same shape as input
    Ok((
        OpKind::BatchNormalization { epsilon },
        input_node.shape.clone(),
        input_node.dtype,
    ))
}

fn translate_maxpool(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("MaxPool has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract attributes
    let kernel_shape = get_ints_attr(node, "kernel_shape")?;
    let strides = get_ints_attr(node, "strides").unwrap_or_else(|_| vec![1, 1]);
    let pads = get_ints_attr(node, "pads").unwrap_or_else(|_| vec![0, 0, 0, 0]);

    // Input: [N, C, H, W]
    let n = input_node.shape[0];
    let c = input_node.shape[1];
    let h = input_node.shape[2];
    let w = input_node.shape[3];

    let kernel_h = kernel_shape[0] as usize;
    let kernel_w = kernel_shape[1] as usize;
    let stride_h = strides[0] as usize;
    let stride_w = strides[1] as usize;
    let pad_h = pads[0] as usize;
    let pad_w = pads[1] as usize;

    // Calculate output spatial dimensions
    let h_out = (h + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kernel_w) / stride_w + 1;

    let output_shape = vec![n, c, h_out, w_out];

    let op = OpKind::MaxPool {
        kernel: (kernel_h, kernel_w),
        stride: (stride_h, stride_w),
        padding: (pad_h, pad_w),
    };

    Ok((op, output_shape, input_node.dtype))
}

fn translate_global_avg_pool(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node
        .input
        .first()
        .context("GlobalAveragePool has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // GlobalAveragePool: Input [N, C, H, W] -> Output [N, C, 1, 1]
    let n = input_node.shape[0];
    let c = input_node.shape[1];
    let output_shape = vec![n, c, 1, 1];

    Ok((OpKind::GlobalAveragePool, output_shape, input_node.dtype))
}

fn translate_flatten(
    node: &proto::NodeProto,
    value_to_node: &HashMap<String, u32>,
    graph: &OperationGraph,
) -> Result<(OpKind, Vec<usize>, DType)> {
    let input_name = node.input.first().context("Flatten has no input")?;
    let input_id = value_to_node.get(input_name).context("Input not found")?;
    let input_node = &graph.nodes[*input_id as usize];

    // Extract axis attribute (default 1)
    let axis = get_int_attr(node, "axis").unwrap_or(1) as usize;

    // Calculate flattened dimensions
    // Everything before axis stays separate, everything from axis onwards is flattened
    let dim0: usize = input_node.shape[..axis].iter().product();
    let dim1: usize = input_node.shape[axis..].iter().product();

    let output_shape = vec![dim0, dim1];

    Ok((
        OpKind::Flatten { start_dim: axis },
        output_shape,
        input_node.dtype,
    ))
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
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create two inputs with same shape
        let input_a = OpNode::new(0, OpKind::Input, vec![10, 20], DType::F32);
        let input_b = OpNode::new(1, OpKind::Input, vec![10, 20], DType::F32);
        graph.nodes.push(input_a);
        graph.nodes.push(input_b);
        value_to_node.insert("input_a".to_string(), 0);
        value_to_node.insert("input_b".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input_a".to_string(), "input_b".to_string()],
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
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create two inputs with same shape
        let input_a = OpNode::new(0, OpKind::Input, vec![8, 16], DType::F32);
        let input_b = OpNode::new(1, OpKind::Input, vec![8, 16], DType::F32);
        graph.nodes.push(input_a);
        graph.nodes.push(input_b);
        value_to_node.insert("input_a".to_string(), 0);
        value_to_node.insert("input_b".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input_a".to_string(), "input_b".to_string()],
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
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create two inputs with same shape
        let input_a = OpNode::new(0, OpKind::Input, vec![5, 5], DType::F32);
        let input_b = OpNode::new(1, OpKind::Input, vec![5, 5], DType::F32);
        graph.nodes.push(input_a);
        graph.nodes.push(input_b);
        value_to_node.insert("input_a".to_string(), 0);
        value_to_node.insert("input_b".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input_a".to_string(), "input_b".to_string()],
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
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Create two inputs with same shape
        let input_a = OpNode::new(0, OpKind::Input, vec![3, 7], DType::F32);
        let input_b = OpNode::new(1, OpKind::Input, vec![3, 7], DType::F32);
        graph.nodes.push(input_a);
        graph.nodes.push(input_b);
        value_to_node.insert("input_a".to_string(), 0);
        value_to_node.insert("input_b".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input_a".to_string(), "input_b".to_string()],
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

    // CNN operation tests

    #[test]
    fn test_conv_translation() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Input: [1, 3, 224, 224] (NCHW format - batch, channels, height, width)
        let input = OpNode::new(0, OpKind::Input, vec![1, 3, 224, 224], DType::F32);
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        // Weight: [64, 3, 7, 7] (out_channels, in_channels, kernel_h, kernel_w)
        let weight = OpNode::new(1, OpKind::Constant, vec![64, 3, 7, 7], DType::F32);
        graph.nodes.push(weight);
        value_to_node.insert("weight".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            op_type: "Conv".to_string(),
            attribute: vec![
                proto::AttributeProto {
                    name: "kernel_shape".to_string(),
                    ints: vec![7, 7],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "strides".to_string(),
                    ints: vec![2, 2],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "pads".to_string(),
                    ints: vec![3, 3, 3, 3], // top, left, bottom, right
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "dilations".to_string(),
                    ints: vec![1, 1],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "group".to_string(),
                    i: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::Conv2d {
            kernel,
            stride,
            padding,
            dilation,
            groups,
        } = op_kind
        {
            assert_eq!(kernel, (7, 7));
            assert_eq!(stride, (2, 2));
            assert_eq!(padding, (3, 3));
            assert_eq!(dilation, (1, 1));
            assert_eq!(groups, 1);
        } else {
            panic!("Expected Conv2d op");
        }

        // Output shape: [1, 64, 112, 112]
        // H_out = (224 + 2*3 - 1*(7-1) - 1) / 2 + 1 = (224 + 6 - 6 - 1) / 2 + 1 = 223/2 + 1 = 111 + 1 = 112
        assert_eq!(shape, vec![1, 64, 112, 112]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_conv_translation_no_padding() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // Input: [1, 16, 32, 32]
        let input = OpNode::new(0, OpKind::Input, vec![1, 16, 32, 32], DType::F32);
        graph.nodes.push(input);
        value_to_node.insert("input".to_string(), 0);

        // Weight: [32, 16, 3, 3]
        let weight = OpNode::new(1, OpKind::Constant, vec![32, 16, 3, 3], DType::F32);
        graph.nodes.push(weight);
        value_to_node.insert("weight".to_string(), 1);

        let node = proto::NodeProto {
            input: vec!["input".to_string(), "weight".to_string()],
            output: vec!["output".to_string()],
            op_type: "Conv".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "kernel_shape".to_string(),
                ints: vec![3, 3],
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::Conv2d {
            kernel,
            stride,
            padding,
            dilation,
            groups,
        } = op_kind
        {
            assert_eq!(kernel, (3, 3));
            assert_eq!(stride, (1, 1)); // Default stride
            assert_eq!(padding, (0, 0)); // Default padding
            assert_eq!(dilation, (1, 1)); // Default dilation
            assert_eq!(groups, 1); // Default group
        } else {
            panic!("Expected Conv2d op");
        }

        // Output shape: [1, 32, 30, 30]
        // H_out = (32 + 0 - 1*(3-1) - 1) / 1 + 1 = (32 - 2 - 1) / 1 + 1 = 30
        assert_eq!(shape, vec![1, 32, 30, 30]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_batchnorm_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 64, 56, 56], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "BatchNormalization".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "epsilon".to_string(),
                f: 1e-5,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::BatchNormalization { epsilon } = op_kind {
            assert!((epsilon - 1e-5).abs() < 1e-10);
        } else {
            panic!("Expected BatchNormalization op");
        }

        // Output shape same as input
        assert_eq!(shape, vec![1, 64, 56, 56]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_batchnorm_translation_default_epsilon() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 128, 28, 28], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "BatchNormalization".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::BatchNormalization { epsilon } = op_kind {
            assert!((epsilon - 1e-5).abs() < 1e-10); // Default epsilon
        } else {
            panic!("Expected BatchNormalization op");
        }

        assert_eq!(shape, vec![2, 128, 28, 28]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_maxpool_translation() {
        let (graph, value_to_node) =
            create_test_graph_with_input(vec![1, 64, 112, 112], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "MaxPool".to_string(),
            attribute: vec![
                proto::AttributeProto {
                    name: "kernel_shape".to_string(),
                    ints: vec![3, 3],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "strides".to_string(),
                    ints: vec![2, 2],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "pads".to_string(),
                    ints: vec![1, 1, 1, 1],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::MaxPool {
            kernel,
            stride,
            padding,
        } = op_kind
        {
            assert_eq!(kernel, (3, 3));
            assert_eq!(stride, (2, 2));
            assert_eq!(padding, (1, 1));
        } else {
            panic!("Expected MaxPool op");
        }

        // Output shape: [1, 64, 56, 56]
        // H_out = (112 + 2*1 - 3) / 2 + 1 = (114 - 3) / 2 + 1 = 111/2 + 1 = 55 + 1 = 56
        assert_eq!(shape, vec![1, 64, 56, 56]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_maxpool_translation_no_padding() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 128, 28, 28], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "MaxPool".to_string(),
            attribute: vec![
                proto::AttributeProto {
                    name: "kernel_shape".to_string(),
                    ints: vec![2, 2],
                    ..Default::default()
                },
                proto::AttributeProto {
                    name: "strides".to_string(),
                    ints: vec![2, 2],
                    ..Default::default()
                },
            ],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::MaxPool {
            kernel,
            stride,
            padding,
        } = op_kind
        {
            assert_eq!(kernel, (2, 2));
            assert_eq!(stride, (2, 2));
            assert_eq!(padding, (0, 0)); // Default padding
        } else {
            panic!("Expected MaxPool op");
        }

        // Output shape: [1, 128, 14, 14]
        // H_out = (28 + 0 - 2) / 2 + 1 = 26/2 + 1 = 13 + 1 = 14
        assert_eq!(shape, vec![1, 128, 14, 14]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_global_avg_pool_translation() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 512, 7, 7], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "GlobalAveragePool".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::GlobalAveragePool));

        // Output shape: [1, 512, 1, 1]
        assert_eq!(shape, vec![1, 512, 1, 1]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_global_avg_pool_translation_large() {
        let (graph, value_to_node) =
            create_test_graph_with_input(vec![4, 2048, 14, 14], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "GlobalAveragePool".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::GlobalAveragePool));

        // Output shape: [4, 2048, 1, 1]
        assert_eq!(shape, vec![4, 2048, 1, 1]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_flatten_translation_default_axis() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![1, 512, 1, 1], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Flatten".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axis".to_string(),
                i: 1,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::Flatten { start_dim } = op_kind {
            assert_eq!(start_dim, 1);
        } else {
            panic!("Expected Flatten op");
        }

        // Output shape: [1, 512] (flatten from axis 1 onwards)
        assert_eq!(shape, vec![1, 512]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_flatten_translation_axis_0() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 3, 4, 5], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Flatten".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axis".to_string(),
                i: 0,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::Flatten { start_dim } = op_kind {
            assert_eq!(start_dim, 0);
        } else {
            panic!("Expected Flatten op");
        }

        // Output shape: [1, 120] (flatten everything)
        assert_eq!(shape, vec![1, 120]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_flatten_translation_axis_2() {
        let (graph, value_to_node) = create_test_graph_with_input(vec![2, 64, 7, 7], DType::F32);

        let node = proto::NodeProto {
            input: vec!["input".to_string()],
            output: vec!["output".to_string()],
            op_type: "Flatten".to_string(),
            attribute: vec![proto::AttributeProto {
                name: "axis".to_string(),
                i: 2,
                ..Default::default()
            }],
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&node, &value_to_node, &graph).unwrap();

        if let OpKind::Flatten { start_dim } = op_kind {
            assert_eq!(start_dim, 2);
        } else {
            panic!("Expected Flatten op");
        }

        // Output shape: [128, 49] (keep first 2 dims, flatten rest)
        // 2 * 64 = 128, 7 * 7 = 49
        assert_eq!(shape, vec![128, 49]);
        assert_eq!(dtype, DType::F32);
    }

    // Binary operation broadcasting tests

    #[test]
    fn test_add_with_broadcasting() {
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        // ResNet18 pattern: [1, 1000] + [1000] → [1, 1000]
        let a_node = OpNode::new(0, OpKind::Input, vec![1, 1000], DType::F32);
        let b_node = OpNode::new(1, OpKind::Constant, vec![1000], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let add_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, dtype) = translate_node(&add_proto, &value_to_node, &graph).unwrap();

        assert!(matches!(op_kind, OpKind::Add));
        assert_eq!(shape, vec![1, 1000]);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_add_same_shapes() {
        // Test exact shape match: [2, 3] + [2, 3] → [2, 3]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a_node = OpNode::new(0, OpKind::Input, vec![2, 3], DType::F32);
        let b_node = OpNode::new(1, OpKind::Input, vec![2, 3], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let add_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let (_, shape, _) = translate_node(&add_proto, &value_to_node, &graph).unwrap();
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_mul_with_broadcasting() {
        // Test Mul with multidim broadcast: [3, 1, 5] * [1, 4, 1] → [3, 4, 5]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a_node = OpNode::new(0, OpKind::Input, vec![3, 1, 5], DType::F32);
        let b_node = OpNode::new(1, OpKind::Input, vec![1, 4, 1], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let mul_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Mul".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, _) = translate_node(&mul_proto, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Mul));
        assert_eq!(shape, vec![3, 4, 5]);
    }

    #[test]
    fn test_add_incompatible_shapes() {
        // Test error for incompatible shapes: [3, 4] + [2, 1] → Error
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a_node = OpNode::new(0, OpKind::Input, vec![3, 4], DType::F32);
        let b_node = OpNode::new(1, OpKind::Input, vec![2, 1], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let add_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Add".to_string(),
            ..Default::default()
        };

        let result = translate_node(&add_proto, &value_to_node, &graph);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_with_broadcasting() {
        // Test Sub: [32, 512] - [512] → [32, 512]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a_node = OpNode::new(0, OpKind::Input, vec![32, 512], DType::F32);
        let b_node = OpNode::new(1, OpKind::Constant, vec![512], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let sub_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Sub".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, _) = translate_node(&sub_proto, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Sub));
        assert_eq!(shape, vec![32, 512]);
    }

    #[test]
    fn test_div_with_broadcasting() {
        // Test Div: [1, 64, 64] / [64, 1] → [1, 64, 64]
        let mut graph = OperationGraph::default();
        let mut value_to_node = HashMap::new();

        let a_node = OpNode::new(0, OpKind::Input, vec![1, 64, 64], DType::F32);
        let b_node = OpNode::new(1, OpKind::Input, vec![64, 1], DType::F32);
        graph.nodes.push(a_node);
        graph.nodes.push(b_node);
        value_to_node.insert("A".to_string(), 0);
        value_to_node.insert("B".to_string(), 1);

        let div_proto = proto::NodeProto {
            input: vec!["A".to_string(), "B".to_string()],
            output: vec!["C".to_string()],
            op_type: "Div".to_string(),
            ..Default::default()
        };

        let (op_kind, shape, _) = translate_node(&div_proto, &value_to_node, &graph).unwrap();
        assert!(matches!(op_kind, OpKind::Div));
        assert_eq!(shape, vec![1, 64, 64]);
    }
}

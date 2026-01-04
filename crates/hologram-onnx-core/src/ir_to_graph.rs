//! Convert hologram-compiler IR to OperationGraph for scheduling and execution.
//!
//! This module provides the bridge between the ONNX translation pipeline (which produces
//! `IRFunction`) and the hologram execution system (which consumes `OperationGraph`).
//!
//! # Pipeline
//!
//! ```text
//! IRFunction (from ONNX translation)
//!     ↓ ir_to_operation_graph()
//! OperationGraph (hologram-compiler)
//!     ↓ Compiler::compile_graph_parallel()
//! ParallelSchedule
//!     ↓ serialize
//! .holo file
//! ```

use std::collections::HashMap;
use std::path::Path;

use hologram_compiler::expr::OpKind;
use hologram_compiler::graph::{GraphBuilder, NodeId as GraphNodeId, OperationGraph, WeightRef};
use hologram_compiler::ir::{BinOp, ConstValue, IRFunction, IRNode, NodeId as IRNodeId, ReduceOp, Type, UnOp};

use crate::{OnnxError, Result};

/// Default threshold for streaming weights to external file (256KB in elements).
/// Tensors larger than this will be written to the weights file.
pub const DEFAULT_WEIGHT_THRESHOLD_ELEMENTS: usize = 64 * 1024; // 64K floats = 256KB

/// Options for converting IR to OperationGraph.
#[derive(Debug, Clone, Default)]
pub struct ConversionOptions {
    /// Threshold for streaming weights to external file (in elements).
    pub weight_threshold_elements: usize,
    /// Enable Resize upscaling. When false, Resize ops pass through without scaling.
    pub enable_resize_upscaling: bool,
}

impl ConversionOptions {
    /// Create default options with upscaling enabled.
    pub fn new() -> Self {
        Self {
            weight_threshold_elements: DEFAULT_WEIGHT_THRESHOLD_ELEMENTS,
            enable_resize_upscaling: true,
        }
    }

    /// Create options with upscaling disabled (for memory-constrained systems).
    pub fn without_resize_upscaling() -> Self {
        Self {
            weight_threshold_elements: DEFAULT_WEIGHT_THRESHOLD_ELEMENTS,
            enable_resize_upscaling: false,
        }
    }
}

/// Convert an IRFunction to a hologram-compiler OperationGraph.
///
/// This function walks the IR nodes in topological order (as stored in the IR body)
/// and constructs an equivalent OperationGraph using GraphBuilder.
///
/// # Arguments
///
/// * `ir_func` - The IR function to convert
///
/// # Returns
///
/// An OperationGraph ready for scheduling and execution via `Compiler::compile_graph_parallel()`.
pub fn ir_to_operation_graph(ir_func: &IRFunction) -> Result<OperationGraph> {
    let mut builder = GraphBuilder::new();
    let mut id_map: HashMap<IRNodeId, GraphNodeId> = HashMap::new();
    let default_options = ConversionOptions::new();

    // Process nodes in order (already topologically sorted)
    for entry in &ir_func.body {
        let graph_id = convert_node(&mut builder, &entry.node, &id_map, &default_options)?;
        id_map.insert(entry.id, graph_id);
    }

    // Set outputs
    for (idx, output_id) in ir_func.outputs.iter().enumerate() {
        if let Some(&graph_id) = id_map.get(output_id) {
            let output_name = if ir_func.outputs.len() == 1 {
                "output".to_string()
            } else {
                format!("output_{}", idx)
            };
            builder.set_output(&output_name, graph_id);
        }
    }

    Ok(builder.build())
}

/// Convert an IRFunction to a hologram-compiler OperationGraph with streaming weights.
///
/// This variant streams large tensors to an external weights file instead of
/// storing them inline in the graph, significantly reducing memory usage for
/// large models.
///
/// # Arguments
///
/// * `ir_func` - The IR function to convert
/// * `weights_path` - Path to write the external weights file
/// * `threshold_elements` - Tensors with more elements than this are streamed to file
///
/// # Returns
///
/// An OperationGraph ready for scheduling and execution via `Compiler::compile_graph_parallel()`.
/// Large weights are stored in the file at `weights_path`.
pub fn ir_to_operation_graph_streaming(
    ir_func: &IRFunction,
    weights_path: impl AsRef<Path>,
    threshold_elements: usize,
) -> Result<OperationGraph> {
    let options = ConversionOptions {
        weight_threshold_elements: threshold_elements,
        enable_resize_upscaling: true,
    };
    ir_to_operation_graph_streaming_with_options(ir_func, weights_path, options)
}

/// Convert an IRFunction to a hologram-compiler OperationGraph with streaming weights and options.
///
/// This variant allows fine-grained control over conversion behavior.
///
/// # Arguments
///
/// * `ir_func` - The IR function to convert
/// * `weights_path` - Path to write the external weights file
/// * `options` - Conversion options (weight threshold, resize upscaling, etc.)
///
/// # Returns
///
/// An OperationGraph ready for scheduling and execution via `Compiler::compile_graph_parallel()`.
/// Large weights are stored in the file at `weights_path`.
pub fn ir_to_operation_graph_streaming_with_options(
    ir_func: &IRFunction,
    weights_path: impl AsRef<Path>,
    options: ConversionOptions,
) -> Result<OperationGraph> {
    let mut builder = GraphBuilder::with_weights_file(weights_path)
        .map_err(|e| OnnxError::InvalidModel(format!("Failed to create weights file: {}", e)))?;

    let mut id_map: HashMap<IRNodeId, GraphNodeId> = HashMap::new();

    // Process nodes in order (already topologically sorted)
    for entry in &ir_func.body {
        let graph_id = convert_node_streaming(&mut builder, &entry.node, &id_map, &options)?;
        id_map.insert(entry.id, graph_id);
    }

    // Set outputs
    for (idx, output_id) in ir_func.outputs.iter().enumerate() {
        if let Some(&graph_id) = id_map.get(output_id) {
            let output_name = if ir_func.outputs.len() == 1 {
                "output".to_string()
            } else {
                format!("output_{}", idx)
            };
            builder.set_output(&output_name, graph_id);
        }
    }

    // Log streaming stats
    let bytes_written = builder.weights_bytes_written();
    if bytes_written > 0 {
        tracing::info!(
            "Streamed {} bytes ({:.2} MB) of weights to external file",
            bytes_written,
            bytes_written as f64 / (1024.0 * 1024.0)
        );
    }

    Ok(builder.build())
}

/// Convert a single IR node to a graph node (streaming version).
fn convert_node_streaming(
    builder: &mut GraphBuilder,
    node: &IRNode,
    id_map: &HashMap<IRNodeId, GraphNodeId>,
    options: &ConversionOptions,
) -> Result<GraphNodeId> {
    match node {
        IRNode::Input { name, ty } => {
            // Extract shape from type if available and concrete
            let shape = extract_concrete_shape(ty);
            if shape.is_empty() {
                tracing::warn!("Adding input '{}' WITHOUT shape (this may cause buffer allocation issues)", name);
                Ok(builder.add_input(name))
            } else {
                tracing::info!("Adding input '{}' WITH shape {:?}", name, shape);
                Ok(builder.add_input_with_shape(name, shape))
            }
        }

        IRNode::Constant { value, .. } => {
            convert_constant_streaming(builder, value, options.weight_threshold_elements)
        }

        // All other nodes are handled the same as non-streaming
        _ => convert_node(builder, node, id_map, options),
    }
}

/// Extract concrete shape dimensions from an IR type.
/// Returns empty vector if the type has no shape or has symbolic dimensions.
fn extract_concrete_shape(ty: &Type) -> Vec<usize> {
    if let Some(shape) = ty.shape() {
        let dims: Vec<usize> = shape
            .dims()
            .iter()
            .filter_map(|d| d.as_concrete())
            .collect();
        tracing::debug!(
            "extract_concrete_shape: type={:?}, shape_dims={}, extracted_dims={:?}",
            ty,
            shape.dims().len(),
            dims
        );
        dims
    } else {
        tracing::debug!("extract_concrete_shape: type={:?}, no shape", ty);
        vec![]
    }
}

/// Convert an IR constant to a graph node with streaming support.
fn convert_constant_streaming(
    builder: &mut GraphBuilder,
    value: &ConstValue,
    threshold_elements: usize,
) -> Result<GraphNodeId> {
    match value {
        ConstValue::F32(v) => Ok(builder.add_constant(*v)),
        ConstValue::F64(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::I32(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::I64(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::Bool(v) => Ok(builder.add_constant(if *v { 1.0 } else { 0.0 })),
        ConstValue::Tensor { shape, data } => {
            // Convert bytes to f32 (assuming little-endian f32)
            let floats: Vec<f32> = data
                .chunks(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect();

            // Use streaming for large tensors
            builder.add_tensor_streaming(floats, shape.clone(), threshold_elements)
                .map_err(|e| OnnxError::InvalidModel(format!("Failed to stream weight: {}", e)))
        }
    }
}

/// Convert a single IR node to a graph node.
fn convert_node(
    builder: &mut GraphBuilder,
    node: &IRNode,
    id_map: &HashMap<IRNodeId, GraphNodeId>,
    options: &ConversionOptions,
) -> Result<GraphNodeId> {
    match node {
        IRNode::Input { name, ty } => {
            // Extract shape from type if available and concrete
            let shape = extract_concrete_shape(ty);
            if shape.is_empty() {
                Ok(builder.add_input(name))
            } else {
                Ok(builder.add_input_with_shape(name, shape))
            }
        }

        IRNode::Constant { value, .. } => {
            convert_constant(builder, value)
        }

        IRNode::WeightRef { offset, size, ty, .. } => {
            // Extract shape from type if available
            let shape = ty.shape().map(|s| {
                s.dims().iter().filter_map(|d| {
                    match d {
                        crate::Dim::Concrete(n) => Some(*n),
                        crate::Dim::Var(_) => None, // Dynamic dims not supported for weights
                        crate::Dim::Expr(_) => None, // Expressions not supported for weights
                    }
                }).collect::<Vec<_>>()
            }).unwrap_or_default();

            let weight_ref = WeightRef {
                offset: *offset,
                length: *size,
                shape: shape.clone(),
            };
            Ok(builder.add_weight_ref(weight_ref, shape))
        }

        IRNode::BinaryOp { op, lhs, rhs } => {
            let lhs_id = lookup_id(id_map, *lhs)?;
            let rhs_id = lookup_id(id_map, *rhs)?;
            let op_kind = binop_to_opkind(*op);
            Ok(builder.add_op(op_kind, vec![lhs_id, rhs_id]))
        }

        IRNode::UnaryOp { op, operand } => {
            let operand_id = lookup_id(id_map, *operand)?;
            let op_kind = unop_to_opkind(*op);
            Ok(builder.add_op(op_kind, vec![operand_id]))
        }

        IRNode::MatMul { lhs, rhs } => {
            let lhs_id = lookup_id(id_map, *lhs)?;
            let rhs_id = lookup_id(id_map, *rhs)?;
            Ok(builder.add_op(OpKind::MatMul, vec![lhs_id, rhs_id]))
        }

        IRNode::Softmax { input, axis } => {
            let input_id = lookup_id(id_map, *input)?;
            Ok(builder.add_op_with_attr(OpKind::Softmax, vec![input_id], vec![*axis as i64]))
        }

        IRNode::Reshape { input, shape } => {
            let input_id = lookup_id(id_map, *input)?;
            // Convert shape to i64 attributes
            let shape_attrs: Vec<i64> = shape.dims().iter().map(|d| {
                match d {
                    crate::Dim::Concrete(n) => *n as i64,
                    crate::Dim::Var(_) => -1, // Dynamic dimension
                    crate::Dim::Expr(_) => -1, // Expression dimension
                }
            }).collect();
            Ok(builder.add_op_with_attr(OpKind::Reshape, vec![input_id], shape_attrs))
        }

        IRNode::Transpose { input, perm } => {
            let input_id = lookup_id(id_map, *input)?;
            if let Some(perm) = perm {
                let perm_attrs: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
                Ok(builder.add_op_with_attr(OpKind::Transpose, vec![input_id], perm_attrs))
            } else {
                // Default transpose (reverse dimensions)
                Ok(builder.add_op(OpKind::Transpose, vec![input_id]))
            }
        }

        IRNode::Broadcast { input, .. } => {
            // For now, treat broadcast as identity (actual broadcasting happens during execution)
            let input_id = lookup_id(id_map, *input)?;
            Ok(builder.add_op(OpKind::Identity, vec![input_id]))
        }

        IRNode::Slice { input, ranges } => {
            let input_id = lookup_id(id_map, *input)?;
            // Pack slice ranges as attributes: [start0, end0, step0, start1, end1, step1, ...]
            let mut attrs = Vec::new();
            for range in ranges {
                attrs.push(range.start.unwrap_or(0) as i64);
                attrs.push(range.end.unwrap_or(i64::MAX as isize) as i64);
                attrs.push(range.step.unwrap_or(1) as i64);
            }
            Ok(builder.add_op_with_attr(OpKind::Slice, vec![input_id], attrs))
        }

        IRNode::Gather { data, indices, axis } => {
            let data_id = lookup_id(id_map, *data)?;
            let indices_id = lookup_id(id_map, *indices)?;
            Ok(builder.add_op_with_attr(OpKind::Gather, vec![data_id, indices_id], vec![*axis as i64]))
        }

        IRNode::Concat { inputs, axis } => {
            let input_ids: Result<Vec<_>> = inputs.iter()
                .map(|id| lookup_id(id_map, *id))
                .collect();
            Ok(builder.add_op_with_attr(OpKind::Concat, input_ids?, vec![*axis as i64]))
        }

        IRNode::Stack { inputs, axis } => {
            let input_ids: Result<Vec<_>> = inputs.iter()
                .map(|id| lookup_id(id_map, *id))
                .collect();
            Ok(builder.add_op_with_attr(OpKind::Stack, input_ids?, vec![*axis as i64]))
        }

        IRNode::VStack { inputs } => {
            let input_ids: Result<Vec<_>> = inputs.iter()
                .map(|id| lookup_id(id_map, *id))
                .collect();
            Ok(builder.add_op(OpKind::VStack, input_ids?))
        }

        IRNode::HStack { inputs } => {
            let input_ids: Result<Vec<_>> = inputs.iter()
                .map(|id| lookup_id(id_map, *id))
                .collect();
            Ok(builder.add_op(OpKind::HStack, input_ids?))
        }

        IRNode::Reduce { op, input, axes, keepdims } => {
            let input_id = lookup_id(id_map, *input)?;
            let op_kind = reduceop_to_opkind(*op);
            // Pack: [keepdims, axis0, axis1, ...]
            let mut attrs = vec![if *keepdims { 1 } else { 0 }];
            attrs.extend(axes.iter().map(|&a| a as i64));
            Ok(builder.add_op_with_attr(op_kind, vec![input_id], attrs))
        }

        IRNode::Select { cond, true_val, false_val } => {
            let cond_id = lookup_id(id_map, *cond)?;
            let true_id = lookup_id(id_map, *true_val)?;
            let false_id = lookup_id(id_map, *false_val)?;
            Ok(builder.add_op(OpKind::Where, vec![cond_id, true_id, false_id]))
        }

        IRNode::Conv2D { input, kernel, bias, stride, padding, dilation, groups } => {
            let input_id = lookup_id(id_map, *input)?;
            let kernel_id = lookup_id(id_map, *kernel)?;

            let mut inputs = vec![input_id, kernel_id];
            if let Some(bias_id) = bias {
                inputs.push(lookup_id(id_map, *bias_id)?);
            }

            // Pack Conv2D attributes: [group, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w]
            let attrs = vec![
                *groups as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
                dilation.0 as i64,
                dilation.1 as i64,
            ];

            Ok(builder.add_op_with_attr(OpKind::Conv, inputs, attrs))
        }

        IRNode::MaxPool { input, kernel_size, stride, padding } => {
            let input_id = lookup_id(id_map, *input)?;
            // Pack: [kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w]
            let attrs = vec![
                kernel_size.0 as i64,
                kernel_size.1 as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
            ];
            Ok(builder.add_op_with_attr(OpKind::ReduceMax, vec![input_id], attrs))
        }

        IRNode::AvgPool { input, kernel_size, stride, padding } => {
            let input_id = lookup_id(id_map, *input)?;
            // Pack: [kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w]
            let attrs = vec![
                kernel_size.0 as i64,
                kernel_size.1 as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
            ];
            Ok(builder.add_op_with_attr(OpKind::ReduceMean, vec![input_id], attrs))
        }

        IRNode::BatchNorm { input, scale, bias, mean, var, epsilon } => {
            let input_id = lookup_id(id_map, *input)?;
            let scale_id = lookup_id(id_map, *scale)?;
            let bias_id = lookup_id(id_map, *bias)?;
            let mean_id = lookup_id(id_map, *mean)?;
            let var_id = lookup_id(id_map, *var)?;

            // BatchNorm decomposition: (x - mean) / sqrt(var + eps) * scale + bias
            let epsilon_const = builder.add_constant(*epsilon);

            // x - mean
            let centered = builder.add_op(OpKind::Sub, vec![input_id, mean_id]);

            // var + epsilon
            let var_eps = builder.add_op(OpKind::Add, vec![var_id, epsilon_const]);

            // sqrt(var + epsilon)
            let std = builder.add_op(OpKind::Sqrt, vec![var_eps]);

            // (x - mean) / std
            let normalized = builder.add_op(OpKind::Div, vec![centered, std]);

            // normalized * scale
            let scaled = builder.add_op(OpKind::Mul, vec![normalized, scale_id]);

            // scaled + bias
            Ok(builder.add_op(OpKind::Add, vec![scaled, bias_id]))
        }

        IRNode::Cast { input, .. } => {
            // For now, treat cast as identity (type conversion handled elsewhere)
            let input_id = lookup_id(id_map, *input)?;
            Ok(builder.add_op(OpKind::Identity, vec![input_id]))
        }

        IRNode::Im2Col { input, kernel_size, stride, padding, dilation } => {
            let input_id = lookup_id(id_map, *input)?;
            // Pack Im2Col attributes
            let attrs = vec![
                kernel_size.0 as i64,
                kernel_size.1 as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
                dilation.0 as i64,
                dilation.1 as i64,
            ];
            // Im2Col followed by MatMul is the decomposed Conv2D
            // For now, use Reshape as a placeholder operation
            Ok(builder.add_op_with_attr(OpKind::Reshape, vec![input_id], attrs))
        }

        IRNode::Col2Im { input, output_size, kernel_size, stride, padding, dilation } => {
            let input_id = lookup_id(id_map, *input)?;
            let attrs = vec![
                output_size.0 as i64,
                output_size.1 as i64,
                kernel_size.0 as i64,
                kernel_size.1 as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
                dilation.0 as i64,
                dilation.1 as i64,
            ];
            Ok(builder.add_op_with_attr(OpKind::Reshape, vec![input_id], attrs))
        }

        IRNode::Unfold { input, kernel_size, stride, padding } => {
            let input_id = lookup_id(id_map, *input)?;
            let attrs = vec![
                kernel_size.0 as i64,
                kernel_size.1 as i64,
                stride.0 as i64,
                stride.1 as i64,
                padding.0 as i64,
                padding.1 as i64,
            ];
            Ok(builder.add_op_with_attr(OpKind::Reshape, vec![input_id], attrs))
        }

        IRNode::Phi { .. } => {
            Err(OnnxError::InvalidModel("Phi nodes not supported in direct execution".into()))
        }

        IRNode::Call { func, args } => {
            // Handle common ONNX operations that weren't lowered to IR primitives
            handle_call_node(builder, func, args, id_map, options)
        }
    }
}

/// Handle IRNode::Call for common ONNX operations.
///
/// When the ONNX translator can't fully lower an operation (e.g., dynamic shapes),
/// it creates a Call node. We map to available OpKind variants here.
fn handle_call_node(
    builder: &mut GraphBuilder,
    func: &str,
    args: &[IRNodeId],
    id_map: &HashMap<IRNodeId, GraphNodeId>,
    options: &ConversionOptions,
) -> Result<GraphNodeId> {
    let input_ids: Result<Vec<_>> = args.iter().map(|id| lookup_id(id_map, *id)).collect();
    let input_ids = input_ids?;

    match func {
        // Operations with direct OpKind mappings
        "onnx.Reshape" => {
            if input_ids.len() >= 2 {
                Ok(builder.add_op(OpKind::Reshape, vec![input_ids[0], input_ids[1]]))
            } else if input_ids.len() == 1 {
                Ok(builder.add_op(OpKind::Identity, vec![input_ids[0]]))
            } else {
                Err(OnnxError::InvalidModel("Reshape requires at least 1 input".into()))
            }
        }

        "onnx.Concat" => Ok(builder.add_op(OpKind::Concat, input_ids)),

        "onnx.Gather" => {
            if input_ids.len() >= 2 {
                Ok(builder.add_op(OpKind::Gather, input_ids))
            } else {
                Err(OnnxError::InvalidModel("Gather requires 2 inputs".into()))
            }
        }

        "onnx.Slice" => {
            if !input_ids.is_empty() {
                Ok(builder.add_op(OpKind::Slice, input_ids))
            } else {
                Err(OnnxError::InvalidModel("Slice requires at least 1 input".into()))
            }
        }

        "onnx.Resize" => {
            if input_ids.is_empty() {
                return Err(OnnxError::InvalidModel("Resize requires at least 1 input".into()));
            }

            tracing::debug!("Resize: {} inputs, enable_upscaling={}", input_ids.len(), options.enable_resize_upscaling);

            // If upscaling is disabled, use identity scales (1x for all dims)
            if !options.enable_resize_upscaling {
                tracing::info!("Resize: upscaling disabled, using identity scales");
                let identity_scales = vec![1000i64, 1000, 1000, 1000];
                return Ok(builder.add_op_with_attr(OpKind::Resize, vec![input_ids[0]], identity_scales));
            }

            // ONNX Resize inputs: X, roi, scales, sizes
            // Try to extract scales/sizes from constant tensor inputs
            let mut scale_attrs: Vec<i64> = Vec::new();

            // Check all inputs after the first one for scales/sizes constants
            for (idx, &input_id) in input_ids.iter().enumerate().skip(1) {
                if let Some((data, shape)) = builder.get_constant_tensor(input_id) {
                    tracing::debug!("Resize: input[{}] id={} is constant data={:?} shape={:?}",
                                   idx, input_id, data, shape);

                    // Check if this looks like scales (floats around 1-4) or sizes (large integers)
                    if !data.is_empty() && data.iter().any(|&v| v != 0.0) {
                        // If values are small (0.5-8), treat as scales
                        // If values are large (>8), treat as sizes
                        let max_val = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

                        if max_val <= 8.0 {
                            // Scales: store as 1000x fixed point
                            scale_attrs = data.iter().map(|&v| (v * 1000.0) as i64).collect();
                            tracing::info!("Resize: using scales {:?} from input[{}]", data, idx);
                            break;
                        } else {
                            // Sizes: store as negative values
                            scale_attrs = data.iter().map(|&v| -(v as i64)).collect();
                            tracing::info!("Resize: using sizes {:?} from input[{}]", data, idx);
                            break;
                        }
                    }
                }
            }

            if !scale_attrs.is_empty() {
                Ok(builder.add_op_with_attr(OpKind::Resize, vec![input_ids[0]], scale_attrs))
            } else {
                // Fallback: default 2x upscale (common in VAE decoders)
                tracing::warn!("Resize: no scales/sizes found, using default 2x upscale");
                // [1000, 1000, 2000, 2000] = 1x, 1x, 2x, 2x scale
                let default_scales = vec![1000i64, 1000, 2000, 2000];
                Ok(builder.add_op_with_attr(OpKind::Resize, vec![input_ids[0]], default_scales))
            }
        }

        "onnx.Conv" => {
            if input_ids.len() >= 2 {
                Ok(builder.add_op(OpKind::Conv, input_ids))
            } else {
                Err(OnnxError::InvalidModel("Conv requires at least 2 inputs".into()))
            }
        }

        "onnx.ConvTranspose" => {
            if input_ids.len() >= 2 {
                Ok(builder.add_op(OpKind::ConvTranspose, input_ids))
            } else {
                Err(OnnxError::InvalidModel("ConvTranspose requires at least 2 inputs".into()))
            }
        }

        "onnx.Transpose" => {
            if !input_ids.is_empty() {
                Ok(builder.add_op(OpKind::Transpose, input_ids))
            } else {
                Err(OnnxError::InvalidModel("Transpose requires 1 input".into()))
            }
        }

        "onnx.MatMul" => {
            if input_ids.len() >= 2 {
                Ok(builder.add_op(OpKind::MatMul, input_ids))
            } else {
                Err(OnnxError::InvalidModel("MatMul requires 2 inputs".into()))
            }
        }

        "onnx.Softmax" => {
            if !input_ids.is_empty() {
                Ok(builder.add_op(OpKind::Softmax, input_ids))
            } else {
                Err(OnnxError::InvalidModel("Softmax requires 1 input".into()))
            }
        }

        // Shape manipulation ops mapped to Reshape (semantically similar)
        "onnx.Squeeze" | "onnx.Unsqueeze" | "onnx.Flatten" | "onnx.Expand" => {
            if !input_ids.is_empty() {
                tracing::debug!("Mapping {} to Reshape", func);
                Ok(builder.add_op(OpKind::Reshape, input_ids))
            } else {
                Err(OnnxError::InvalidModel(format!("{} requires at least 1 input", func)))
            }
        }

        // Operations that need runtime support - pass through first input
        "onnx.Split" | "onnx.Pad" | "onnx.Tile" | "onnx.Shape" | "onnx.Size"
        | "onnx.ConstantOfShape" | "onnx.NonZero" | "onnx.ScatterND" | "onnx.GatherND" => {
            if !input_ids.is_empty() {
                tracing::warn!("{} not fully supported, using pass-through", func);
                Ok(builder.add_op(OpKind::Identity, vec![input_ids[0]]))
            } else {
                Err(OnnxError::InvalidModel(format!("{} requires at least 1 input", func)))
            }
        }

        _ => {
            // For unknown operations, use Identity as pass-through
            if !input_ids.is_empty() {
                tracing::warn!("Unknown call '{}', using identity pass-through", func);
                Ok(builder.add_op(OpKind::Identity, vec![input_ids[0]]))
            } else {
                Err(OnnxError::InvalidModel(format!(
                    "Function call '{}' not supported and has no inputs to pass through",
                    func
                )))
            }
        }
    }
}

/// Convert an IR constant to a graph node.
fn convert_constant(builder: &mut GraphBuilder, value: &ConstValue) -> Result<GraphNodeId> {
    match value {
        ConstValue::F32(v) => Ok(builder.add_constant(*v)),
        ConstValue::F64(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::I32(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::I64(v) => Ok(builder.add_constant(*v as f32)),
        ConstValue::Bool(v) => Ok(builder.add_constant(if *v { 1.0 } else { 0.0 })),
        ConstValue::Tensor { shape, data } => {
            // Convert bytes to f32 (assuming little-endian f32)
            let floats: Vec<f32> = data
                .chunks(4)
                .map(|chunk| {
                    let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                    f32::from_le_bytes(arr)
                })
                .collect();
            Ok(builder.add_constant_tensor(floats, shape.clone()))
        }
    }
}

/// Look up a graph node ID from an IR node ID.
fn lookup_id(id_map: &HashMap<IRNodeId, GraphNodeId>, ir_id: IRNodeId) -> Result<GraphNodeId> {
    id_map.get(&ir_id).copied().ok_or_else(|| {
        OnnxError::InvalidModel(format!("IR node {:?} not found in id_map", ir_id))
    })
}

/// Convert IR binary operation to OpKind.
fn binop_to_opkind(op: BinOp) -> OpKind {
    match op {
        BinOp::Add => OpKind::Add,
        BinOp::Sub => OpKind::Sub,
        BinOp::Mul => OpKind::Mul,
        BinOp::Div => OpKind::Div,
        BinOp::Pow => OpKind::Pow,
        BinOp::Mod => OpKind::Div, // Fallback - no direct Mod in OpKind
        BinOp::Min => OpKind::Minimum,
        BinOp::Max => OpKind::Maximum,
        BinOp::Eq => OpKind::Equal,
        BinOp::Ne => OpKind::NotEqual,
        BinOp::Lt => OpKind::Less,
        BinOp::Le => OpKind::LessEqual,
        BinOp::Gt => OpKind::Greater,
        BinOp::Ge => OpKind::GreaterEqual,
        BinOp::And => OpKind::LogicalAnd,
        BinOp::Or => OpKind::LogicalOr,
    }
}

/// Convert IR unary operation to OpKind.
fn unop_to_opkind(op: UnOp) -> OpKind {
    match op {
        UnOp::Neg => OpKind::Neg,
        UnOp::Abs => OpKind::Abs,
        UnOp::Not => OpKind::LogicalNot,
        UnOp::Sqrt => OpKind::Sqrt,
        UnOp::Rsqrt => OpKind::Rsqrt,
        UnOp::Exp => OpKind::Exp,
        UnOp::Log => OpKind::Log,
        UnOp::Sin => OpKind::Sin,
        UnOp::Cos => OpKind::Cos,
        UnOp::Tan => OpKind::Tan,
        UnOp::Floor => OpKind::Floor,
        UnOp::Ceil => OpKind::Ceil,
        UnOp::Round => OpKind::Round,
        UnOp::Sigmoid => OpKind::Sigmoid,
        UnOp::Tanh => OpKind::Tanh,
        UnOp::ReLU => OpKind::ReLU,
        UnOp::GELU => OpKind::GELU,
        UnOp::Erf => OpKind::Erf,
    }
}

/// Convert IR reduce operation to OpKind.
fn reduceop_to_opkind(op: ReduceOp) -> OpKind {
    match op {
        ReduceOp::Sum => OpKind::ReduceSum,
        ReduceOp::Prod => OpKind::ReduceProd,
        ReduceOp::Mean => OpKind::ReduceMean,
        ReduceOp::Max => OpKind::ReduceMax,
        ReduceOp::Min => OpKind::ReduceMin,
        ReduceOp::ArgMax => OpKind::ArgMax,
        ReduceOp::ArgMin => OpKind::ArgMin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_compiler::ir::{IRBuilder, ScalarType, Type};
    use hologram_compiler::shapes::{Dim, Shape};

    #[test]
    fn test_simple_sigmoid() {
        // Build IR: sigmoid(x)
        let mut builder = IRBuilder::new("test");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Concrete(4)]));
        let x = builder.add_input("x", input_type);
        let sigmoid = builder.sigmoid(x);
        builder.set_output(sigmoid);
        let ir_func = builder.build();

        // Convert to OperationGraph
        let graph = ir_to_operation_graph(&ir_func).unwrap();

        assert_eq!(graph.len(), 2); // input + sigmoid
        assert!(graph.inputs.contains_key("x"));
    }

    #[test]
    fn test_binary_add() {
        // Build IR: x + y
        let mut builder = IRBuilder::new("test");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Concrete(4)]));
        let x = builder.add_input("x", input_type.clone());
        let y = builder.add_input("y", input_type);
        let sum = builder.add(x, y);
        builder.set_output(sum);
        let ir_func = builder.build();

        let graph = ir_to_operation_graph(&ir_func).unwrap();

        assert_eq!(graph.len(), 3); // x, y, add
        assert!(graph.inputs.contains_key("x"));
        assert!(graph.inputs.contains_key("y"));
    }

    #[test]
    fn test_matmul() {
        // Build IR: matmul(x, w)
        let mut builder = IRBuilder::new("test");
        let x_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Concrete(4), Dim::Concrete(8)]));
        let w_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Concrete(8), Dim::Concrete(16)]));
        let x = builder.add_input("x", x_type);
        let w = builder.add_input("w", w_type);
        let matmul = builder.matmul(x, w);
        builder.set_output(matmul);
        let ir_func = builder.build();

        let graph = ir_to_operation_graph(&ir_func).unwrap();

        assert_eq!(graph.len(), 3); // x, w, matmul
    }

    #[test]
    fn test_constant() {
        // Build IR: x * 2.0
        let mut builder = IRBuilder::new("test");
        let input_type = Type::tensor(ScalarType::F32, Shape::new(vec![Dim::Concrete(4)]));
        let x = builder.add_input("x", input_type);
        let two = builder.add_f32(2.0);
        let product = builder.mul(x, two);
        builder.set_output(product);
        let ir_func = builder.build();

        let graph = ir_to_operation_graph(&ir_func).unwrap();

        assert_eq!(graph.len(), 3); // x, constant, mul
    }
}

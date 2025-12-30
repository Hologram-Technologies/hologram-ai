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

use hologram_compiler::expr::OpKind;
use hologram_compiler::graph::{GraphBuilder, NodeId as GraphNodeId, OperationGraph, WeightRef};
use hologram_compiler::ir::{BinOp, ConstValue, IRFunction, IRNode, NodeId as IRNodeId, ReduceOp, UnOp};

use crate::{OnnxError, Result};

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

    // Process nodes in order (already topologically sorted)
    for entry in &ir_func.body {
        let graph_id = convert_node(&mut builder, &entry.node, &id_map)?;
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

/// Convert a single IR node to a graph node.
fn convert_node(
    builder: &mut GraphBuilder,
    node: &IRNode,
    id_map: &HashMap<IRNodeId, GraphNodeId>,
) -> Result<GraphNodeId> {
    match node {
        IRNode::Input { name, .. } => {
            Ok(builder.add_input(name))
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

        IRNode::Call { func, .. } => {
            Err(OnnxError::InvalidModel(format!("Function call '{}' not supported in direct execution", func)))
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

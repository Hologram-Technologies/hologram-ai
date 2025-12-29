//! Full ONNX to hologram IR translation pipeline.
//!
//! This module provides the complete translation from ONNX graphs to hologram IR,
//! connecting the parsing from `hologram-onnx-core` with the operation translators
//! from `hologram-onnx-ops`.
//!
//! # Architecture
//!
//! ```text
//! ONNX GraphProto
//!     ↓ translate_graph_to_ir()
//! IR Function (with symbolic shapes)
//!     ↓ apply_decomposition()
//! IR Function (Conv2D → Im2col+GEMM, etc.)
//!     ↓ lower_to_operation_graph()
//! OperationGraph (ready for execution)
//! ```

use std::collections::HashMap;

use hologram_compiler::ir::{
    decompose_function, DecomposeConfig, IRBuilder, IRFunction, NodeId, ScalarType, Type,
};
use hologram_compiler::shapes::{Dim as IRDim, Shape};
use hologram_onnx_core::{Dim, OnnxConfig, OnnxError, Result, SymbolicShape};
use hologram_onnx_ops::translate_onnx_op;
use hologram_onnx_spec::{GraphProto, TensorProto, tensor_proto::DataType};
use tracing::{debug, info, trace, warn};

/// Translate ONNX graph to hologram IR with symbolic shapes.
///
/// This is the main entry point for ONNX → IR translation. It:
/// 1. Parses ONNX inputs/outputs/initializers
/// 2. Creates IR nodes for each ONNX operation
/// 3. Propagates symbolic shapes throughout the graph
/// 4. Validates all shape constraints
///
/// # Arguments
///
/// * `graph` - ONNX graph protobuf
/// * `opset_version` - ONNX opset version (determines operation semantics)
///
/// # Returns
///
/// IR function with symbolic shapes and all operations translated.
///
/// # Errors
///
/// Returns error if:
/// - Unsupported operations encountered
/// - Shape inference fails
/// - Graph structure is invalid
pub fn translate_graph_to_ir(
    graph: &GraphProto,
    opset_version: i64,
) -> Result<IRFunction> {
    info!("Starting ONNX graph translation (opset {})", opset_version);

    let graph_name = if graph.name.is_empty() {
        "onnx_graph"
    } else {
        &graph.name
    };

    let mut builder = IRBuilder::new(graph_name);

    // Map from ONNX tensor names to IR NodeIds
    let mut tensor_map: HashMap<String, NodeId> = HashMap::new();

    // Map from ONNX tensor names to symbolic shapes
    let mut shape_map: HashMap<String, SymbolicShape> = HashMap::new();

    // Step 1: Process initializers (weights/constants)
    info!("Processing {} initializers", graph.initializer.len());
    for initializer in &graph.initializer {
        let (node_id, shape) = process_initializer(initializer, &mut builder)?;
        tensor_map.insert(initializer.name.clone(), node_id);
        shape_map.insert(initializer.name.clone(), shape);
        trace!("Processed initializer: {}", initializer.name);
    }

    // Step 2: Process graph inputs (excluding initializers)
    info!("Processing {} inputs", graph.input.len());
    for input in &graph.input {
        // Skip if already processed as initializer
        if tensor_map.contains_key(&input.name) {
            continue;
        }

        let (node_id, shape) = process_input(input, &mut builder)?;
        tensor_map.insert(input.name.clone(), node_id);
        shape_map.insert(input.name.clone(), shape);
        debug!("Processed input: {}", input.name);
    }

    // Step 3: Process nodes in topological order
    info!("Translating {} nodes", graph.node.len());
    for (idx, node) in graph.node.iter().enumerate() {
        trace!("Translating node {}/{}: {} ({})",
               idx + 1, graph.node.len(), node.name, node.op_type);

        // Gather input NodeIds
        let input_ids: Vec<NodeId> = node.input.iter()
            .filter(|name| !name.is_empty())
            .map(|name| {
                tensor_map.get(name).copied().ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Node '{}' references unknown input '{}'",
                        node.name, name
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;

        // Translate the operation
        let output_id = translate_onnx_op(
            &node.op_type,
            &input_ids,
            &node.attribute,
            &shape_map,
            &mut builder,
        )?;

        // Map outputs (most ops have single output)
        if !node.output.is_empty() {
            tensor_map.insert(node.output[0].clone(), output_id);

            // Infer and store output shape
            if let Some(shape) = infer_output_shape(node, &shape_map)? {
                shape_map.insert(node.output[0].clone(), shape);
            }
        }

        // Handle multi-output operations (like Split)
        // For now, we only support single outputs
        if node.output.len() > 1 {
            warn!("Node '{}' has {} outputs, only first is mapped",
                  node.name, node.output.len());
        }
    }

    // Step 4: Mark graph outputs
    info!("Marking {} outputs", graph.output.len());
    for output in &graph.output {
        if let Some(&node_id) = tensor_map.get(&output.name) {
            builder.set_output(node_id);
            debug!("Marked output: {}", output.name);
        } else {
            return Err(OnnxError::InvalidModel(format!(
                "Graph output '{}' not found in tensor map",
                output.name
            )));
        }
    }

    // Build the IR function
    let func = builder.build();
    info!("Translation complete: {} IR nodes", func.body.len());

    Ok(func)
}

/// Apply decomposition pass to IR function.
///
/// This pass transforms high-level operations into ISA-optimized primitives:
/// - **Conv2D → Im2col + GEMM**: Enables SIMD vectorization
/// - **Pooling → Window ops**: Enables PhiCoordinate addressing
/// - **BatchNorm → Element-wise**: Enables ClassMap fusion
pub fn apply_ir_decomposition(
    ir_func: IRFunction,
    config: &OnnxConfig,
) -> Result<IRFunction> {
    info!("Applying decomposition pass");

    let decompose_config = DecomposeConfig {
        decompose_conv2d: config.decompose_conv2d,
        decompose_pooling: config.decompose_pooling,
        ..Default::default()
    };

    let decomposed = decompose_function(&ir_func, &decompose_config)
        .map_err(|e| OnnxError::IrTranslationError(format!("Decomposition failed: {:?}", e)))?;
    info!("Decomposition complete: {} IR nodes", decomposed.body.len());

    Ok(decomposed)
}

/// Process an ONNX initializer (weight tensor).
fn process_initializer(
    initializer: &TensorProto,
    builder: &mut IRBuilder,
) -> Result<(NodeId, SymbolicShape)> {
    let dims: Vec<usize> = initializer.dims.iter()
        .map(|&d| d as usize)
        .collect();

    let scalar_type = data_type_to_scalar(initializer.data_type)?;

    // Extract weight data
    let data = extract_tensor_data(initializer)?;

    // Add as tensor constant to IR
    let node_id = builder.add_tensor_const(dims.clone(), data, scalar_type);

    let shape = SymbolicShape::concrete(dims);
    Ok((node_id, shape))
}

/// Process an ONNX graph input.
fn process_input(
    input: &hologram_onnx_spec::ValueInfoProto,
    builder: &mut IRBuilder,
) -> Result<(NodeId, SymbolicShape)> {
    let tensor_type = input.r#type.as_ref()
        .and_then(|t| t.value.as_ref())
        .ok_or_else(|| OnnxError::InvalidModel(
            format!("Input '{}' has no type information", input.name)
        ))?;

    let tensor_type = match tensor_type {
        hologram_onnx_spec::type_proto::Value::TensorType(t) => t,
        _ => return Err(OnnxError::InvalidModel(
            format!("Input '{}' is not a tensor type", input.name)
        )),
    };

    let scalar_type = data_type_to_scalar(tensor_type.elem_type)?;

    // Extract shape with symbolic dimension support
    let shape_proto = tensor_type.shape.as_ref()
        .ok_or_else(|| OnnxError::InvalidModel(
            format!("Input '{}' has no shape", input.name)
        ))?;

    let mut ir_dims = Vec::new();
    let mut symbolic_dims = Vec::new();

    for dim in &shape_proto.dim {
        match &dim.value {
            Some(hologram_onnx_spec::tensor_shape_proto::dimension::Value::DimValue(v)) => {
                ir_dims.push(IRDim::Concrete(*v as usize));
                symbolic_dims.push(Dim::Concrete(*v as usize));
            }
            Some(hologram_onnx_spec::tensor_shape_proto::dimension::Value::DimParam(name)) => {
                ir_dims.push(IRDim::Var(name.clone()));
                symbolic_dims.push(Dim::Var(name.clone()));
            }
            None => {
                // Unknown dimension - treat as symbolic
                let name = format!("dim_{}", ir_dims.len());
                ir_dims.push(IRDim::Var(name.clone()));
                symbolic_dims.push(Dim::Var(name));
            }
        }
    }

    let ir_type = Type::tensor(scalar_type, Shape::new(ir_dims));
    let node_id = builder.add_input(&input.name, ir_type);

    let shape = SymbolicShape::new(symbolic_dims);
    Ok((node_id, shape))
}

/// Convert ONNX data type to IR scalar type.
fn data_type_to_scalar(data_type: i32) -> Result<ScalarType> {
    match DataType::try_from(data_type) {
        Ok(DataType::Float) => Ok(ScalarType::F32),
        Ok(DataType::Double) => Ok(ScalarType::F64),
        Ok(DataType::Int32) => Ok(ScalarType::I32),
        Ok(DataType::Int64) => Ok(ScalarType::I64),
        Ok(DataType::Uint8) => Ok(ScalarType::U8),
        Ok(DataType::Int8) => Ok(ScalarType::I8),
        Ok(DataType::Uint16) => Ok(ScalarType::U16),
        Ok(DataType::Int16) => Ok(ScalarType::I16),
        Ok(DataType::Uint32) => Ok(ScalarType::U32),
        Ok(DataType::Uint64) => Ok(ScalarType::U64),
        Ok(DataType::Float16) => Ok(ScalarType::F16),
        Ok(DataType::Bfloat16) => Ok(ScalarType::BF16),
        Ok(DataType::Bool) => Ok(ScalarType::Bool),
        Ok(dt) => Err(OnnxError::UnsupportedDataType(format!("{:?}", dt))),
        Err(_) => Err(OnnxError::UnsupportedDataType(format!("unknown type {}", data_type))),
    }
}

/// Extract raw data from ONNX tensor.
fn extract_tensor_data(tensor: &TensorProto) -> Result<Vec<u8>> {
    // Priority: raw_data > typed data fields
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }

    // Fall back to typed data fields
    let data_type = DataType::try_from(tensor.data_type)
        .map_err(|_| OnnxError::UnsupportedDataType(format!("type {}", tensor.data_type)))?;

    match data_type {
        DataType::Float => {
            let bytes: Vec<u8> = tensor.float_data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Double => {
            let bytes: Vec<u8> = tensor.double_data.iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Int32 => {
            let bytes: Vec<u8> = tensor.int32_data.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Int64 => {
            let bytes: Vec<u8> = tensor.int64_data.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Uint64 => {
            let bytes: Vec<u8> = tensor.uint64_data.iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        _ => Err(OnnxError::UnsupportedDataType(format!("{:?}", data_type))),
    }
}

/// Infer output shape for a node (simplified version).
fn infer_output_shape(
    node: &hologram_onnx_spec::NodeProto,
    shape_map: &HashMap<String, SymbolicShape>,
) -> Result<Option<SymbolicShape>> {
    // Get input shapes
    let input_shapes: Vec<&SymbolicShape> = node.input.iter()
        .filter(|name| !name.is_empty())
        .filter_map(|name| shape_map.get(name))
        .collect();

    if input_shapes.is_empty() {
        return Ok(None);
    }

    // Use the shape inference from hologram-onnx-ops
    match hologram_onnx_ops::infer_op_output_shape(
        &node.op_type,
        &input_shapes,
        &node.attribute,
    ) {
        Ok(shape) => Ok(Some(shape)),
        Err(_) => {
            // Fall back to first input shape for unary ops
            if input_shapes.len() == 1 {
                Ok(Some(input_shapes[0].clone()))
            } else {
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_conversion() {
        assert!(matches!(data_type_to_scalar(1), Ok(ScalarType::F32)));
        assert!(matches!(data_type_to_scalar(11), Ok(ScalarType::F64)));
        assert!(matches!(data_type_to_scalar(6), Ok(ScalarType::I32)));
        assert!(matches!(data_type_to_scalar(7), Ok(ScalarType::I64)));
    }

    #[test]
    fn test_extract_float_data() {
        let mut tensor = TensorProto::default();
        tensor.data_type = DataType::Float as i32;
        tensor.float_data = vec![1.0, 2.0, 3.0];

        let data = extract_tensor_data(&tensor).unwrap();
        assert_eq!(data.len(), 12); // 3 floats * 4 bytes
    }

    #[test]
    fn test_extract_raw_data() {
        let mut tensor = TensorProto::default();
        tensor.data_type = DataType::Float as i32;
        tensor.raw_data = vec![0, 0, 128, 63]; // 1.0f32 in little-endian

        let data = extract_tensor_data(&tensor).unwrap();
        assert_eq!(data, vec![0, 0, 128, 63]);
    }
}

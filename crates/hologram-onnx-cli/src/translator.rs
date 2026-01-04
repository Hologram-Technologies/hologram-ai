//! Full ONNX to hologram IR translation pipeline for CLI.
//!
//! This module provides the complete translation from ONNX graphs to hologram IR,
//! connecting the parsing from `hologram-onnx-core` with the operation translators
//! from `hologram-onnx-ops`.

use std::collections::HashMap;

use hologram_compiler::ir::{
    DecomposeConfig, IRBuilder, IRFunction, NodeId, ScalarType, decompose_function,
};
use hologram_compiler::shapes::{Dim as IRDim, Shape};
use hologram_onnx_core::{Dim, OnnxConfig, OnnxError, SymbolicShape};
use hologram_onnx_ops::translate_onnx_op;
use hologram_onnx_spec::{GraphProto, TensorProto, tensor_proto::DataType};
use tracing::{debug, info, trace, warn};

/// Constant tensor data extracted from initializers.
/// Maps tensor name to its raw data and shape.
#[derive(Debug, Clone)]
struct ConstantData {
    data: Vec<u8>,
    #[allow(dead_code)]
    dims: Vec<i64>,
    data_type: i32,
}

type Result<T> = std::result::Result<T, OnnxError>;

/// Translate ONNX graph to hologram IR with symbolic shapes.
///
/// # Arguments
/// * `graph` - The ONNX graph to translate
/// * `opset_version` - The ONNX opset version
pub fn translate_graph_to_ir(graph: &GraphProto, opset_version: i64) -> Result<IRFunction> {
    translate_graph_to_ir_with_path(graph, opset_version, None)
}

/// Translate ONNX graph to hologram IR with symbolic shapes and external data support.
///
/// # Arguments
/// * `graph` - The ONNX graph to translate
/// * `opset_version` - The ONNX opset version
/// * `model_path` - Optional path to the ONNX model for resolving external data
pub fn translate_graph_to_ir_with_path(
    graph: &GraphProto,
    opset_version: i64,
    model_path: Option<&std::path::Path>,
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

    // Map from tensor names to constant data (for Reshape shape extraction)
    let mut constant_map: HashMap<String, ConstantData> = HashMap::new();

    // Step 1: Process initializers (weights/constants)
    info!("Processing {} initializers", graph.initializer.len());
    for initializer in &graph.initializer {
        let (node_id, shape) = process_initializer(initializer, &mut builder)?;
        tensor_map.insert(initializer.name.clone(), node_id);
        shape_map.insert(initializer.name.clone(), shape);

        // Store constant data for operations that need it (e.g., Reshape)
        let constant_data = ConstantData {
            data: extract_tensor_data_with_path(initializer, model_path)?,
            dims: initializer.dims.clone(),
            data_type: initializer.data_type,
        };
        constant_map.insert(initializer.name.clone(), constant_data);

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
        trace!(
            "Translating node {}/{}: {} ({})",
            idx + 1,
            graph.node.len(),
            node.name,
            node.op_type
        );

        // Gather input NodeIds
        let input_ids: Vec<NodeId> = node
            .input
            .iter()
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
        // Special handling for operations that need constant tensor values
        let output_id = if node.op_type == "Reshape" && node.input.len() >= 2 {
            // Reshape with shape from second input - try to get constant shape
            // If shape is not constant, fall back to dynamic Call node
            match translate_reshape_with_constants(
                &input_ids,
                &node.input,
                &node.attribute,
                &constant_map,
                &mut builder,
            ) {
                Ok(id) => id,
                Err(_) => {
                    // Fall back to dynamic reshape using Call node
                    debug!("Reshape shape is dynamic, using Call node");
                    builder.call("onnx.Reshape", input_ids.clone())
                }
            }
        } else {
            translate_onnx_op(
                &node.op_type,
                &input_ids,
                &node.attribute,
                &shape_map,
                &mut builder,
            )?
        };

        // Map outputs (most ops have single output)
        if !node.output.is_empty() {
            tensor_map.insert(node.output[0].clone(), output_id);

            // Infer and store output shape
            if let Some(shape) = infer_output_shape(node, &shape_map)? {
                shape_map.insert(node.output[0].clone(), shape);
            }
        }

        // Handle multi-output operations (like Split)
        if node.output.len() > 1 {
            warn!(
                "Node '{}' has {} outputs, only first is mapped",
                node.name,
                node.output.len()
            );
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
pub fn apply_ir_decomposition(ir_func: IRFunction, config: &OnnxConfig) -> Result<IRFunction> {
    info!("Applying decomposition pass");

    let decompose_config = DecomposeConfig {
        decompose_conv2d: config.decompose_conv2d,
        decompose_pooling: config.decompose_pooling,
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
    let dims: Vec<usize> = initializer.dims.iter().map(|&d| d as usize).collect();

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
    let tensor_type = input
        .r#type
        .as_ref()
        .and_then(|t| t.value.as_ref())
        .ok_or_else(|| {
            OnnxError::InvalidModel(format!("Input '{}' has no type information", input.name))
        })?;

    let tensor_type = match tensor_type {
        hologram_onnx_spec::type_proto::Value::TensorType(t) => t,
        _ => {
            return Err(OnnxError::InvalidModel(format!(
                "Input '{}' is not a tensor type",
                input.name
            )));
        }
    };

    let scalar_type = data_type_to_scalar(tensor_type.elem_type)?;

    // Extract shape with symbolic dimension support
    let shape_proto = tensor_type
        .shape
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidModel(format!("Input '{}' has no shape", input.name)))?;

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

    let ir_type = hologram_compiler::ir::Type::tensor(scalar_type, Shape::new(ir_dims));
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
        Err(_) => Err(OnnxError::UnsupportedDataType(format!(
            "unknown type {}",
            data_type
        ))),
    }
}

/// Extract raw data from ONNX tensor, with support for external data files.
///
/// # Arguments
/// * `tensor` - The tensor proto
/// * `model_path` - Optional path to the ONNX model file for resolving external data
fn extract_tensor_data_with_path(tensor: &TensorProto, model_path: Option<&std::path::Path>) -> Result<Vec<u8>> {
    // Check for external data (data_location == 1 means EXTERNAL)
    if tensor.data_location == 1 && !tensor.external_data.is_empty() {
        let mut location: Option<&str> = None;
        let mut offset: u64 = 0;
        let mut length: Option<u64> = None;

        for entry in &tensor.external_data {
            match entry.key.as_str() {
                "location" => location = Some(&entry.value),
                "offset" => offset = entry.value.parse().unwrap_or(0),
                "length" => length = entry.value.parse().ok(),
                _ => {}
            }
        }

        if let (Some(loc), Some(model_path)) = (location, model_path) {
            let external_path = if std::path::Path::new(loc).is_absolute() {
                std::path::PathBuf::from(loc)
            } else {
                model_path.parent().unwrap_or(std::path::Path::new(".")).join(loc)
            };

            let mut file = std::fs::File::open(&external_path).map_err(|e| {
                OnnxError::InvalidModel(format!(
                    "Failed to open external data file '{}': {}",
                    external_path.display(), e
                ))
            })?;

            use std::io::{Read, Seek, SeekFrom};
            file.seek(SeekFrom::Start(offset)).map_err(|e| {
                OnnxError::InvalidModel(format!("Failed to seek: {}", e))
            })?;

            let bytes_to_read = if let Some(len) = length {
                len as usize
            } else {
                let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
                let bytes_per_element = match tensor.data_type {
                    1 => 4,  // FLOAT
                    10 => 2, // FLOAT16
                    11 => 8, // DOUBLE
                    6 => 4,  // INT32
                    7 => 8,  // INT64
                    _ => 4,
                };
                num_elements * bytes_per_element
            };

            let mut raw_data = vec![0u8; bytes_to_read];
            file.read_exact(&mut raw_data).map_err(|e| {
                OnnxError::InvalidModel(format!("Failed to read external data: {}", e))
            })?;

            return Ok(raw_data);
        }
    }

    // Priority: raw_data > typed data fields
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }

    // Fall back to extract_tensor_data
    extract_tensor_data(tensor)
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
            let bytes: Vec<u8> = tensor
                .float_data
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Double => {
            let bytes: Vec<u8> = tensor
                .double_data
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Int32 => {
            let bytes: Vec<u8> = tensor
                .int32_data
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Int64 => {
            let bytes: Vec<u8> = tensor
                .int64_data
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        DataType::Uint64 => {
            let bytes: Vec<u8> = tensor
                .uint64_data
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect();
            Ok(bytes)
        }
        _ => Err(OnnxError::UnsupportedDataType(format!("{:?}", data_type))),
    }
}

/// Translate Reshape operation with constant shape extraction.
///
/// This function handles the ONNX Reshape operation by extracting the target
/// shape from a constant initializer (second input).
fn translate_reshape_with_constants(
    input_ids: &[NodeId],
    input_names: &[String],
    _attrs: &[hologram_onnx_spec::AttributeProto],
    constant_map: &HashMap<String, ConstantData>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if input_ids.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Reshape expects at least 1 input, got 0".to_string(),
        ));
    }

    let data = input_ids[0];

    // Get shape tensor name (second input)
    let shape_name = input_names
        .get(1)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| OnnxError::InvalidModel("Reshape requires shape input".to_string()))?;

    // Look up constant shape data
    let shape_const = constant_map.get(shape_name).ok_or_else(|| {
        OnnxError::IrTranslationError(format!(
            "Reshape shape input '{}' is not a constant - dynamic shapes not supported",
            shape_name
        ))
    })?;

    // Extract shape values from constant
    let target_dims = extract_shape_values(shape_const)?;

    debug!("Reshape with constant shape: {:?}", target_dims);

    // Convert to IR shape
    // Handle -1 (inferred dimension) by using symbolic variable
    let ir_dims: Vec<IRDim> = target_dims
        .iter()
        .enumerate()
        .map(|(i, &dim)| {
            if dim == -1 {
                // Use symbolic dimension for inferred size
                IRDim::Var(format!("reshape_inferred_{}", i))
            } else {
                IRDim::Concrete(dim as usize)
            }
        })
        .collect();

    let target_shape = Shape::new(ir_dims);

    // Create reshape node
    let result = builder.reshape(data, target_shape);
    Ok(result)
}

/// Extract i64 shape values from constant tensor data.
fn extract_shape_values(constant: &ConstantData) -> Result<Vec<i64>> {
    let data_type = DataType::try_from(constant.data_type)
        .map_err(|_| OnnxError::UnsupportedDataType(format!("type {}", constant.data_type)))?;

    match data_type {
        DataType::Int64 => {
            // Read as i64 values
            if !constant.data.len().is_multiple_of(8) {
                return Err(OnnxError::InvalidModel(
                    "Invalid int64 tensor data length".to_string(),
                ));
            }
            let values: Vec<i64> = constant
                .data
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect();
            Ok(values)
        }
        DataType::Int32 => {
            // Read as i32 and convert to i64
            if !constant.data.len().is_multiple_of(4) {
                return Err(OnnxError::InvalidModel(
                    "Invalid int32 tensor data length".to_string(),
                ));
            }
            let values: Vec<i64> = constant
                .data
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()) as i64)
                .collect();
            Ok(values)
        }
        _ => Err(OnnxError::UnsupportedDataType(format!(
            "Shape tensor must be int64 or int32, got {:?}",
            data_type
        ))),
    }
}

/// Infer output shape for a node.
fn infer_output_shape(
    node: &hologram_onnx_spec::NodeProto,
    shape_map: &HashMap<String, SymbolicShape>,
) -> Result<Option<SymbolicShape>> {
    // Get input shapes
    let input_shapes: Vec<&SymbolicShape> = node
        .input
        .iter()
        .filter(|name| !name.is_empty())
        .filter_map(|name| shape_map.get(name))
        .collect();

    if input_shapes.is_empty() {
        return Ok(None);
    }

    // Use the shape inference from hologram-onnx-ops
    match hologram_onnx_ops::infer_op_output_shape(&node.op_type, &input_shapes, &node.attribute) {
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

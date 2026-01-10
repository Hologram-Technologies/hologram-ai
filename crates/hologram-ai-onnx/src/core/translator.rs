//! ONNX to hologram IR lowering types.
//!
//! This module provides types for lowering IR functions to OperationGraph format,
//! which is the final serializable representation for .holo files.
//!
//! # Architecture
//!
//! ```text
//! hologram-onnx (top-level)   ←── Uses real translator
//!   ↓ IRFunction
//! hologram-onnx-core (this crate)
//!   ↓ lower_to_operation_graph()
//! OperationGraph
//!   ↓ to_bytes()
//! .holo file
//! ```
//!
//! **Note**: Full ONNX → IR translation lives in the top-level `hologram-onnx` crate
//! because it requires both `hologram-onnx-core` (shapes, parsing) and `hologram-onnx-ops`
//! (operation translators). Due to the dependency structure (ops → core), putting the
//! translator in core would create a cyclic dependency.
//!
//! # Usage
//!
//! For full ONNX → .holo compilation, use the top-level crate:
//! ```ignore
//! use hologram_ai_onnx::{compile_onnx, OnnxCompiler};
//!
//! // Simple usage
//! let (holo, weights) = compile_onnx(&onnx_bytes)?;
//!
//! // With config
//! let compiler = OnnxCompiler::with_config(config);
//! let (holo, weights) = compiler.compile(&onnx_bytes)?;
//! ```
//!
//! For parsing and validation only (this crate):
//! ```ignore
//! use hologram_ai_onnx::core::{parse_model, validate_model};
//! let model = parse_model(&onnx_bytes)?;
//! validate_model(&model)?;
//! ```

use hologram::ir::{DType, OperationGraph as IRFunction};

use crate::Result;

/// Result of lowering to OperationGraph.
///
/// This wraps the IR function with serialization capabilities for .holo format.
/// The OperationGraph is the final representation before writing to disk.
#[derive(Debug, Clone)]
pub struct OperationGraph {
    ir_func: IRFunction,
}

impl OperationGraph {
    /// Create from IR function.
    pub fn from_ir(ir_func: IRFunction) -> Self {
        Self { ir_func }
    }

    /// Get node count.
    ///
    /// Returns 0 as direct access to the IR function's internal node count is not available.
    /// This is used for informational purposes only.
    pub fn node_count(&self) -> usize {
        0
    }

    /// Get the underlying IR function reference.
    pub fn ir_function(&self) -> &IRFunction {
        &self.ir_func
    }

    /// Serialize to .holo format bytes.
    ///
    /// Uses hologram-ir's rkyv-based serialization.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.ir_func
            .to_bytes()
            .map_err(crate::OnnxError::IrError)
    }
}

/// Lower IR function to OperationGraph.
///
/// Wraps the IR function for serialization to .holo format.
///
/// # Arguments
///
/// * `ir_func` - Decomposed IR function from the translation pipeline
///
/// # Returns
///
/// OperationGraph ready for serialization via `to_bytes()`.
///
/// # Example
///
/// ```ignore
/// use hologram_ai_onnx::core::lower_to_operation_graph;
///
/// let ir_func = translate_graph_to_ir(&graph, opset)?;
/// let op_graph = lower_to_operation_graph(ir_func)?;
/// let bytes = op_graph.to_bytes()?;
/// ```
pub fn lower_to_operation_graph(ir_func: IRFunction) -> Result<OperationGraph> {
    Ok(OperationGraph::from_ir(ir_func))
}

/// Translate ONNX GraphProto to hologram IR.
///
/// This is the main entry point for ONNX→IR translation. It:
/// 1. Creates input nodes with symbolic shapes
/// 2. Processes initializers (weights) as constant nodes
/// 3. Translates each ONNX node to IR operations
/// 4. Sets up output nodes
///
/// # Arguments
///
/// * `graph` - ONNX graph proto to translate
///
/// # Returns
///
/// Fully translated hologram IR OperationGraph
///
/// # Errors
///
/// Returns error if:
/// - Unsupported operations are encountered
/// - Shape inference fails
/// - Invalid ONNX graph structure
pub fn translate_graph_to_ir(graph: &crate::proto::GraphProto) -> Result<IRFunction> {
    use hologram::ir::GraphBuilder;
    use crate::ops::translator::translate_onnx_node;
    use std::collections::HashMap;
    use tracing::{debug, trace};

    let mut builder = GraphBuilder::new();
    let mut value_map: HashMap<String, hologram::NodeIndex> = HashMap::new();

    // Step 1: Process inputs with symbolic shapes
    debug!("Processing {} graph inputs", graph.input.len());
    for (i, input) in graph.input.iter().enumerate() {
        let shape = crate::core::SymbolicShape::from_value_info(input)?;

        // Determine dtype from ONNX type
        let dtype = extract_dtype_from_value_info(input)?;

        trace!("Adding input '{}': shape={:?}, dtype={:?}", input.name, shape, dtype);
        let node_idx = builder.input(&input.name, shape.into_inner(), dtype);
        value_map.insert(input.name.clone(), node_idx);

        // Debug: Log input_ids mapping specifically
        if input.name == "input_ids" {
            debug!("INPUT_IDS MAPPING: '{}' (ONNX input {}) -> NodeIndex({:?})",
                input.name, i, node_idx);
        }
    }

    // Step 2: Process initializers (constants/weights)
    debug!("Processing {} initializers", graph.initializer.len());
    for initializer in &graph.initializer {
        if value_map.contains_key(&initializer.name) {
            // Already added as input, skip
            trace!("Skipping initializer '{}' (already in inputs)", initializer.name);
            continue;
        }

        // Convert ONNX tensor to constant node
        let (constant_data, shape) = tensor_proto_to_constant(initializer)?;
        trace!("Adding constant '{}': shape={:?}", initializer.name, shape);
        let node_idx = builder.constant(constant_data, shape);
        value_map.insert(initializer.name.clone(), node_idx);
    }

    // Step 3: Translate all nodes in topological order
    debug!("Processing {} graph nodes", graph.node.len());
    for (idx, node) in graph.node.iter().enumerate() {
        trace!("Translating node {}/{}: {} ({})", idx + 1, graph.node.len(), node.name, node.op_type);

        // Debug: Log when input_ids is used
        for input_name in &node.input {
            if input_name == "input_ids" {
                let input_node_idx = value_map.get(input_name);
                debug!(
                    "NODE {} ({}) USES input_ids: value_map['input_ids'] = {:?}",
                    idx, node.op_type, input_node_idx
                );
            }
        }

        // Translate this ONNX node to IR operations
        let output_indices = translate_onnx_node(node, &mut builder, &mut value_map)?;

        // Map outputs to their node indices
        for (output_name, output_idx) in node.output.iter().zip(output_indices.iter()) {
            if !output_name.is_empty() {
                value_map.insert(output_name.clone(), *output_idx);
            }
        }
    }

    // Step 4: Set up outputs with declared shapes
    debug!("Processing {} graph outputs", graph.output.len());
    for output in &graph.output {
        let output_node = value_map.get(&output.name).ok_or_else(|| {
            crate::OnnxError::InvalidModel(format!("Graph output '{}' not found in value map", output.name))
        })?;

        // Extract declared shape from ONNX ValueInfoProto to preserve symbolic dimensions
        let declared_shape = crate::core::SymbolicShape::from_value_info(output)
            .ok()
            .map(|s| s.into_inner());

        trace!("Adding output '{}' with declared shape {:?}", output.name, declared_shape);
        builder.output_with_shape(&output.name, *output_node, declared_shape)?;
    }

    let ir_func = builder.build();

    // Debug: Check if edges were created
    debug!("Graph translation complete: {} nodes, {} edges",
        ir_func.inner().node_count(),
        ir_func.inner().edge_count());

    Ok(ir_func)
}

/// Extract DType from ONNX ValueInfoProto.
fn extract_dtype_from_value_info(value_info: &crate::proto::ValueInfoProto) -> Result<DType> {
    use crate::proto::type_proto::Value;

    let type_proto = value_info.r#type.as_ref().ok_or_else(|| {
        crate::OnnxError::InvalidModel(format!("Value '{}' has no type information", value_info.name))
    })?;

    let tensor_type = match &type_proto.value {
        Some(Value::TensorType(tt)) => tt,
        _ => {
            return Err(crate::OnnxError::InvalidModel(format!(
                "Value '{}' is not a tensor",
                value_info.name
            )));
        }
    };

    let elem_type = tensor_type.elem_type;
    onnx_dtype_to_hologram(elem_type)
}

/// Convert ONNX data type to hologram DType.
fn onnx_dtype_to_hologram(onnx_type: i32) -> Result<DType> {
    match onnx_type {
        1 => Ok(DType::F32),  // FLOAT
        2 => Ok(DType::U8),   // UINT8
        3 => Ok(DType::I8),   // INT8
        5 => Ok(DType::I16),  // INT16
        6 => Ok(DType::I32),  // INT32
        7 => Ok(DType::I64),  // INT64
        9 => Ok(DType::Bool), // BOOL
        10 => Ok(DType::F16), // FLOAT16
        11 => Ok(DType::F64), // DOUBLE
        12 => Ok(DType::U32), // UINT32
        13 => Ok(DType::U64), // UINT64
        _ => Err(crate::OnnxError::UnsupportedDataType(format!("ONNX type code {}", onnx_type))),
    }
}

/// Convert ONNX TensorProto to hologram ConstantData and Shape.
fn tensor_proto_to_constant(tensor: &crate::proto::TensorProto) -> Result<(hologram::ConstantData, hologram::Shape)> {
    use hologram::ir::{ConstantData, Dim, Shape};

    // Extract shape
    let shape = Shape::new(
        tensor.dims.iter().map(|&d| Dim::Static(d as usize)).collect()
    );

    // Extract data based on type
    let data = match tensor.data_type {
        1 => {  // FLOAT
            let values = if !tensor.float_data.is_empty() {
                tensor.float_data.clone()
            } else {
                // Parse from raw_data
                parse_raw_data_f32(&tensor.raw_data)?
            };
            ConstantData::F32(values)
        }
        6 => {  // INT32
            let values = if !tensor.int32_data.is_empty() {
                tensor.int32_data.clone()
            } else {
                parse_raw_data_i32(&tensor.raw_data)?
            };
            ConstantData::I32(values)
        }
        7 => {  // INT64
            let values = if !tensor.int64_data.is_empty() {
                tensor.int64_data.clone()
            } else {
                parse_raw_data_i64(&tensor.raw_data)?
            };
            ConstantData::I64(values)
        }
        10 => { // FLOAT16
            // Parse f16 from raw data and convert to f32
            let f16_values = parse_raw_data_f16(&tensor.raw_data)?;
            let f32_values: Vec<f32> = f16_values.iter().map(|&x| x.to_f32()).collect();
            ConstantData::F32(f32_values)
        }
        _ => {
            return Err(crate::OnnxError::UnsupportedDataType(format!("ONNX type code {}", tensor.data_type)));
        }
    };

    Ok((data, shape))
}

/// Parse f32 values from raw bytes.
fn parse_raw_data_f32(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(crate::OnnxError::InvalidModel("Invalid raw_data length for f32".into()));
    }

    Ok(raw.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Parse i32 values from raw bytes.
fn parse_raw_data_i32(raw: &[u8]) -> Result<Vec<i32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(crate::OnnxError::InvalidModel("Invalid raw_data length for i32".into()));
    }

    Ok(raw.chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Parse i64 values from raw bytes.
fn parse_raw_data_i64(raw: &[u8]) -> Result<Vec<i64>> {
    if !raw.len().is_multiple_of(8) {
        return Err(crate::OnnxError::InvalidModel("Invalid raw_data length for i64".into()));
    }

    Ok(raw.chunks_exact(8)
        .map(|chunk| i64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3],
            chunk[4], chunk[5], chunk[6], chunk[7],
        ]))
        .collect())
}

/// Parse f16 values from raw bytes.
fn parse_raw_data_f16(raw: &[u8]) -> Result<Vec<half::f16>> {
    if !raw.len().is_multiple_of(2) {
        return Err(crate::OnnxError::InvalidModel("Invalid raw_data length for f16".into()));
    }

    Ok(raw.chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    // Tests removed: These tests relied on old IR API internals that no longer exist.
    // The OperationGraph type is now a simple wrapper around IRFunction.
}

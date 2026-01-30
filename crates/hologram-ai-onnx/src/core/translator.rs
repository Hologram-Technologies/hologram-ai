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
        self.ir_func.to_bytes().map_err(crate::OnnxError::IrError)
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

/// Global translator registry (lazily initialized).
static REGISTRY: std::sync::LazyLock<crate::translators::TranslatorRegistry> =
    std::sync::LazyLock::new(crate::translators::TranslatorRegistry::new);

/// Translate a single ONNX node using the trait-based registry.
///
/// This function resolves inputs from the value map and dispatches to the
/// appropriate translator from the registry.
fn translate_node_via_registry(
    node: &crate::proto::NodeProto,
    builder: &mut hologram::ir::GraphBuilder,
    value_map: &std::collections::HashMap<String, hologram::NodeIndex>,
) -> Result<Vec<hologram::NodeIndex>> {
    // Resolve inputs from value map (filtering empty optional inputs)
    let inputs: Result<Vec<hologram::NodeIndex>> = node
        .input
        .iter()
        .filter(|input_name| !input_name.is_empty())
        .map(|input_name| {
            value_map.get(input_name).copied().ok_or_else(|| {
                crate::OnnxError::MissingInput(format!(
                    "Input '{}' not found for node '{}' ({})",
                    input_name, node.name, node.op_type
                ))
            })
        })
        .collect();

    let inputs = inputs?;

    // Dispatch to registry
    REGISTRY
        .translate(node, &inputs, builder)
        .map_err(crate::OnnxError::from)
}

fn translate_if_node(
    node: &crate::proto::NodeProto,
    builder: &mut hologram::ir::GraphBuilder,
    parent_values: &std::collections::HashMap<String, hologram::NodeIndex>,
) -> Result<Vec<hologram::NodeIndex>> {
    use tracing::debug;

    let condition_name = node.input.first().ok_or_else(|| {
        crate::OnnxError::InvalidModel("If node missing condition input".to_string())
    })?;
    let condition = parent_values.get(condition_name).copied().ok_or_else(|| {
        crate::OnnxError::MissingInput(format!(
            "If condition '{}' not found in parent graph",
            condition_name
        ))
    })?;

    let then_graph = find_if_branch(node, "then_branch")?;
    let else_graph = find_if_branch(node, "else_branch")?;

    debug!(
        "Translating If node '{}' with {} then nodes, {} else nodes",
        node.name,
        then_graph.node.len(),
        else_graph.node.len()
    );

    let then_outputs = translate_subgraph(then_graph, builder, parent_values)?;
    let else_outputs = translate_subgraph(else_graph, builder, parent_values)?;

    if then_outputs.len() != else_outputs.len() {
        return Err(crate::OnnxError::InvalidModel(format!(
            "If branches output count mismatch: then={}, else={}",
            then_outputs.len(),
            else_outputs.len()
        )));
    }

    if node.output.len() != then_outputs.len() {
        return Err(crate::OnnxError::InvalidModel(format!(
            "If node outputs {} do not match branch outputs {}",
            node.output.len(),
            then_outputs.len()
        )));
    }

    let mut selected = Vec::with_capacity(then_outputs.len());
    for (then_out, else_out) in then_outputs.iter().zip(else_outputs.iter()) {
        let out = builder.where_select(condition, *then_out, *else_out)?;
        selected.push(out);
    }

    Ok(selected)
}

fn find_if_branch<'a>(
    node: &'a crate::proto::NodeProto,
    name: &str,
) -> Result<&'a crate::proto::GraphProto> {
    let attr = node
        .attribute
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| {
            crate::OnnxError::InvalidModel(format!("If node missing '{}' attribute", name))
        })?;

    attr.g.as_ref().ok_or_else(|| {
        crate::OnnxError::InvalidModel(format!("If node attribute '{}' missing graph value", name))
    })
}

fn translate_subgraph(
    graph: &crate::proto::GraphProto,
    builder: &mut hologram::ir::GraphBuilder,
    parent_values: &std::collections::HashMap<String, hologram::NodeIndex>,
) -> Result<Vec<hologram::NodeIndex>> {
    use std::collections::HashMap;
    use tracing::trace;

    let mut value_map: HashMap<String, hologram::NodeIndex> = parent_values.clone();

    for input in &graph.input {
        if value_map.contains_key(&input.name) {
            continue;
        }
        // Preserve symbolic dimensions for proper DimExpr resolution at runtime
        let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;
        let dtype = extract_dtype_from_value_info(input)?;
        let node_idx = builder.input(&input.name, shape.into_inner(), dtype);
        value_map.insert(input.name.clone(), node_idx);
    }

    for initializer in &graph.initializer {
        if value_map.contains_key(&initializer.name) {
            continue;
        }
        let (constant_data, shape) = tensor_proto_to_constant(initializer)?;
        let node_idx = builder.constant(constant_data, shape);
        value_map.insert(initializer.name.clone(), node_idx);
    }

    for (idx, node) in graph.node.iter().enumerate() {
        trace!(
            "Translating subgraph node {}/{}: {} ({})",
            idx + 1,
            graph.node.len(),
            node.name,
            node.op_type
        );
        let outputs = if node.op_type == "If" {
            translate_if_node(node, builder, &value_map)?
        } else {
            translate_node_via_registry(node, builder, &value_map)?
        };
        for (output_name, output_idx) in node.output.iter().zip(outputs.iter()) {
            if !output_name.is_empty() {
                value_map.insert(output_name.clone(), *output_idx);
            }
        }
    }

    let mut outputs = Vec::with_capacity(graph.output.len());
    for output in &graph.output {
        let idx = value_map.get(&output.name).copied().ok_or_else(|| {
            crate::OnnxError::InvalidModel(format!("Subgraph output '{}' not found", output.name))
        })?;
        outputs.push(idx);
    }

    Ok(outputs)
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
    use std::collections::HashMap;
    use tracing::{debug, trace};

    let mut builder = GraphBuilder::new();
    let mut value_map: HashMap<String, hologram::NodeIndex> = HashMap::new();

    // Step 0.5: Build value_info lookup map for intermediate tensor shapes
    let mut value_info_shapes: HashMap<String, crate::core::SymbolicShape> = HashMap::new();
    for value_info in &graph.value_info {
        if let Ok(shape) = crate::core::SymbolicShape::from_value_info(value_info) {
            trace!("Registered shape for '{}': {:?}", value_info.name, &shape);
            value_info_shapes.insert(value_info.name.clone(), shape);
        }
    }
    debug!(
        "Loaded {} value_info shape entries",
        value_info_shapes.len()
    );

    // Step 1: Process inputs with symbolic shapes
    // Use preserve_symbolic to maintain symbolic dimensions for DimExpr resolution
    debug!("Processing {} graph inputs", graph.input.len());
    for (i, input) in graph.input.iter().enumerate() {
        let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;

        // Determine dtype from ONNX type
        let dtype = extract_dtype_from_value_info(input)?;

        trace!(
            "Adding input '{}': shape={:?}, dtype={:?}",
            input.name, shape, dtype
        );
        let node_idx = builder.input(&input.name, shape.into_inner(), dtype);
        value_map.insert(input.name.clone(), node_idx);

        // Debug: Log input_ids mapping specifically
        if input.name == "input_ids" {
            debug!(
                "INPUT_IDS MAPPING: '{}' (ONNX input {}) -> NodeIndex({:?})",
                input.name, i, node_idx
            );
        }
    }

    // Step 2: Process initializers (constants/weights)
    debug!("Processing {} initializers", graph.initializer.len());
    for initializer in &graph.initializer {
        if value_map.contains_key(&initializer.name) {
            // Already added as input, skip
            trace!(
                "Skipping initializer '{}' (already in inputs)",
                initializer.name
            );
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
        trace!(
            "Translating node {}/{}: {} ({})",
            idx + 1,
            graph.node.len(),
            node.name,
            node.op_type
        );

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

        // Translate this ONNX node to IR operations using the new registry
        let output_indices = if node.op_type == "If" {
            translate_if_node(node, &mut builder, &value_map)?
        } else {
            translate_node_via_registry(node, &mut builder, &value_map)?
        };

        // Map outputs to their node indices and validate shapes
        for (output_name, output_idx) in node.output.iter().zip(output_indices.iter()) {
            if !output_name.is_empty() {
                value_map.insert(output_name.clone(), *output_idx);

                // Log shape information from value_info for debugging
                if let (Some(expected_shape), Some(ir_node)) = (
                    value_info_shapes.get(output_name),
                    builder.graph().node(*output_idx),
                ) {
                    let ir_shape = &ir_node.op.shape;
                    // Only log if ranks differ
                    if ir_shape.rank() != expected_shape.rank() {
                        debug!(
                            "Rank mismatch for '{}': IR rank={}, value_info rank={}",
                            output_name,
                            ir_shape.rank(),
                            expected_shape.rank()
                        );
                    }
                }
            }
        }
    }

    // Step 4: Set up outputs with declared shapes
    debug!("Processing {} graph outputs", graph.output.len());
    for output in &graph.output {
        let output_node = value_map.get(&output.name).ok_or_else(|| {
            crate::OnnxError::InvalidModel(format!(
                "Graph output '{}' not found in value map",
                output.name
            ))
        })?;

        // Extract declared shape from ONNX ValueInfoProto to preserve symbolic dimensions
        let declared_shape = crate::core::SymbolicShape::from_value_info(output)
            .ok()
            .map(|s| s.into_inner());

        trace!(
            "Adding output '{}' with declared shape {:?}",
            output.name, declared_shape
        );
        builder.output_with_shape(&output.name, *output_node, declared_shape)?;
    }

    let ir_func = builder.build();

    // Debug: Check if edges were created
    debug!(
        "Graph translation complete: {} nodes, {} edges",
        ir_func.inner().node_count(),
        ir_func.inner().edge_count()
    );

    Ok(ir_func)
}

/// Translate ONNX GraphProto to hologram IR with execution groups.
///
/// This variant detects attention patterns (Q, K, V projections) and creates
/// execution groups for parallel execution. Use this for transformer models
/// where Q, K, V projections can run in parallel.
///
/// # Arguments
///
/// * `graph` - ONNX graph proto to translate
///
/// # Returns
///
/// Fully translated hologram IR OperationGraph with execution groups configured.
/// Call `parallel_groups()` on the result to get parallelizable group levels.
///
/// # Example
///
/// ```ignore
/// let ir_func = translate_graph_to_ir_with_groups(graph)?;
/// let levels = ir_func.inner().parallel_groups();
/// for (level_idx, groups) in levels.iter().enumerate() {
///     println!("Level {}: {} groups can run in parallel", level_idx, groups.len());
/// }
/// ```
pub fn translate_graph_to_ir_with_groups(graph: &crate::proto::GraphProto) -> Result<IRFunction> {
    use crate::core::activation_fusion::{chain_name, detect_activation_chains};
    use crate::core::attention_detection::{
        assign_execution_groups, detect_attention_patterns, get_group_dependencies,
    };
    use crate::core::op_hints::add_composed_view_hint;
    use hologram::ir::GraphBuilder;
    use std::collections::HashMap;
    use tracing::{debug, info, trace};

    // Detect attention patterns before translation
    let attention_patterns = detect_attention_patterns(graph);
    let has_groups = !attention_patterns.is_empty();

    if has_groups {
        info!(
            "Detected {} attention patterns - enabling parallel execution groups",
            attention_patterns.len()
        );
    }

    // Detect activation chains for fusion
    let activation_chains = detect_activation_chains(graph);
    if !activation_chains.is_empty() {
        info!(
            "Detected {} activation chains for fusion",
            activation_chains.len()
        );
        for chain in &activation_chains {
            debug!("  Chain: {}", chain_name(chain));
        }
    }

    // Pre-compute group assignments for each node
    let group_assignments = assign_execution_groups(&attention_patterns, graph.node.len());
    let group_deps = get_group_dependencies(&attention_patterns);

    let mut builder = GraphBuilder::new();
    let mut value_map: HashMap<String, hologram::NodeIndex> = HashMap::new();

    // Create execution groups if attention patterns were detected
    let mut group_id_map: HashMap<u64, hologram::ir::GroupId> = HashMap::new();

    if has_groups {
        // Find max group ID needed
        let max_group = group_assignments.iter().copied().max().unwrap_or(0);

        // Create groups (group 0 already exists as default)
        for group_num in 1..=max_group {
            let gid = builder.create_group();
            group_id_map.insert(group_num, gid);
            debug!("Created execution group {} -> {:?}", group_num, gid);
        }

        // Set up group dependencies
        for (dependent, dependency) in group_deps {
            let dep_gid = group_id_map.get(&dependent).copied();
            let dependency_gid = if dependency == 0 {
                // Group 0 is the default group (already exists)
                hologram::ir::GroupId::new(0)
            } else {
                group_id_map
                    .get(&dependency)
                    .copied()
                    .unwrap_or(hologram::ir::GroupId::new(0))
            };

            if let Some(dep) = dep_gid {
                builder.add_group_dependency(dep, dependency_gid);
                debug!("Group {:?} depends on {:?}", dep, dependency_gid);
            }
        }
    }

    // Step 1: Process inputs with symbolic shapes
    // Use preserve_symbolic to maintain symbolic dimensions for DimExpr resolution
    debug!("Processing {} graph inputs", graph.input.len());
    for input in &graph.input {
        let shape = crate::core::SymbolicShape::from_value_info_preserve_symbolic(input)?;
        let dtype = extract_dtype_from_value_info(input)?;

        trace!(
            "Adding input '{}': shape={:?}, dtype={:?}",
            input.name, shape, dtype
        );
        let node_idx = builder.input(&input.name, shape.into_inner(), dtype);
        value_map.insert(input.name.clone(), node_idx);
    }

    // Step 2: Process initializers (constants/weights)
    debug!("Processing {} initializers", graph.initializer.len());
    for initializer in &graph.initializer {
        if value_map.contains_key(&initializer.name) {
            trace!(
                "Skipping initializer '{}' (already in inputs)",
                initializer.name
            );
            continue;
        }

        let (constant_data, shape) = tensor_proto_to_constant(initializer)?;
        trace!("Adding constant '{}': shape={:?}", initializer.name, shape);
        let node_idx = builder.constant(constant_data, shape);
        value_map.insert(initializer.name.clone(), node_idx);
    }

    // Step 3: Translate all nodes
    debug!("Processing {} graph nodes", graph.node.len());
    for (idx, node) in graph.node.iter().enumerate() {
        let assigned_group = group_assignments.get(idx).copied().unwrap_or(0);

        trace!(
            "Translating node {}/{}: {} ({}) [group {}]",
            idx + 1,
            graph.node.len(),
            node.name,
            node.op_type,
            assigned_group
        );

        // Group assignment is handled at graph level by attention_detection.rs.
        // Individual node translation uses the standard registry - this is correct
        // separation of concerns (pattern detection vs. operation translation).
        let output_indices = if node.op_type == "If" {
            translate_if_node(node, &mut builder, &value_map)?
        } else {
            translate_node_via_registry(node, &mut builder, &value_map)?
        };

        for (output_name, output_idx) in node.output.iter().zip(output_indices.iter()) {
            if !output_name.is_empty() {
                value_map.insert(output_name.clone(), *output_idx);
            }
        }
    }

    // Step 3b: Apply composed view hints for activation chains
    if !activation_chains.is_empty() {
        debug!(
            "Applying composed view hints for {} activation chains",
            activation_chains.len()
        );
        for chain in &activation_chains {
            // Find the output node of the chain (last activation in the chain)
            if let Some(&output_node) = value_map.get(&chain.output_tensor) {
                let table_ids = chain.table_ids();
                add_composed_view_hint(builder.graph_mut(), output_node, &table_ids);
                trace!(
                    "Added composed view hint for chain '{}' -> node {:?} with tables {:?}",
                    chain_name(chain),
                    output_node,
                    table_ids
                );
            }
        }
    }

    // Step 4: Set up outputs with declared shapes
    debug!("Processing {} graph outputs", graph.output.len());
    for output in &graph.output {
        let output_node = value_map.get(&output.name).ok_or_else(|| {
            crate::OnnxError::InvalidModel(format!(
                "Graph output '{}' not found in value map",
                output.name
            ))
        })?;

        let declared_shape = crate::core::SymbolicShape::from_value_info(output)
            .ok()
            .map(|s| s.into_inner());

        trace!(
            "Adding output '{}' with declared shape {:?}",
            output.name, declared_shape
        );
        builder.output_with_shape(&output.name, *output_node, declared_shape)?;
    }

    // Step 5: Resolve execution order if groups were created
    if has_groups {
        builder.resolve_execution_order()?;
        let levels = builder.parallel_groups();
        info!(
            "Execution groups resolved: {} levels, {} total groups",
            levels.len(),
            builder.num_groups()
        );
        for (level_idx, groups) in levels.iter().enumerate() {
            debug!(
                "  Level {}: {} groups can run in parallel",
                level_idx,
                groups.len()
            );
        }
    }

    let ir_func = builder.build();

    debug!(
        "Graph translation complete: {} nodes, {} edges, {} groups",
        ir_func.inner().node_count(),
        ir_func.inner().edge_count(),
        ir_func.num_groups()
    );

    Ok(ir_func)
}

/// Extract DType from ONNX ValueInfoProto.
fn extract_dtype_from_value_info(value_info: &crate::proto::ValueInfoProto) -> Result<DType> {
    use crate::proto::type_proto::Value;

    let type_proto = value_info.r#type.as_ref().ok_or_else(|| {
        crate::OnnxError::InvalidModel(format!(
            "Value '{}' has no type information",
            value_info.name
        ))
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
        _ => Err(crate::OnnxError::UnsupportedDataType(format!(
            "ONNX type code {}",
            onnx_type
        ))),
    }
}

/// Convert ONNX TensorProto to hologram ConstantData and Shape.
fn tensor_proto_to_constant(
    tensor: &crate::proto::TensorProto,
) -> Result<(hologram::ConstantData, hologram::Shape)> {
    use hologram::ir::{ConstantData, Dim, Shape};

    // Extract shape
    let shape = Shape::new(
        tensor
            .dims
            .iter()
            .map(|&d| Dim::Static(d as usize))
            .collect(),
    );

    // Extract data based on type
    let data = match tensor.data_type {
        1 => {
            // FLOAT
            let values = if !tensor.float_data.is_empty() {
                tensor.float_data.clone()
            } else {
                // Parse from raw_data
                parse_raw_data_f32(&tensor.raw_data)?
            };
            ConstantData::F32(values)
        }
        6 => {
            // INT32
            let values = if !tensor.int32_data.is_empty() {
                tensor.int32_data.clone()
            } else {
                parse_raw_data_i32(&tensor.raw_data)?
            };
            ConstantData::I32(values)
        }
        7 => {
            // INT64
            let values = if !tensor.int64_data.is_empty() {
                tensor.int64_data.clone()
            } else {
                parse_raw_data_i64(&tensor.raw_data)?
            };
            ConstantData::I64(values)
        }
        10 => {
            // FLOAT16
            // Parse f16 from raw data and convert to f32
            let f16_values = parse_raw_data_f16(&tensor.raw_data)?;
            let f32_values: Vec<f32> = f16_values.iter().map(|&x| x.to_f32()).collect();
            ConstantData::F32(f32_values)
        }
        _ => {
            return Err(crate::OnnxError::UnsupportedDataType(format!(
                "ONNX type code {}",
                tensor.data_type
            )));
        }
    };

    Ok((data, shape))
}

/// Parse f32 values from raw bytes.
fn parse_raw_data_f32(raw: &[u8]) -> Result<Vec<f32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(crate::OnnxError::InvalidModel(
            "Invalid raw_data length for f32".into(),
        ));
    }

    Ok(raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Parse i32 values from raw bytes.
fn parse_raw_data_i32(raw: &[u8]) -> Result<Vec<i32>> {
    if !raw.len().is_multiple_of(4) {
        return Err(crate::OnnxError::InvalidModel(
            "Invalid raw_data length for i32".into(),
        ));
    }

    Ok(raw
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Parse i64 values from raw bytes.
fn parse_raw_data_i64(raw: &[u8]) -> Result<Vec<i64>> {
    if !raw.len().is_multiple_of(8) {
        return Err(crate::OnnxError::InvalidModel(
            "Invalid raw_data length for i64".into(),
        ));
    }

    Ok(raw
        .chunks_exact(8)
        .map(|chunk| {
            i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])
        })
        .collect())
}

/// Parse f16 values from raw bytes.
fn parse_raw_data_f16(raw: &[u8]) -> Result<Vec<half::f16>> {
    if !raw.len().is_multiple_of(2) {
        return Err(crate::OnnxError::InvalidModel(
            "Invalid raw_data length for f16".into(),
        ));
    }

    Ok(raw
        .chunks_exact(2)
        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::tensor_shape_proto::dimension::Value as DimValue;
    use crate::proto::type_proto::Value as TypeValue;
    use crate::proto::{
        AttributeProto, GraphProto, NodeProto, TensorShapeProto, TypeProto, ValueInfoProto,
    };
    use hologram::ir::NodeOp;

    fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> ValueInfoProto {
        let shape = TensorShapeProto {
            dim: dims
                .iter()
                .map(|&d| crate::proto::tensor_shape_proto::Dimension {
                    value: Some(DimValue::DimValue(d)),
                    ..Default::default()
                })
                .collect(),
        };
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                value: Some(TypeValue::TensorType(crate::proto::type_proto::Tensor {
                    elem_type,
                    shape: Some(shape),
                })),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn build_if_graph() -> GraphProto {
        let then_graph = GraphProto {
            name: "then_branch".to_string(),
            input: vec![value_info("x", 1, &[1])],
            output: vec![value_info("out", 1, &[1])],
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["x".to_string()],
                output: vec!["out".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };

        let else_graph = GraphProto {
            name: "else_branch".to_string(),
            input: vec![value_info("y", 1, &[1])],
            output: vec![value_info("out", 1, &[1])],
            node: vec![NodeProto {
                op_type: "Identity".to_string(),
                input: vec!["y".to_string()],
                output: vec!["out".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        };

        GraphProto {
            name: "if_graph".to_string(),
            input: vec![
                value_info("cond", 9, &[1]),
                value_info("x", 1, &[1]),
                value_info("y", 1, &[1]),
            ],
            output: vec![value_info("out", 1, &[1])],
            node: vec![NodeProto {
                name: "if_node".to_string(),
                op_type: "If".to_string(),
                input: vec!["cond".to_string()],
                output: vec!["out".to_string()],
                attribute: vec![
                    AttributeProto {
                        name: "then_branch".to_string(),
                        g: Some(then_graph),
                        ..Default::default()
                    },
                    AttributeProto {
                        name: "else_branch".to_string(),
                        g: Some(else_graph),
                        ..Default::default()
                    },
                ],
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_if_node_creates_where() {
        let graph = build_if_graph();
        let ir_func = translate_graph_to_ir(&graph).expect("translation failed");

        let has_where = ir_func
            .nodes()
            .any(|(_, node)| matches!(node.op.op, NodeOp::Where));
        assert!(has_where, "expected Where op for If lowering");

        assert!(
            ir_func.outputs.contains_key("out"),
            "expected output 'out' to be registered"
        );
    }
}

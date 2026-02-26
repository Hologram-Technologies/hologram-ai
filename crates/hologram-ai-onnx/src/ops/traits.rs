//! Traits for ONNX operation translation.
//!
//! All ONNX operations implement [`OpTranslator`] which provides a unified interface
//! for both constant folding and runtime translation.

use std::collections::HashMap;

use anyhow::Result;
use hologram::compiler::{ConstantData, DType, OpKind, OpNode, OperationGraph};

use crate::proto;

/// Information about broadcasting needed for binary operations.
///
/// When a binary operation (Add, Sub, Mul, Div) has inputs with different shapes,
/// the hologram ISA requires both inputs to have identical element counts.
/// This struct captures which inputs need to be expanded before the binary op.
#[derive(Debug, Clone)]
pub struct BroadcastInfo {
    /// Shape of the first input.
    pub a_shape: Vec<usize>,
    /// Shape of the second input.
    pub b_shape: Vec<usize>,
    /// Target shape (broadcast output shape).
    pub output_shape: Vec<usize>,
    /// Whether input A needs to be expanded to output_shape.
    pub a_needs_broadcast: bool,
    /// Whether input B needs to be expanded to output_shape.
    pub b_needs_broadcast: bool,
}

/// Result of translating an ONNX operation.
#[derive(Debug, Clone)]
pub struct TranslateResult {
    /// The hologram operation kind.
    pub op_kind: OpKind,
    /// Output shape.
    pub shape: Vec<usize>,
    /// Output data type.
    pub dtype: DType,
    /// Optional constant data (for constant-folded operations).
    pub constant_data: Option<ConstantData>,
    /// Number of data inputs (edges) for this operation.
    /// This is the count from the start of the ONNX input list.
    /// Remaining inputs are metadata (shape, axes, etc.) and don't create edges.
    /// Default: None = all inputs create edges.
    pub data_input_count: Option<usize>,
    /// Optional broadcast info for binary operations requiring input expansion.
    /// When present, the builder should insert Expand nodes before the binary op.
    pub broadcast_info: Option<BroadcastInfo>,
}

impl TranslateResult {
    /// Create a new translate result for a runtime operation.
    pub fn runtime(op_kind: OpKind, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            op_kind,
            shape,
            dtype,
            constant_data: None,
            data_input_count: None,
            broadcast_info: None,
        }
    }

    /// Create a result with a specific number of data inputs.
    pub fn runtime_with_inputs(
        op_kind: OpKind,
        shape: Vec<usize>,
        dtype: DType,
        data_inputs: usize,
    ) -> Self {
        Self {
            op_kind,
            shape,
            dtype,
            constant_data: None,
            data_input_count: Some(data_inputs),
            broadcast_info: None,
        }
    }

    /// Create a new translate result for a constant-folded operation.
    pub fn constant(shape: Vec<usize>, dtype: DType, data: ConstantData) -> Self {
        Self {
            op_kind: OpKind::Constant,
            shape,
            dtype,
            constant_data: Some(data),
            data_input_count: Some(0), // Constants have no data inputs
            broadcast_info: None,
        }
    }

    /// Create a result for a binary operation that requires broadcasting.
    ///
    /// This signals to the graph builder that it needs to insert Expand nodes
    /// before the binary operation to ensure both inputs have the same shape.
    pub fn broadcast_binary(
        op_kind: OpKind,
        output_shape: Vec<usize>,
        dtype: DType,
        broadcast_info: BroadcastInfo,
    ) -> Self {
        Self {
            op_kind,
            shape: output_shape,
            dtype,
            constant_data: None,
            data_input_count: Some(2), // Binary ops have 2 data inputs
            broadcast_info: Some(broadcast_info),
        }
    }
}

/// Context for operation translation.
///
/// Provides access to the graph being built and helper methods for
/// looking up input nodes and their properties.
pub struct TranslateContext<'a> {
    /// The operation graph being built.
    pub graph: &'a OperationGraph,
    /// Mapping from ONNX value names to node IDs.
    pub value_to_node: &'a HashMap<String, u32>,
    /// Constant data stored in the graph, indexed by constant node position.
    pub constants: &'a [ConstantData],
}

impl<'a> TranslateContext<'a> {
    /// Create a new translation context.
    pub fn new(
        graph: &'a OperationGraph,
        value_to_node: &'a HashMap<String, u32>,
        constants: &'a [ConstantData],
    ) -> Self {
        Self {
            graph,
            value_to_node,
            constants,
        }
    }

    /// Get a node by its ONNX name.
    pub fn get_node(&self, name: &str) -> Option<&OpNode> {
        self.value_to_node
            .get(name)
            .map(|&id| &self.graph.nodes[id as usize])
    }

    /// Get node ID by name.
    #[allow(dead_code)]
    pub fn get_node_id(&self, name: &str) -> Option<u32> {
        self.value_to_node.get(name).copied()
    }

    /// Check if a node is a constant.
    pub fn is_constant(&self, name: &str) -> bool {
        self.get_node(name)
            .map(|n| matches!(n.op, OpKind::Constant))
            .unwrap_or(false)
    }

    /// Get constant data for a node by name.
    ///
    /// Returns `None` if the node doesn't exist or isn't a constant.
    pub fn get_constant_data(&self, name: &str) -> Option<&ConstantData> {
        let node_id = *self.value_to_node.get(name)?;
        let node = &self.graph.nodes[node_id as usize];

        if !matches!(node.op, OpKind::Constant) {
            return None;
        }

        // Count constants up to and including this node
        let const_idx = self
            .graph
            .nodes
            .iter()
            .take(node_id as usize + 1)
            .filter(|n| matches!(n.op, OpKind::Constant))
            .count()
            .checked_sub(1)?;

        self.constants.get(const_idx)
    }

    /// Get constant data as i64 values.
    pub fn get_constant_i64(&self, name: &str) -> Option<Vec<i64>> {
        match self.get_constant_data(name)? {
            ConstantData::I64(data) => Some(data.clone()),
            ConstantData::I32(data) => Some(data.iter().map(|&x| x as i64).collect()),
            _ => None,
        }
    }

    /// Get constant data as f32 values.
    #[allow(dead_code)]
    pub fn get_constant_f32(&self, name: &str) -> Option<Vec<f32>> {
        match self.get_constant_data(name)? {
            ConstantData::F32(data) => Some(data.clone()),
            ConstantData::F64(data) => Some(data.iter().map(|&x| x as f32).collect()),
            _ => None,
        }
    }
}

/// Trait for ONNX operation translators.
///
/// Each ONNX operation type implements this trait to handle both constant folding
/// (compile-time evaluation) and runtime translation (converting to hologram ops).
///
/// # Example
///
/// ```ignore
/// pub struct ShapeOp;
///
/// impl OpTranslator for ShapeOp {
///     fn op_type(&self) -> &'static str { "Shape" }
///
///     fn try_fold(&self, node: &NodeProto, ctx: &TranslateContext) -> Option<TranslateResult> {
///         // Shape always produces a constant from input shape
///         let input = ctx.get_node(node.input.first()?)?;
///         let shape_values: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
///         Some(TranslateResult::constant(
///             vec![shape_values.len()],
///             DType::I64,
///             ConstantData::I64(shape_values),
///         ))
///     }
///
///     fn translate(&self, node: &NodeProto, ctx: &TranslateContext) -> Result<TranslateResult> {
///         // Shape should always be folded, but fallback just in case
///         let input = ctx.get_node(node.input.first().context("no input")?).context("not found")?;
///         Ok(TranslateResult::runtime(
///             OpKind::Constant,
///             vec![input.shape.len()],
///             DType::I64,
///         ))
///     }
/// }
/// ```
pub trait OpTranslator: Send + Sync {
    /// The ONNX operation type name (e.g., "Shape", "MatMul", "Add").
    #[allow(dead_code)]
    fn op_type(&self) -> &'static str;

    /// Attempt to constant-fold this operation.
    ///
    /// Returns `Some(result)` if the operation can be evaluated at compile time,
    /// `None` if constant folding is not possible (e.g., non-constant inputs).
    ///
    /// The default implementation returns `None` (no constant folding).
    fn try_fold(
        &self,
        _node: &proto::NodeProto,
        _ctx: &TranslateContext,
    ) -> Option<TranslateResult> {
        None
    }

    /// Translate this operation to a hologram operation.
    ///
    /// This is called when constant folding is not possible. It should convert
    /// the ONNX operation to the equivalent hologram `OpKind`.
    fn translate(&self, node: &proto::NodeProto, ctx: &TranslateContext)
    -> Result<TranslateResult>;

    /// Returns true if this operation requires special graph expansion.
    ///
    /// Some ONNX operations (like Gemm) expand into multiple hologram nodes
    /// rather than a single node. The builder handles these specially.
    fn requires_expansion(&self) -> bool {
        false
    }
}

/// Helper trait for extracting scalar values from constant data.
pub trait ConstantScalar {
    /// Extract a scalar i64 value.
    fn as_i64_scalar(&self) -> Option<i64>;
    /// Extract a scalar f32 value.
    fn as_f32_scalar(&self) -> Option<f32>;
}

impl ConstantScalar for ConstantData {
    fn as_i64_scalar(&self) -> Option<i64> {
        match self {
            ConstantData::I64(data) if data.len() == 1 => Some(data[0]),
            ConstantData::I32(data) if data.len() == 1 => Some(data[0] as i64),
            _ => None,
        }
    }

    fn as_f32_scalar(&self) -> Option<f32> {
        match self {
            ConstantData::F32(data) if data.len() == 1 => Some(data[0]),
            ConstantData::F64(data) if data.len() == 1 => Some(data[0] as f32),
            _ => None,
        }
    }
}

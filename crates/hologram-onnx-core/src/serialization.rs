#![allow(missing_docs)]
//! Serialization of IR functions to .holo format.
//!
//! This module provides proper serialization of compiled models including:
//! - IR graph structure (nodes and connections)
//! - Weight data (embedded or external based on threshold)
//! - Input/output specifications
//! - Type information
//!
//! # .holo Format (v1)
//!
//! ```text
//! +------------------------+
//! | Header (32 bytes)      |
//! +------------------------+
//! | Metadata (JSON)        |
//! +------------------------+
//! | Node Graph (bincode)   |
//! +------------------------+
//! | Embedded Weights       |
//! +------------------------+
//! ```
//!
//! ## Header
//! - Magic: "HOLO" (4 bytes)
//! - Version: u32 (4 bytes)
//! - Flags: u32 (4 bytes)
//! - Metadata offset: u64 (8 bytes)
//! - Graph offset: u64 (8 bytes)
//! - Weights offset: u64 (8 bytes) - 0 if external

use std::collections::HashMap;
use std::io;
use std::path::Path;

use hologram_compiler::ir::{BinOp, ConstValue, ReduceOp, UnOp};
use hologram_compiler::ir::{IRFunction, IRNode, IRNodeEntry, NodeId, Type};
use hologram_compiler::shapes::{Dim, Shape};
use serde::{Deserialize, Serialize};

use crate::{OnnxError, Result};

// =============================================================================
// Constants
// =============================================================================

/// Magic bytes for .holo files
pub const HOLO_MAGIC: &[u8; 4] = b"HOLO";

/// Current format version
pub const FORMAT_VERSION: u32 = 1;

/// Header size in bytes
/// Layout: magic(4) + version(4) + flags(4) + metadata_offset(8) + graph_offset(8) + weights_offset(8) = 36
/// Padded to 40 for alignment
pub const HEADER_SIZE: usize = 40;

/// Flag: has external weights file
pub const FLAG_EXTERNAL_WEIGHTS: u32 = 0x01;

/// Flag: has compressed data
pub const FLAG_COMPRESSED: u32 = 0x02;

// =============================================================================
// Serializable Types
// =============================================================================

/// Serializable representation of a .holo file header.
#[derive(Debug, Clone)]
pub struct HoloHeader {
    pub version: u32,
    pub flags: u32,
    pub metadata_offset: u64,
    pub graph_offset: u64,
    pub weights_offset: u64,
}

impl HoloHeader {
    /// Convert header to raw bytes for writing.
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(HOLO_MAGIC);
        buf[4..8].copy_from_slice(&self.version.to_le_bytes());
        buf[8..12].copy_from_slice(&self.flags.to_le_bytes());
        buf[12..20].copy_from_slice(&self.metadata_offset.to_le_bytes());
        buf[20..28].copy_from_slice(&self.graph_offset.to_le_bytes());
        buf[28..36].copy_from_slice(&self.weights_offset.to_le_bytes());
        // bytes 36-40 are reserved padding
        buf
    }

    /// Parse header from raw bytes.
    pub fn from_bytes(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_SIZE {
            return Err(OnnxError::InvalidModel("Header too small".into()));
        }
        if &buf[0..4] != HOLO_MAGIC {
            return Err(OnnxError::InvalidModel("Invalid magic bytes".into()));
        }

        Ok(Self {
            version: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            flags: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
            metadata_offset: u64::from_le_bytes([
                buf[12], buf[13], buf[14], buf[15], buf[16], buf[17], buf[18], buf[19],
            ]),
            graph_offset: u64::from_le_bytes([
                buf[20], buf[21], buf[22], buf[23], buf[24], buf[25], buf[26], buf[27],
            ]),
            weights_offset: u64::from_le_bytes([
                buf[28], buf[29], buf[30], buf[31], buf[32], buf[33], buf[34], buf[35],
            ]),
        })
    }

    pub fn has_external_weights(&self) -> bool {
        self.flags & FLAG_EXTERNAL_WEIGHTS != 0
    }
}

/// Serializable metadata for the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoloMetadata {
    /// Function/model name
    pub name: String,
    /// Input specifications
    pub inputs: Vec<InputSpec>,
    /// Output specifications
    pub outputs: Vec<OutputSpec>,
    /// Total embedded weight size in bytes
    pub embedded_weight_size: u64,
    /// Total external weight size in bytes
    pub external_weight_size: u64,
    /// Number of nodes in the graph
    pub node_count: usize,
}

/// Input specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSpec {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<DimSpec>,
}

/// Output specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputSpec {
    pub node_id: usize,
    pub dtype: String,
    pub shape: Vec<DimSpec>,
}

/// Dimension specification (concrete or symbolic)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DimSpec {
    Concrete(usize),
    Symbolic(String),
}

// =============================================================================
// Serializable Node Graph
// =============================================================================

/// Serializable node entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerNode {
    pub id: usize,
    pub node: SerNodeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shape: Option<Vec<DimSpec>>,
}

/// Serializable node kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SerNodeKind {
    // Values
    Input {
        name: String,
    },
    Constant {
        weight_id: usize,
    }, // Reference to weight table
    ScalarConst {
        value: f64,
    }, // Inline scalar constant
    WeightRef {
        name: String,
        offset: u64,
        size: usize,
    },

    // Arithmetic
    BinaryOp {
        op: String,
        lhs: usize,
        rhs: usize,
    },
    UnaryOp {
        op: String,
        operand: usize,
    },

    // Matrix
    MatMul {
        lhs: usize,
        rhs: usize,
    },
    Softmax {
        input: usize,
        axis: i32,
    },

    // Shape operations
    Reshape {
        input: usize,
        shape: Vec<DimSpec>,
    },
    Transpose {
        input: usize,
        perm: Vec<usize>,
    },
    Broadcast {
        input: usize,
        shape: Vec<DimSpec>,
    },
    Slice {
        input: usize,
        ranges: Vec<(Option<i64>, Option<i64>, Option<i64>)>,
    },
    Gather {
        input: usize,
        indices: usize,
        axis: i32,
    },
    Concat {
        inputs: Vec<usize>,
        axis: i32,
    },

    // Reduction
    Reduce {
        op: String,
        input: usize,
        axes: Vec<i32>,
        keepdims: bool,
    },

    // Control flow
    Select {
        cond: usize,
        on_true: usize,
        on_false: usize,
    },
    Phi {
        inputs: Vec<usize>,
    },

    // Neural network ops
    Conv2D {
        input: usize,
        weight: usize,
        bias: Option<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },
    BatchNorm {
        input: usize,
        scale: usize,
        bias: usize,
        mean: usize,
        var: usize,
        epsilon: f32,
    },
    MaxPool {
        input: usize,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },
    AvgPool {
        input: usize,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },

    // Other
    Cast {
        input: usize,
        dtype: String,
    },
    Call {
        func: String,
        args: Vec<usize>,
    },
    Im2Col {
        input: usize,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },
    Col2Im {
        input: usize,
        output_shape: Vec<usize>,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },
    Unfold {
        input: usize,
        kernel: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },
    Stack {
        inputs: Vec<usize>,
        axis: i32,
    },
    VStack {
        inputs: Vec<usize>,
    },
    HStack {
        inputs: Vec<usize>,
    },
}

/// Serializable graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerGraph {
    pub nodes: Vec<SerNode>,
    pub outputs: Vec<usize>,
}

/// Weight table entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightEntry {
    pub id: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub offset: u64,    // Offset in weights section
    pub size: usize,    // Size in bytes
    pub external: bool, // True if in external file
}

/// Complete serializable model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerModel {
    pub metadata: HoloMetadata,
    pub graph: SerGraph,
    pub weights: Vec<WeightEntry>,
}

// =============================================================================
// Serialization
// =============================================================================

/// Serialization context
pub struct HoloSerializer {
    weight_threshold: usize,
    weights: Vec<WeightEntry>,
    embedded_data: Vec<u8>,
    external_data: Vec<u8>,
    weight_id_map: HashMap<NodeId, usize>,
}

impl HoloSerializer {
    /// Create a new serializer with the given weight threshold.
    /// Weights larger than this threshold will be stored externally.
    pub fn new(weight_threshold: usize) -> Self {
        Self {
            weight_threshold,
            weights: Vec::new(),
            embedded_data: Vec::new(),
            external_data: Vec::new(),
            weight_id_map: HashMap::new(),
        }
    }

    /// Serialize an IR function to .holo format.
    pub fn serialize(&mut self, func: &IRFunction) -> Result<(Vec<u8>, Vec<u8>)> {
        // First pass: extract weights and build weight table
        self.extract_weights(func)?;

        // Build serializable graph
        let graph = self.build_graph(func)?;

        // Build metadata
        let metadata = self.build_metadata(func);

        // Create model structure
        let model = SerModel {
            metadata,
            graph,
            weights: self.weights.clone(),
        };

        // Serialize model to JSON
        let model_json =
            serde_json::to_vec(&model).map_err(|e| OnnxError::SerializationError(e.to_string()))?;

        // Build final .holo file
        let mut holo_data = Vec::new();

        // Calculate offsets
        let metadata_offset = HEADER_SIZE as u64;
        let model_json_len = model_json.len() as u64;
        let weights_offset = if self.embedded_data.is_empty() {
            0
        } else {
            metadata_offset + model_json_len
        };

        // Create header
        let mut flags = 0u32;
        if !self.external_data.is_empty() {
            flags |= FLAG_EXTERNAL_WEIGHTS;
        }

        let header = HoloHeader {
            version: FORMAT_VERSION,
            flags,
            metadata_offset,
            graph_offset: metadata_offset, // Same as metadata for v1 (single JSON blob)
            weights_offset,
        };

        // Write header
        holo_data.extend_from_slice(&header.to_bytes());

        // Write model JSON
        holo_data.extend_from_slice(&model_json);

        // Write embedded weights
        if !self.embedded_data.is_empty() {
            holo_data.extend_from_slice(&self.embedded_data);
        }

        Ok((holo_data, self.external_data.clone()))
    }

    /// Extract weights from IR function
    fn extract_weights(&mut self, func: &IRFunction) -> Result<()> {
        for entry in &func.body {
            if let IRNode::Constant {
                value: ConstValue::Tensor { shape, data },
                ty,
            } = &entry.node
            {
                let weight_id = self.weights.len();
                let size = data.len();
                let external = size > self.weight_threshold;

                let dtype = type_to_dtype_string(ty);

                let offset = if external {
                    let off = self.external_data.len() as u64;
                    self.external_data.extend_from_slice(data);
                    off
                } else {
                    let off = self.embedded_data.len() as u64;
                    self.embedded_data.extend_from_slice(data);
                    off
                };

                self.weights.push(WeightEntry {
                    id: weight_id,
                    name: format!("weight_{}", weight_id),
                    shape: shape.clone(),
                    dtype,
                    offset,
                    size,
                    external,
                });

                self.weight_id_map.insert(entry.id, weight_id);
            }
        }
        Ok(())
    }

    /// Build serializable graph from IR function
    fn build_graph(&self, func: &IRFunction) -> Result<SerGraph> {
        let mut nodes = Vec::new();

        for entry in &func.body {
            let node = self.convert_node(entry)?;
            nodes.push(node);
        }

        let outputs = func.outputs.iter().map(|id| id.0).collect();

        Ok(SerGraph { nodes, outputs })
    }

    /// Convert an IR node to serializable form
    fn convert_node(&self, entry: &IRNodeEntry) -> Result<SerNode> {
        let id = entry.id.0;
        let dtype = Some(type_to_dtype_string(&entry.ty));
        let shape = type_to_shape_spec(&entry.ty);

        let node = match &entry.node {
            IRNode::Input { name, .. } => SerNodeKind::Input { name: name.clone() },

            IRNode::Constant { value, .. } => {
                // Check if this is a tensor constant (stored in weight table)
                // or a scalar constant (stored inline)
                if let Some(&weight_id) = self.weight_id_map.get(&entry.id) {
                    SerNodeKind::Constant { weight_id }
                } else {
                    // Scalar constant - convert to f64 for inline storage
                    let scalar_value = match value {
                        ConstValue::F32(v) => *v as f64,
                        ConstValue::F64(v) => *v,
                        ConstValue::I32(v) => *v as f64,
                        ConstValue::I64(v) => *v as f64,
                        ConstValue::Bool(v) => {
                            if *v {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        ConstValue::Tensor { .. } => {
                            return Err(OnnxError::SerializationError(format!(
                                "Tensor constant {} missing from weight table",
                                id
                            )));
                        }
                    };
                    SerNodeKind::ScalarConst {
                        value: scalar_value,
                    }
                }
            }

            IRNode::WeightRef {
                name, offset, size, ..
            } => SerNodeKind::WeightRef {
                name: name.clone(),
                offset: *offset,
                size: *size,
            },

            IRNode::BinaryOp { op, lhs, rhs } => SerNodeKind::BinaryOp {
                op: binop_to_string(*op),
                lhs: lhs.0,
                rhs: rhs.0,
            },

            IRNode::UnaryOp { op, operand } => SerNodeKind::UnaryOp {
                op: unop_to_string(*op),
                operand: operand.0,
            },

            IRNode::MatMul { lhs, rhs } => SerNodeKind::MatMul {
                lhs: lhs.0,
                rhs: rhs.0,
            },

            IRNode::Softmax { input, axis } => SerNodeKind::Softmax {
                input: input.0,
                axis: *axis as i32,
            },

            IRNode::Reshape { input, shape } => SerNodeKind::Reshape {
                input: input.0,
                shape: shape_to_spec(shape),
            },

            IRNode::Transpose { input, perm } => SerNodeKind::Transpose {
                input: input.0,
                perm: perm.clone().unwrap_or_default(),
            },

            IRNode::Broadcast { input, target } => SerNodeKind::Broadcast {
                input: input.0,
                shape: shape_to_spec(target),
            },

            IRNode::Slice { input, ranges } => SerNodeKind::Slice {
                input: input.0,
                ranges: ranges
                    .iter()
                    .map(|r| {
                        (
                            r.start.map(|s| s as i64),
                            r.end.map(|e| e as i64),
                            r.step.map(|s| s as i64),
                        )
                    })
                    .collect(),
            },

            IRNode::Gather {
                data,
                indices,
                axis,
            } => SerNodeKind::Gather {
                input: data.0,
                indices: indices.0,
                axis: *axis as i32,
            },

            IRNode::Concat { inputs, axis } => SerNodeKind::Concat {
                inputs: inputs.iter().map(|id| id.0).collect(),
                axis: *axis as i32,
            },

            IRNode::Reduce {
                op,
                input,
                axes,
                keepdims,
            } => SerNodeKind::Reduce {
                op: reduceop_to_string(*op),
                input: input.0,
                axes: axes.iter().map(|&a| a as i32).collect(),
                keepdims: *keepdims,
            },

            IRNode::Select {
                cond,
                true_val,
                false_val,
            } => SerNodeKind::Select {
                cond: cond.0,
                on_true: true_val.0,
                on_false: false_val.0,
            },

            IRNode::Phi { branches } => SerNodeKind::Phi {
                inputs: branches.iter().map(|(_, id)| id.0).collect(),
            },

            IRNode::Conv2D {
                input,
                kernel,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => SerNodeKind::Conv2D {
                input: input.0,
                weight: kernel.0,
                bias: bias.map(|id| id.0),
                stride: tuple_to_vec(*stride),
                padding: tuple_to_vec(*padding),
                dilation: tuple_to_vec(*dilation),
                groups: *groups,
            },

            IRNode::BatchNorm {
                input,
                scale,
                bias,
                mean,
                var,
                epsilon,
            } => SerNodeKind::BatchNorm {
                input: input.0,
                scale: scale.0,
                bias: bias.0,
                mean: mean.0,
                var: var.0,
                epsilon: *epsilon,
            },

            IRNode::MaxPool {
                input,
                kernel_size,
                stride,
                padding,
            } => SerNodeKind::MaxPool {
                input: input.0,
                kernel: tuple_to_vec(*kernel_size),
                stride: tuple_to_vec(*stride),
                padding: tuple_to_vec(*padding),
            },

            IRNode::AvgPool {
                input,
                kernel_size,
                stride,
                padding,
            } => SerNodeKind::AvgPool {
                input: input.0,
                kernel: tuple_to_vec(*kernel_size),
                stride: tuple_to_vec(*stride),
                padding: tuple_to_vec(*padding),
            },

            IRNode::Cast { input, target_type } => SerNodeKind::Cast {
                input: input.0,
                dtype: type_to_dtype_string(target_type),
            },

            IRNode::Call { func, args } => SerNodeKind::Call {
                func: func.clone(),
                args: args.iter().map(|id| id.0).collect(),
            },

            IRNode::Im2Col {
                input,
                kernel_size,
                stride,
                padding,
                dilation,
            } => SerNodeKind::Im2Col {
                input: input.0,
                kernel: tuple_to_vec(*kernel_size),
                stride: tuple_to_vec(*stride),
                padding: tuple_to_vec(*padding),
                dilation: tuple_to_vec(*dilation),
            },

            IRNode::Col2Im {
                input,
                output_size,
                kernel_size,
                stride,
                padding,
                dilation,
            } => SerNodeKind::Col2Im {
                input: input.0,
                output_shape: tuple_to_vec(*output_size),
                kernel: tuple_to_vec(*kernel_size),
                stride: tuple_to_vec(*stride),
                padding: tuple_to_vec(*padding),
                dilation: tuple_to_vec(*dilation),
            },

            IRNode::Unfold {
                input,
                kernel_size,
                stride,
                padding,
            } => {
                SerNodeKind::Unfold {
                    input: input.0,
                    kernel: tuple_to_vec(*kernel_size),
                    stride: tuple_to_vec(*stride),
                    padding: tuple_to_vec(*padding),
                    dilation: vec![1, 1], // Unfold doesn't have dilation field
                }
            }

            IRNode::Stack { inputs, axis } => SerNodeKind::Stack {
                inputs: inputs.iter().map(|id| id.0).collect(),
                axis: *axis as i32,
            },

            IRNode::VStack { inputs } => SerNodeKind::VStack {
                inputs: inputs.iter().map(|id| id.0).collect(),
            },

            IRNode::HStack { inputs } => SerNodeKind::HStack {
                inputs: inputs.iter().map(|id| id.0).collect(),
            },
        };

        Ok(SerNode {
            id,
            node,
            dtype,
            shape,
        })
    }

    /// Build metadata from IR function
    fn build_metadata(&self, func: &IRFunction) -> HoloMetadata {
        let inputs: Vec<InputSpec> = func
            .params
            .iter()
            .map(|(name, ty)| InputSpec {
                name: name.clone(),
                dtype: type_to_dtype_string(ty),
                shape: type_to_shape_spec(ty).unwrap_or_default(),
            })
            .collect();

        let outputs: Vec<OutputSpec> = func
            .outputs
            .iter()
            .map(|id| {
                let entry = func.get_node(*id);
                OutputSpec {
                    node_id: id.0,
                    dtype: entry
                        .map(|e| type_to_dtype_string(&e.ty))
                        .unwrap_or_default(),
                    shape: entry
                        .and_then(|e| type_to_shape_spec(&e.ty))
                        .unwrap_or_default(),
                }
            })
            .collect();

        let embedded_weight_size: u64 = self
            .weights
            .iter()
            .filter(|w| !w.external)
            .map(|w| w.size as u64)
            .sum();

        let external_weight_size: u64 = self
            .weights
            .iter()
            .filter(|w| w.external)
            .map(|w| w.size as u64)
            .sum();

        HoloMetadata {
            name: func.name.clone(),
            inputs,
            outputs,
            embedded_weight_size,
            external_weight_size,
            node_count: func.body.len(),
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn type_to_dtype_string(ty: &Type) -> String {
    match ty {
        Type::Tensor(t) => format!("{:?}", t.elem).to_lowercase(),
        Type::Scalar(s) => format!("{:?}", s).to_lowercase(),
        Type::Tuple(_) => "tuple".to_string(),
        Type::Function(_) => "function".to_string(),
        Type::Void => "void".to_string(),
        Type::Unknown => "unknown".to_string(),
    }
}

fn type_to_shape_spec(ty: &Type) -> Option<Vec<DimSpec>> {
    match ty {
        Type::Tensor(t) => Some(t.shape.dims().iter().map(dim_to_spec).collect()),
        _ => None,
    }
}

fn shape_to_spec(shape: &Shape) -> Vec<DimSpec> {
    shape.dims().iter().map(dim_to_spec).collect()
}

fn dim_to_spec(d: &Dim) -> DimSpec {
    match d {
        Dim::Concrete(n) => DimSpec::Concrete(*n),
        Dim::Var(name) => DimSpec::Symbolic(name.clone()),
        Dim::Expr(expr) => DimSpec::Symbolic(format!("{:?}", expr)),
    }
}

fn tuple_to_vec(t: (usize, usize)) -> Vec<usize> {
    vec![t.0, t.1]
}

fn binop_to_string(op: BinOp) -> String {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "sub",
        BinOp::Mul => "mul",
        BinOp::Div => "div",
        BinOp::Pow => "pow",
        BinOp::Mod => "mod",
        BinOp::Min => "min",
        BinOp::Max => "max",
        BinOp::Eq => "eq",
        BinOp::Ne => "ne",
        BinOp::Lt => "lt",
        BinOp::Le => "le",
        BinOp::Gt => "gt",
        BinOp::Ge => "ge",
        BinOp::And => "and",
        BinOp::Or => "or",
    }
    .to_string()
}

fn unop_to_string(op: UnOp) -> String {
    match op {
        UnOp::Neg => "neg",
        UnOp::Abs => "abs",
        UnOp::Not => "not",
        UnOp::Sqrt => "sqrt",
        UnOp::Rsqrt => "rsqrt",
        UnOp::Exp => "exp",
        UnOp::Log => "log",
        UnOp::Sin => "sin",
        UnOp::Cos => "cos",
        UnOp::Tan => "tan",
        UnOp::Floor => "floor",
        UnOp::Ceil => "ceil",
        UnOp::Round => "round",
        UnOp::Sigmoid => "sigmoid",
        UnOp::Tanh => "tanh",
        UnOp::ReLU => "relu",
        UnOp::GELU => "gelu",
    }
    .to_string()
}

fn reduceop_to_string(op: ReduceOp) -> String {
    match op {
        ReduceOp::Sum => "sum",
        ReduceOp::Prod => "prod",
        ReduceOp::Mean => "mean",
        ReduceOp::Max => "max",
        ReduceOp::Min => "min",
        ReduceOp::ArgMax => "argmax",
        ReduceOp::ArgMin => "argmin",
    }
    .to_string()
}

// =============================================================================
// Public API
// =============================================================================

/// Serialize an IR function to .holo format.
///
/// Returns (holo_bytes, weights_bytes) where weights_bytes is empty if all
/// weights are embedded.
pub fn serialize_ir_function(
    func: &IRFunction,
    weight_threshold: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut serializer = HoloSerializer::new(weight_threshold);
    serializer.serialize(func)
}

/// Write compiled model to files.
pub fn write_compiled_model(
    func: &IRFunction,
    output_path: &Path,
    weight_threshold: usize,
) -> Result<(usize, usize)> {
    let (holo_bytes, weight_bytes) = serialize_ir_function(func, weight_threshold)?;

    let holo_path = output_path.with_extension("holo");
    std::fs::write(&holo_path, &holo_bytes).map_err(|e| {
        OnnxError::IoError(io::Error::other(format!("Failed to write .holo: {}", e)))
    })?;

    let weights_size = if !weight_bytes.is_empty() {
        let weights_path = output_path.with_extension("weights");
        std::fs::write(&weights_path, &weight_bytes).map_err(|e| {
            OnnxError::IoError(io::Error::other(format!("Failed to write .weights: {}", e)))
        })?;
        weight_bytes.len()
    } else {
        0
    };

    Ok((holo_bytes.len(), weights_size))
}

// =============================================================================
// Deserialization
// =============================================================================

/// A loaded .holo model ready for execution.
#[derive(Debug)]
pub struct HoloModel {
    /// Header information
    pub header: HoloHeader,
    /// Model metadata
    pub metadata: HoloMetadata,
    /// Graph structure
    pub graph: SerGraph,
    /// Weight entries (metadata only)
    pub weight_entries: Vec<WeightEntry>,
    /// Embedded weight data (weights that were below threshold)
    pub embedded_weights: Vec<u8>,
    /// External weight data (weights that were above threshold)
    pub external_weights: Vec<u8>,
}

impl HoloModel {
    /// Get weight data by weight ID.
    pub fn get_weight(&self, weight_id: usize) -> Option<&[u8]> {
        let entry = self.weight_entries.get(weight_id)?;
        let offset = entry.offset as usize;
        let size = entry.size;

        if entry.external {
            if offset + size <= self.external_weights.len() {
                Some(&self.external_weights[offset..offset + size])
            } else {
                None
            }
        } else if offset + size <= self.embedded_weights.len() {
            Some(&self.embedded_weights[offset..offset + size])
        } else {
            None
        }
    }

    /// Get weight data as f32 slice (assumes f32 dtype).
    pub fn get_weight_f32(&self, weight_id: usize) -> Option<&[f32]> {
        let data = self.get_weight(weight_id)?;
        // Safety: we trust the format specifies correct dtype
        if data.len() % 4 != 0 {
            return None;
        }
        let len = data.len() / 4;
        // Safe because Vec<u8> from file is properly aligned
        Some(unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, len) })
    }

    /// Get all inputs required by this model.
    pub fn inputs(&self) -> &[InputSpec] {
        &self.metadata.inputs
    }

    /// Get output specifications.
    pub fn outputs(&self) -> &[OutputSpec] {
        &self.metadata.outputs
    }

    /// Get the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.metadata.node_count
    }
}

/// Load a .holo model from a file.
///
/// If the model has external weights (FLAG_EXTERNAL_WEIGHTS), this function
/// will look for a .weights file alongside the .holo file.
pub fn load_holo_file(path: &Path) -> Result<HoloModel> {
    let holo_bytes = std::fs::read(path).map_err(|e| {
        OnnxError::IoError(io::Error::other(format!(
            "Failed to read .holo file: {}",
            e
        )))
    })?;

    load_holo_bytes(&holo_bytes, Some(path))
}

/// Load a .holo model from bytes.
///
/// If `holo_path` is provided and the model has external weights,
/// this function will attempt to load the .weights file.
pub fn load_holo_bytes(holo_bytes: &[u8], holo_path: Option<&Path>) -> Result<HoloModel> {
    if holo_bytes.len() < HEADER_SIZE {
        return Err(OnnxError::InvalidModel("File too small for header".into()));
    }

    // Parse header
    let header = HoloHeader::from_bytes(&holo_bytes[..HEADER_SIZE])?;

    // Validate version
    if header.version != FORMAT_VERSION {
        return Err(OnnxError::InvalidModel(format!(
            "Unsupported format version: {} (expected {})",
            header.version, FORMAT_VERSION
        )));
    }

    // Parse model JSON
    let metadata_start = header.metadata_offset as usize;
    let weights_start = if header.weights_offset > 0 {
        header.weights_offset as usize
    } else {
        holo_bytes.len()
    };

    if metadata_start >= holo_bytes.len() {
        return Err(OnnxError::InvalidModel("Invalid metadata offset".into()));
    }

    let model_json = &holo_bytes[metadata_start..weights_start];
    let model: SerModel = serde_json::from_slice(model_json)
        .map_err(|e| OnnxError::InvalidModel(format!("Failed to parse model JSON: {}", e)))?;

    // Extract embedded weights
    let embedded_weights = if header.weights_offset > 0 && weights_start < holo_bytes.len() {
        holo_bytes[weights_start..].to_vec()
    } else {
        Vec::new()
    };

    // Load external weights if needed
    let external_weights = if header.has_external_weights() {
        if let Some(path) = holo_path {
            let weights_path = path.with_extension("weights");
            std::fs::read(&weights_path).map_err(|e| {
                OnnxError::IoError(io::Error::other(format!(
                    "Failed to read external weights file '{}': {}",
                    weights_path.display(),
                    e
                )))
            })?
        } else {
            return Err(OnnxError::InvalidModel(
                "Model has external weights but no path provided".into(),
            ));
        }
    } else {
        Vec::new()
    };

    Ok(HoloModel {
        header,
        metadata: model.metadata,
        graph: model.graph,
        weight_entries: model.weights,
        embedded_weights,
        external_weights,
    })
}

/// Validate a .holo file without fully loading it.
pub fn validate_holo_file(path: &Path) -> Result<HoloMetadata> {
    let holo_bytes = std::fs::read(path).map_err(|e| {
        OnnxError::IoError(io::Error::other(format!(
            "Failed to read .holo file: {}",
            e
        )))
    })?;

    if holo_bytes.len() < HEADER_SIZE {
        return Err(OnnxError::InvalidModel("File too small for header".into()));
    }

    let header = HoloHeader::from_bytes(&holo_bytes[..HEADER_SIZE])?;

    if header.version != FORMAT_VERSION {
        return Err(OnnxError::InvalidModel(format!(
            "Unsupported format version: {} (expected {})",
            header.version, FORMAT_VERSION
        )));
    }

    let metadata_start = header.metadata_offset as usize;
    let weights_start = if header.weights_offset > 0 {
        header.weights_offset as usize
    } else {
        holo_bytes.len()
    };

    let model_json = &holo_bytes[metadata_start..weights_start];
    let model: SerModel = serde_json::from_slice(model_json)
        .map_err(|e| OnnxError::InvalidModel(format!("Failed to parse model JSON: {}", e)))?;

    // Verify external weights exist if needed
    if header.has_external_weights() {
        let weights_path = path.with_extension("weights");
        if !weights_path.exists() {
            return Err(OnnxError::InvalidModel(format!(
                "External weights file not found: {}",
                weights_path.display()
            )));
        }
    }

    Ok(model.metadata)
}

/// Print detailed info about a .holo file.
pub fn inspect_holo_file(path: &Path) -> Result<String> {
    let model = load_holo_file(path)?;
    let mut output = String::new();

    use std::fmt::Write;

    writeln!(output, "=== HOLO Model: {} ===", path.display()).unwrap();
    writeln!(output, "\nFormat Version: {}", model.header.version).unwrap();
    writeln!(output, "Flags: 0x{:08x}", model.header.flags).unwrap();
    if model.header.has_external_weights() {
        writeln!(output, "  - External weights: yes").unwrap();
    }

    writeln!(output, "\n--- Metadata ---").unwrap();
    writeln!(output, "Name: {}", model.metadata.name).unwrap();
    writeln!(output, "Node count: {}", model.metadata.node_count).unwrap();
    writeln!(
        output,
        "Embedded weight size: {} bytes",
        model.metadata.embedded_weight_size
    )
    .unwrap();
    writeln!(
        output,
        "External weight size: {} bytes",
        model.metadata.external_weight_size
    )
    .unwrap();

    writeln!(output, "\n--- Inputs ---").unwrap();
    for input in &model.metadata.inputs {
        let shape_str: Vec<String> = input
            .shape
            .iter()
            .map(|d| match d {
                DimSpec::Concrete(n) => n.to_string(),
                DimSpec::Symbolic(s) => s.clone(),
            })
            .collect();
        writeln!(
            output,
            "  {}: {} [{}]",
            input.name,
            input.dtype,
            shape_str.join(", ")
        )
        .unwrap();
    }

    writeln!(output, "\n--- Outputs ---").unwrap();
    for (i, output_spec) in model.metadata.outputs.iter().enumerate() {
        let shape_str: Vec<String> = output_spec
            .shape
            .iter()
            .map(|d| match d {
                DimSpec::Concrete(n) => n.to_string(),
                DimSpec::Symbolic(s) => s.clone(),
            })
            .collect();
        writeln!(
            output,
            "  output_{}: {} [{}] (node {})",
            i,
            output_spec.dtype,
            shape_str.join(", "),
            output_spec.node_id
        )
        .unwrap();
    }

    writeln!(output, "\n--- Weights ({}) ---", model.weight_entries.len()).unwrap();
    for entry in &model.weight_entries {
        let loc = if entry.external {
            "external"
        } else {
            "embedded"
        };
        writeln!(
            output,
            "  {}: {} {:?} @ offset {} ({} bytes, {})",
            entry.id, entry.dtype, entry.shape, entry.offset, entry.size, loc
        )
        .unwrap();
    }

    writeln!(
        output,
        "\n--- Graph ({} nodes) ---",
        model.graph.nodes.len()
    )
    .unwrap();
    for node in &model.graph.nodes {
        let kind = match &node.node {
            SerNodeKind::Input { name } => format!("Input({})", name),
            SerNodeKind::Constant { weight_id } => format!("Constant(weight_{})", weight_id),
            SerNodeKind::ScalarConst { value } => format!("ScalarConst({})", value),
            SerNodeKind::WeightRef { name, .. } => format!("WeightRef({})", name),
            SerNodeKind::BinaryOp { op, lhs, rhs } => format!("{}({}, {})", op, lhs, rhs),
            SerNodeKind::UnaryOp { op, operand } => format!("{}({})", op, operand),
            SerNodeKind::MatMul { lhs, rhs } => format!("MatMul({}, {})", lhs, rhs),
            SerNodeKind::Softmax { input, axis } => format!("Softmax({}, axis={})", input, axis),
            SerNodeKind::Reshape { input, .. } => format!("Reshape({})", input),
            SerNodeKind::Transpose { input, .. } => format!("Transpose({})", input),
            SerNodeKind::Broadcast { input, .. } => format!("Broadcast({})", input),
            SerNodeKind::Slice { input, .. } => format!("Slice({})", input),
            SerNodeKind::Gather {
                input,
                indices,
                axis,
            } => format!("Gather({}, {}, axis={})", input, indices, axis),
            SerNodeKind::Concat { inputs, axis } => format!("Concat({:?}, axis={})", inputs, axis),
            SerNodeKind::Reduce {
                op, input, axes, ..
            } => format!("Reduce{}({}, axes={:?})", op, input, axes),
            SerNodeKind::Select {
                cond,
                on_true,
                on_false,
            } => format!("Select({}, {}, {})", cond, on_true, on_false),
            SerNodeKind::Phi { inputs } => format!("Phi({:?})", inputs),
            SerNodeKind::Conv2D { input, weight, .. } => format!("Conv2D({}, {})", input, weight),
            SerNodeKind::BatchNorm { input, .. } => format!("BatchNorm({})", input),
            SerNodeKind::MaxPool { input, .. } => format!("MaxPool({})", input),
            SerNodeKind::AvgPool { input, .. } => format!("AvgPool({})", input),
            SerNodeKind::Cast { input, dtype } => format!("Cast({}, {})", input, dtype),
            SerNodeKind::Call { func, args } => format!("Call({}, {:?})", func, args),
            SerNodeKind::Im2Col { input, .. } => format!("Im2Col({})", input),
            SerNodeKind::Col2Im { input, .. } => format!("Col2Im({})", input),
            SerNodeKind::Unfold { input, .. } => format!("Unfold({})", input),
            SerNodeKind::Stack { inputs, axis } => format!("Stack({:?}, axis={})", inputs, axis),
            SerNodeKind::VStack { inputs } => format!("VStack({:?})", inputs),
            SerNodeKind::HStack { inputs } => format!("HStack({:?})", inputs),
        };
        writeln!(output, "  [{}] {}", node.id, kind).unwrap();
    }

    writeln!(output, "\nOutputs: {:?}", model.graph.outputs).unwrap();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram_compiler::ir::{IRBuilder, ScalarType};

    #[test]
    fn test_serialize_simple_function() {
        let mut builder = IRBuilder::new("test");
        let x = builder.add_input(
            "x",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        let y = builder.add_input(
            "y",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        let sum = builder.add(x, y);
        builder.set_output(sum);
        let func = builder.build();

        let (holo_bytes, weights_bytes) = serialize_ir_function(&func, 4096).unwrap();

        // Check magic
        assert_eq!(&holo_bytes[0..4], HOLO_MAGIC);
        // No weights for this simple function
        assert!(weights_bytes.is_empty());
    }

    #[test]
    fn test_serialize_with_weights() {
        let mut builder = IRBuilder::new("test");
        let x = builder.add_input(
            "x",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        // Add a small weight tensor
        let w = builder.add_tensor_const(vec![3, 4], vec![0u8; 48], ScalarType::F32);
        let result = builder.matmul(x, w);
        builder.set_output(result);
        let func = builder.build();

        let (holo_bytes, weights_bytes) = serialize_ir_function(&func, 4096).unwrap();

        // Check magic
        assert_eq!(&holo_bytes[0..4], HOLO_MAGIC);
        // Small weights should be embedded
        assert!(weights_bytes.is_empty());
        // Holo file should contain embedded weights
        assert!(holo_bytes.len() > HEADER_SIZE + 48);
    }

    #[test]
    fn test_serialize_with_external_weights() {
        let mut builder = IRBuilder::new("test");
        let x = builder.add_input(
            "x",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        // Add a large weight tensor (over threshold)
        let w = builder.add_tensor_const(vec![100, 100], vec![0u8; 40000], ScalarType::F32);
        let result = builder.matmul(x, w);
        builder.set_output(result);
        let func = builder.build();

        // Use small threshold to force external storage
        let (holo_bytes, weights_bytes) = serialize_ir_function(&func, 1000).unwrap();

        // Check magic
        assert_eq!(&holo_bytes[0..4], HOLO_MAGIC);
        // Large weights should be external
        assert_eq!(weights_bytes.len(), 40000);
    }

    #[test]
    fn test_roundtrip_simple() {
        let mut builder = IRBuilder::new("roundtrip_test");
        let x = builder.add_input(
            "x",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        let y = builder.add_input(
            "y",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );
        let sum = builder.add(x, y);
        builder.set_output(sum);
        let func = builder.build();

        let (holo_bytes, _weights_bytes) = serialize_ir_function(&func, 4096).unwrap();

        // Load without file path (no external weights)
        let model = load_holo_bytes(&holo_bytes, None).unwrap();

        // Verify metadata
        assert_eq!(model.metadata.name, "roundtrip_test");
        assert_eq!(model.metadata.inputs.len(), 2);
        assert_eq!(model.metadata.inputs[0].name, "x");
        assert_eq!(model.metadata.inputs[1].name, "y");

        // Verify graph
        assert_eq!(model.graph.nodes.len(), 3); // x, y, add
        assert_eq!(model.graph.outputs.len(), 1);
    }

    #[test]
    fn test_roundtrip_with_weights() {
        let mut builder = IRBuilder::new("weights_test");
        let x = builder.add_input(
            "x",
            Type::tensor(ScalarType::F32, Shape::concrete(vec![2, 3])),
        );

        // Create weight data with known values
        let weight_data: Vec<u8> = (0..48u8).collect(); // 12 f32 values
        let w = builder.add_tensor_const(vec![3, 4], weight_data.clone(), ScalarType::F32);
        let result = builder.matmul(x, w);
        builder.set_output(result);
        let func = builder.build();

        let (holo_bytes, _weights_bytes) = serialize_ir_function(&func, 4096).unwrap();

        let model = load_holo_bytes(&holo_bytes, None).unwrap();

        // Verify weight is accessible
        assert_eq!(model.weight_entries.len(), 1);
        let weight = model.get_weight(0).unwrap();
        assert_eq!(weight.len(), 48);
        assert_eq!(weight, &weight_data[..]);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = HoloHeader {
            version: FORMAT_VERSION,
            flags: FLAG_EXTERNAL_WEIGHTS,
            metadata_offset: 40,
            graph_offset: 100,
            weights_offset: 500,
        };

        let bytes = header.to_bytes();
        let parsed = HoloHeader::from_bytes(&bytes).unwrap();

        assert_eq!(parsed.version, header.version);
        assert_eq!(parsed.flags, header.flags);
        assert_eq!(parsed.metadata_offset, header.metadata_offset);
        assert_eq!(parsed.graph_offset, header.graph_offset);
        assert_eq!(parsed.weights_offset, header.weights_offset);
        assert!(parsed.has_external_weights());
    }

    #[test]
    fn test_invalid_magic() {
        let bad_bytes = b"NOPE0000000000000000000000000000000000000";
        let result = HoloHeader::from_bytes(bad_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_holo_bytes_too_small() {
        let small_bytes = b"HOLO";
        let result = load_holo_bytes(small_bytes, None);
        assert!(result.is_err());
    }
}

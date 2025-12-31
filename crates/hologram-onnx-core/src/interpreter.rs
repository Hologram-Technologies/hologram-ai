//! CPU-based interpreter for executing .holo models with optional backend acceleration.
//!
//! This module provides a simple interpreter that executes SerGraph nodes
//! using ndarray for tensor operations. It's designed for:
//! - Correctness verification
//! - CPU fallback when backend execution is not available
//! - Simple deployment without GPU requirements
//!
//! # Example
//!
//! ```ignore
//! use hologram_onnx_core::{load_holo_file, Interpreter};
//!
//! let model = load_holo_file("model.holo")?;
//! let mut interpreter = Interpreter::new(&model)?;
//!
//! // Set input tensor
//! interpreter.set_input("input", &input_data)?;
//!
//! // Run inference
//! interpreter.run()?;
//!
//! // Get output
//! let output = interpreter.get_output(0)?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use hologram_backend::Backend;
use hologram_core::{BackendType, Buffer, Executor};
use matrixmultiply::sgemm;
use ndarray::{Array, ArrayD, Axis, IxDyn};
use rayon::prelude::*;
use tracing::{debug, info, trace, warn};

use crate::serialization::{DimSpec, HoloModel, PackedWeightKind, SerNode, SerNodeKind};
use crate::{OnnxError, Result};

/// Runtime backend selection for interpreter acceleration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    /// Use backend auto-detection (preferred).
    Auto,
    /// Force CPU backend.
    Cpu,
    /// Force CUDA backend.
    Cuda,
    /// Force Metal backend.
    Metal,
    /// Force WebGPU backend.
    WebGpu,
}

impl RuntimeBackend {
    /// Parse a backend name from configuration.
    pub fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "metal" => Ok(Self::Metal),
            "webgpu" => Ok(Self::WebGpu),
            other => Err(OnnxError::InvalidModel(format!(
                "Unknown runtime backend '{}'",
                other
            ))),
        }
    }

    fn backend_type(self) -> Option<BackendType> {
        match self {
            Self::Auto => None,
            Self::Cpu => Some(BackendType::Cpu),
            Self::Cuda => Some(BackendType::Cuda),
            Self::Metal => Some(BackendType::Metal),
            Self::WebGpu => Some(BackendType::WebGpu),
        }
    }
}

struct BackendContext {
    exec: Executor,
}

impl BackendContext {
    fn new(backend: RuntimeBackend) -> Result<Self> {
        let exec = Executor::new_with_boxed_backend(
            Box::new(Backend::detect()),
            backend.backend_type(),
        )
        .map_err(|e| {
            OnnxError::HologramError(format!("Failed to initialize backend executor: {}", e))
        })?;
        Ok(Self { exec })
    }
}

/// Tensor value in the interpreter
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Data stored as f32 (we convert from other types as needed)
    pub data: ArrayD<f32>,
}

impl Tensor {
    /// Create a new tensor from a flat vector and shape
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(OnnxError::InvalidModel(format!(
                "Data size {} doesn't match shape {:?} (expected {})",
                data.len(),
                shape,
                expected_size
            )));
        }
        let array = ArrayD::from_shape_vec(IxDyn(shape), data)
            .map_err(|e| OnnxError::InvalidModel(format!("Failed to create tensor: {}", e)))?;
        Ok(Self { data: array })
    }

    /// Create a tensor of zeros with the given shape
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: ArrayD::zeros(IxDyn(shape)),
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the data as a flat slice
    pub fn as_slice(&self) -> Option<&[f32]> {
        self.data.as_slice()
    }

    /// Get the data as a flat vector
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().copied().collect()
    }
}

/// CPU-based interpreter for .holo models
pub struct Interpreter<'a> {
    /// Reference to the loaded model
    model: &'a HoloModel,
    /// Tensor storage indexed by node ID
    tensors: HashMap<usize, Tensor>,
    constant_weight_ids: HashMap<usize, usize>,
    packed_weight_cache: HashMap<usize, Arc<Vec<f32>>>,
    runtime_packed_matmul_rhs: HashMap<usize, Arc<Vec<f32>>>,
    runtime_packed_conv2d: HashMap<usize, Arc<Vec<f32>>>,
    /// Flag indicating if the model has been executed
    executed: bool,
    /// Optional hologram backend executor for accelerated ops.
    backend: Option<BackendContext>,
}

impl<'a> Interpreter<'a> {
    /// Create a new interpreter for the given model
    pub fn new(model: &'a HoloModel) -> Result<Self> {
        let constant_weight_ids = build_constant_weight_map(model);
        Ok(Self {
            model,
            tensors: HashMap::new(),
            constant_weight_ids,
            packed_weight_cache: HashMap::new(),
            runtime_packed_matmul_rhs: HashMap::new(),
            runtime_packed_conv2d: HashMap::new(),
            executed: false,
            backend: None,
        })
    }

    /// Create a new interpreter using hologram backend acceleration.
    pub fn new_with_backend(model: &'a HoloModel, backend: RuntimeBackend) -> Result<Self> {
        let constant_weight_ids = build_constant_weight_map(model);
        Ok(Self {
            model,
            tensors: HashMap::new(),
            constant_weight_ids,
            packed_weight_cache: HashMap::new(),
            runtime_packed_matmul_rhs: HashMap::new(),
            runtime_packed_conv2d: HashMap::new(),
            executed: false,
            backend: Some(BackendContext::new(backend)?),
        })
    }

    /// Set an input tensor by name
    pub fn set_input(&mut self, name: &str, data: &[f32]) -> Result<()> {
        // Find the input node
        for node in &self.model.graph.nodes {
            if let SerNodeKind::Input { name: input_name } = &node.node
                && input_name == name
            {
                // Get shape from node
                let shape = self.get_concrete_shape(node)?;
                let tensor = Tensor::from_vec(data.to_vec(), &shape)?;
                self.tensors.insert(node.id, tensor);
                debug!(
                    "Set input '{}' (node {}) with shape {:?}",
                    name, node.id, shape
                );
                return Ok(());
            }
        }
        Err(OnnxError::InvalidModel(format!(
            "Input '{}' not found",
            name
        )))
    }

    /// Set an input tensor by node ID
    pub fn set_input_by_id(&mut self, node_id: usize, data: &[f32], shape: &[usize]) -> Result<()> {
        let tensor = Tensor::from_vec(data.to_vec(), shape)?;
        self.tensors.insert(node_id, tensor);
        debug!("Set input node {} with shape {:?}", node_id, shape);
        Ok(())
    }

    /// Run inference
    pub fn run(&mut self) -> Result<()> {
        debug!(
            "Running inference on {} nodes",
            self.model.graph.nodes.len()
        );

        // Execute nodes in order (assumes topological sort)
        let total_nodes = self.model.graph.nodes.len();
        let log_interval = 50usize;
        let slow_node_threshold = Duration::from_secs(1);
        let start = Instant::now();

        for (index, node) in self.model.graph.nodes.iter().enumerate() {
            let node_start = Instant::now();
            self.execute_node(node)?;

            let node_elapsed = node_start.elapsed();
            if node_elapsed >= slow_node_threshold {
                let (kind_label, input_shapes, output_shape) = self.slow_node_details(node);
                info!(
                    "Interpreter slow node: {}/{} (id {}, {}) took {:?} inputs={:?} output={:?}",
                    index + 1,
                    total_nodes,
                    node.id,
                    kind_label,
                    node_elapsed,
                    input_shapes,
                    output_shape
                );
            }

            if total_nodes > 0
                && ((index + 1) % log_interval == 0 || index + 1 == total_nodes)
            {
                let elapsed = start.elapsed();
                let percent = ((index + 1) as f64 / total_nodes as f64) * 100.0;
                info!(
                    "Interpreter progress: {}/{} ({:.1}%) nodes in {:?}",
                    index + 1,
                    total_nodes,
                    percent,
                    elapsed
                );
            }
        }

        self.executed = true;
        info!("Interpreter inference complete in {:?}", start.elapsed());
        Ok(())
    }

    fn slow_node_details(
        &self,
        node: &SerNode,
    ) -> (String, Vec<Vec<usize>>, Option<Vec<usize>>) {
        let kind_label = self.node_kind_label(&node.node);
        let input_ids = self.node_input_ids(&node.node);
        let input_shapes = input_ids
            .iter()
            .filter_map(|id| self.tensors.get(id).map(|tensor| tensor.shape().to_vec()))
            .collect();
        let output_shape = self
            .tensors
            .get(&node.id)
            .map(|tensor| tensor.shape().to_vec());
        (kind_label, input_shapes, output_shape)
    }

    fn node_kind_label(&self, node: &SerNodeKind) -> String {
        match node {
            SerNodeKind::Input { .. } => "Input".to_string(),
            SerNodeKind::Constant { .. } => "Constant".to_string(),
            SerNodeKind::ScalarConst { .. } => "ScalarConst".to_string(),
            SerNodeKind::WeightRef { .. } => "WeightRef".to_string(),
            SerNodeKind::BinaryOp { op, .. } => format!("BinaryOp:{}", op),
            SerNodeKind::UnaryOp { op, .. } => format!("UnaryOp:{}", op),
            SerNodeKind::MatMul { .. } => "MatMul".to_string(),
            SerNodeKind::Softmax { .. } => "Softmax".to_string(),
            SerNodeKind::Reshape { .. } => "Reshape".to_string(),
            SerNodeKind::Transpose { .. } => "Transpose".to_string(),
            SerNodeKind::Broadcast { .. } => "Broadcast".to_string(),
            SerNodeKind::Slice { .. } => "Slice".to_string(),
            SerNodeKind::Gather { .. } => "Gather".to_string(),
            SerNodeKind::Concat { .. } => "Concat".to_string(),
            SerNodeKind::Reduce { op, .. } => format!("Reduce:{}", op),
            SerNodeKind::Select { .. } => "Select".to_string(),
            SerNodeKind::Phi { .. } => "Phi".to_string(),
            SerNodeKind::Conv2D { .. } => "Conv2D".to_string(),
            SerNodeKind::BatchNorm { .. } => "BatchNorm".to_string(),
            SerNodeKind::MaxPool { .. } => "MaxPool".to_string(),
            SerNodeKind::AvgPool { .. } => "AvgPool".to_string(),
            SerNodeKind::Cast { .. } => "Cast".to_string(),
            SerNodeKind::Call { func, .. } => format!("Call:{}", func),
            SerNodeKind::Im2Col { .. } => "Im2Col".to_string(),
            SerNodeKind::Col2Im { .. } => "Col2Im".to_string(),
            SerNodeKind::Unfold { .. } => "Unfold".to_string(),
            SerNodeKind::Stack { .. } => "Stack".to_string(),
            SerNodeKind::VStack { .. } => "VStack".to_string(),
            SerNodeKind::HStack { .. } => "HStack".to_string(),
        }
    }

    fn node_input_ids(&self, node: &SerNodeKind) -> Vec<usize> {
        match node {
            SerNodeKind::Input { .. }
            | SerNodeKind::Constant { .. }
            | SerNodeKind::ScalarConst { .. }
            | SerNodeKind::WeightRef { .. } => Vec::new(),
            SerNodeKind::BinaryOp { lhs, rhs, .. } => vec![*lhs, *rhs],
            SerNodeKind::UnaryOp { operand, .. } => vec![*operand],
            SerNodeKind::MatMul { lhs, rhs } => vec![*lhs, *rhs],
            SerNodeKind::Softmax { input, .. } => vec![*input],
            SerNodeKind::Reshape { input, .. } => vec![*input],
            SerNodeKind::Transpose { input, .. } => vec![*input],
            SerNodeKind::Broadcast { input, .. } => vec![*input],
            SerNodeKind::Slice { input, .. } => vec![*input],
            SerNodeKind::Gather { input, indices, .. } => vec![*input, *indices],
            SerNodeKind::Concat { inputs, .. } => inputs.clone(),
            SerNodeKind::Reduce { input, .. } => vec![*input],
            SerNodeKind::Select {
                cond,
                on_true,
                on_false,
            } => vec![*cond, *on_true, *on_false],
            SerNodeKind::Phi { inputs } => inputs.clone(),
            SerNodeKind::Conv2D {
                input,
                weight,
                bias,
                ..
            } => {
                let mut ids = vec![*input, *weight];
                if let Some(bias_id) = bias {
                    ids.push(*bias_id);
                }
                ids
            }
            SerNodeKind::BatchNorm {
                input,
                scale,
                bias,
                mean,
                var,
                ..
            } => vec![*input, *scale, *bias, *mean, *var],
            SerNodeKind::MaxPool { input, .. } => vec![*input],
            SerNodeKind::AvgPool { input, .. } => vec![*input],
            SerNodeKind::Cast { input, .. } => vec![*input],
            SerNodeKind::Call { args, .. } => args.clone(),
            SerNodeKind::Im2Col { input, .. } => vec![*input],
            SerNodeKind::Col2Im { input, .. } => vec![*input],
            SerNodeKind::Unfold { input, .. } => vec![*input],
            SerNodeKind::Stack { inputs, .. } => inputs.clone(),
            SerNodeKind::VStack { inputs, .. } => inputs.clone(),
            SerNodeKind::HStack { inputs, .. } => inputs.clone(),
        }
    }

    fn node_debug_info(&self, node_id: usize) -> Option<String> {
        let node = self.model.graph.nodes.iter().find(|node| node.id == node_id)?;
        let shape = self.get_concrete_shape(node).ok();
        let extra = match &node.node {
            SerNodeKind::Reshape { input, shape } => {
                let input_info = self.node_debug_info(*input);
                Some(format!(
                    "reshape={:?}, input={:?}",
                    shape, input_info
                ))
            }
            SerNodeKind::Transpose { input, perm } => {
                let input_info = self.node_debug_info(*input);
                Some(format!("transpose={:?}, input={:?}", perm, input_info))
            }
            SerNodeKind::Im2Col {
                kernel,
                stride,
                padding,
                dilation,
                ..
            } => Some(format!(
                "im2col(kernel={:?}, stride={:?}, padding={:?}, dilation={:?})",
                kernel, stride, padding, dilation
            )),
            _ => None,
        };

        Some(match extra {
            Some(extra) => format!(
                "kind={}, shape={:?}, {}",
                self.node_kind_label(&node.node),
                shape,
                extra
            ),
            None => format!(
                "kind={}, shape={:?}",
                self.node_kind_label(&node.node),
                shape
            ),
        })
    }

    /// Get output tensor by index
    pub fn get_output(&self, index: usize) -> Result<&Tensor> {
        if !self.executed {
            return Err(OnnxError::InvalidModel("Model not yet executed".into()));
        }

        let output_id = self.model.graph.outputs.get(index).ok_or_else(|| {
            OnnxError::InvalidModel(format!(
                "Output index {} out of range (have {} outputs)",
                index,
                self.model.graph.outputs.len()
            ))
        })?;

        self.tensors.get(output_id).ok_or_else(|| {
            OnnxError::InvalidModel(format!("Output node {} not found in tensors", output_id))
        })
    }

    /// Get all output tensors
    pub fn get_outputs(&self) -> Result<Vec<&Tensor>> {
        if !self.executed {
            return Err(OnnxError::InvalidModel("Model not yet executed".into()));
        }

        self.model
            .graph
            .outputs
            .iter()
            .map(|id| {
                self.tensors
                    .get(id)
                    .ok_or_else(|| OnnxError::InvalidModel(format!("Output node {} not found", id)))
            })
            .collect()
    }

    /// Execute a single node
    fn execute_node(&mut self, node: &SerNode) -> Result<()> {
        trace!(
            "Executing node {}: {:?}",
            node.id,
            std::mem::discriminant(&node.node)
        );

        let result = match &node.node {
            SerNodeKind::Input { .. } => {
                // Inputs should already be set
                if !self.tensors.contains_key(&node.id) {
                    return Err(OnnxError::InvalidModel(format!(
                        "Input node {} not set",
                        node.id
                    )));
                }
                return Ok(());
            }

            SerNodeKind::Constant { weight_id } => self.load_constant(*weight_id, node)?,

            SerNodeKind::ScalarConst { value } => {
                // Create a scalar tensor from the inline value
                Tensor {
                    data: ArrayD::from_elem(IxDyn(&[]), *value as f32),
                }
            }

            SerNodeKind::BinaryOp { op, lhs, rhs } => self.execute_binary_op(op, *lhs, *rhs)?,

            SerNodeKind::UnaryOp { op, operand } => self.execute_unary_op(op, *operand)?,

            SerNodeKind::MatMul { lhs, rhs } => self.execute_matmul(node.id, *lhs, *rhs)?,

            SerNodeKind::Reshape { input, shape } => self.execute_reshape(node.id, *input, shape)?,

            SerNodeKind::Transpose { input, perm } => self.execute_transpose(*input, perm)?,

            SerNodeKind::Reduce {
                op,
                input,
                axes,
                keepdims,
            } => self.execute_reduce(op, *input, axes, *keepdims)?,

            SerNodeKind::Im2Col {
                input,
                kernel,
                stride,
                padding,
                dilation,
            } => self.execute_im2col(*input, kernel, stride, padding, dilation)?,

            SerNodeKind::Unfold {
                input,
                kernel,
                stride,
                padding,
                ..
            } => self.execute_unfold(*input, kernel, stride, padding)?,

            SerNodeKind::Softmax { input, axis } => self.execute_softmax(*input, *axis)?,

            SerNodeKind::Concat { inputs, axis } => self.execute_concat(inputs, *axis)?,

            SerNodeKind::Gather {
                input,
                indices,
                axis,
            } => self.execute_gather(*input, *indices, *axis)?,

            SerNodeKind::Slice { input, ranges } => self.execute_slice(*input, ranges)?,

            SerNodeKind::Broadcast { input, shape } => self.execute_broadcast(*input, shape)?,

            SerNodeKind::Select {
                cond,
                on_true,
                on_false,
            } => self.execute_select(*cond, *on_true, *on_false)?,

            SerNodeKind::Conv2D {
                input,
                weight,
                bias,
                stride,
                padding,
                dilation,
                groups,
            } => self.execute_conv2d(*input, *weight, *bias, stride, padding, dilation, *groups)?,

            SerNodeKind::MaxPool {
                input,
                kernel,
                stride,
                padding,
            } => self.execute_maxpool(*input, kernel, stride, padding)?,

            SerNodeKind::AvgPool {
                input,
                kernel,
                stride,
                padding,
            } => self.execute_avgpool(*input, kernel, stride, padding)?,

            SerNodeKind::BatchNorm {
                input,
                scale,
                bias,
                mean,
                var,
                epsilon,
            } => self.execute_batchnorm(*input, *scale, *bias, *mean, *var, *epsilon)?,

            SerNodeKind::Cast { input, dtype } => self.execute_cast(*input, dtype)?,

            SerNodeKind::Stack { inputs, axis } => self.execute_stack(inputs, *axis)?,

            SerNodeKind::VStack { inputs } => self.execute_vstack(inputs)?,

            SerNodeKind::HStack { inputs } => self.execute_hstack(inputs)?,

            // Not implemented yet
            SerNodeKind::WeightRef { .. } => {
                return Err(OnnxError::unsupported_op("WeightRef", 0));
            }
            SerNodeKind::Phi { .. } => {
                return Err(OnnxError::unsupported_op("Phi", 0));
            }
            SerNodeKind::Call { func, args } => self.execute_call(node.id, func, args)?,
            SerNodeKind::Col2Im { .. } => {
                return Err(OnnxError::unsupported_op("Col2Im", 0));
            }
        };

        self.tensors.insert(node.id, result);
        Ok(())
    }

    /// Load a constant weight tensor
    fn load_constant(&self, weight_id: usize, _node: &SerNode) -> Result<Tensor> {
        let entry =
            self.model.weight_entries.get(weight_id).ok_or_else(|| {
                OnnxError::InvalidModel(format!("Weight {} not found", weight_id))
            })?;

        let data = self.model.get_weight(weight_id).ok_or_else(|| {
            OnnxError::InvalidModel(format!("Weight {} data not available", weight_id))
        })?;

        // Convert bytes to f32 based on dtype
        let floats = match entry.dtype.as_str() {
            "f32" => bytes_to_f32(data),
            "f64" => bytes_to_f64_as_f32(data),
            "i64" => bytes_to_i64_as_f32(data),
            "i32" => bytes_to_i32_as_f32(data),
            _ => {
                debug!("Unknown dtype '{}', treating as f32", entry.dtype);
                bytes_to_f32(data)
            }
        };

        let shape = &entry.shape;
        trace!("Loaded constant {} with shape {:?}", weight_id, shape);

        Tensor::from_vec(floats, shape)
    }

    /// Get concrete shape from node, resolving symbolic dims
    fn get_concrete_shape(&self, node: &SerNode) -> Result<Vec<usize>> {
        let shape = node
            .shape
            .as_ref()
            .ok_or_else(|| OnnxError::InvalidModel(format!("Node {} has no shape", node.id)))?;

        shape
            .iter()
            .map(|d| match d {
                DimSpec::Concrete(n) => Ok(*n),
                DimSpec::Symbolic(s) => {
                    // Default to 1 for unknown symbolic dims
                    debug!("Symbolic dim '{}' defaulting to 1", s);
                    Ok(1)
                }
            })
            .collect()
    }

    /// Execute binary operation
    fn execute_binary_op(&self, op: &str, lhs_id: usize, rhs_id: usize) -> Result<Tensor> {
        let lhs = self.get_tensor(lhs_id)?;
        let rhs = self.get_tensor(rhs_id)?;

        let result = match op {
            "add" => &lhs.data + &rhs.data,
            "sub" => &lhs.data - &rhs.data,
            "mul" => &lhs.data * &rhs.data,
            "div" => &lhs.data / &rhs.data,
            "pow" => lhs.data.mapv(|x| x.powf(rhs.data[[]])),
            "min" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| a.min(b)),
            "max" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| a.max(b)),
            "eq" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if (a - b).abs() < 1e-6 { 1.0 } else { 0.0 }),
            "ne" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if (a - b).abs() >= 1e-6 { 1.0 } else { 0.0 }),
            "lt" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a < b { 1.0 } else { 0.0 }),
            "le" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a <= b { 1.0 } else { 0.0 }),
            "gt" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a > b { 1.0 } else { 0.0 }),
            "ge" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a >= b { 1.0 } else { 0.0 }),
            "and" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a != 0.0 && b != 0.0 { 1.0 } else { 0.0 }),
            "or" => ndarray::Zip::from(&lhs.data)
                .and(&rhs.data)
                .map_collect(|&a, &b| if a != 0.0 || b != 0.0 { 1.0 } else { 0.0 }),
            _ => return Err(OnnxError::unsupported_op(format!("BinaryOp::{}", op), 0)),
        };

        Ok(Tensor { data: result })
    }

    /// Execute unary operation
    fn execute_unary_op(&self, op: &str, operand_id: usize) -> Result<Tensor> {
        let operand = self.get_tensor(operand_id)?;

        let result = match op {
            "neg" => operand.data.mapv(|x| -x),
            "abs" => operand.data.mapv(|x| x.abs()),
            "not" => operand.data.mapv(|x| if x == 0.0 { 1.0 } else { 0.0 }),
            "sqrt" => operand.data.mapv(|x| x.sqrt()),
            "rsqrt" => operand.data.mapv(|x| 1.0 / x.sqrt()),
            "exp" => operand.data.mapv(|x| x.exp()),
            "log" => operand.data.mapv(|x| x.ln()),
            "sin" => operand.data.mapv(|x| x.sin()),
            "cos" => operand.data.mapv(|x| x.cos()),
            "tan" => operand.data.mapv(|x| x.tan()),
            "floor" => operand.data.mapv(|x| x.floor()),
            "ceil" => operand.data.mapv(|x| x.ceil()),
            "round" => operand.data.mapv(|x| x.round()),
            "sigmoid" => operand.data.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            "tanh" => operand.data.mapv(|x| x.tanh()),
            "relu" => operand.data.mapv(|x| x.max(0.0)),
            "gelu" => operand.data.mapv(|x| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
            _ => return Err(OnnxError::unsupported_op(format!("UnaryOp::{}", op), 0)),
        };

        Ok(Tensor { data: result })
    }

    /// Execute matrix multiplication
    fn execute_matmul(&mut self, node_id: usize, lhs_id: usize, rhs_id: usize) -> Result<Tensor> {
        let (lhs_shape, rhs_shape) = {
            let lhs = self.get_tensor(lhs_id)?;
            let rhs = self.get_tensor(rhs_id)?;
            (lhs.shape().to_vec(), rhs.shape().to_vec())
        };
        let rhs_weight_id = if rhs_shape.len() == 2 {
            self.constant_weight_ids.get(&rhs_id).copied()
        } else {
            None
        };
        let mut rhs_storage = rhs_weight_id
            .and_then(|weight_id| self.packed_weight_vec(weight_id, PackedWeightKind::MatMulRhs));

        // Handle different dimension cases
        let result = if lhs_shape.len() == 2 && rhs_shape.len() == 2 {
            let m = lhs_shape[0];
            let k = lhs_shape[1];
            let n = rhs_shape[1];

            if k != rhs_shape[0] {
                let lhs_info = self.node_debug_info(lhs_id);
                let rhs_info = self.node_debug_info(rhs_id);
                return Err(OnnxError::InvalidModel(format!(
                    "MatMul node {} dimension mismatch: {:?} x {:?} (lhs {:?}, rhs {:?})",
                    node_id, lhs_shape, rhs_shape, lhs_info, rhs_info
                )));
            }

            let (lhs_vec, mut rhs_vec_fallback) = {
                let lhs = self.get_tensor(lhs_id)?;
                let rhs = self.get_tensor(rhs_id)?;
                let lhs_vec = self.contiguous_vec(&lhs.data)?;
                let rhs_vec_fallback = if rhs_storage.is_some() {
                    None
                } else {
                    Some(self.contiguous_vec(&rhs.data)?)
                };
                (lhs_vec, rhs_vec_fallback)
            };
            if rhs_storage.is_none() {
                if let (Some(weight_id), Some(rhs_vec)) = (rhs_weight_id, rhs_vec_fallback.take())
                {
                    rhs_storage = Some(self.runtime_pack_matmul_rhs(weight_id, rhs_vec)?);
                }
            }
            let rhs_vec_fallback = if rhs_storage.is_none() && rhs_vec_fallback.is_none() {
                let rhs = self.get_tensor(rhs_id)?;
                Some(self.contiguous_vec(&rhs.data)?)
            } else {
                rhs_vec_fallback
            };
            let rhs_vec = match (&rhs_storage, &rhs_vec_fallback) {
                (Some(storage), _) => storage.as_slice(),
                (None, Some(vec)) => vec.as_slice(),
                (None, None) => {
                    return Err(OnnxError::InvalidModel(
                        "MatMul rhs data missing".into(),
                    ))
                }
            };
            let output = if self.backend.is_some() {
                self.matmul_2d_backend(&lhs_vec, rhs_vec, m, k, n)?
            } else {
                Self::matmul_2d_sgemm(&lhs_vec, rhs_vec, m, k, n)?
            };
            ArrayD::from_shape_vec(IxDyn(&[m, n]), output)
                .map_err(|e| OnnxError::InvalidModel(format!("MatMul output shape error: {}", e)))?
        } else if lhs_shape.len() > 2 || rhs_shape.len() > 2 {
            let (lhs_batch, lhs_m, lhs_k) = self.split_batch_matmul_shape(&lhs_shape)?;
            let (rhs_batch, rhs_k, rhs_n) = self.split_batch_matmul_shape(&rhs_shape)?;

            if lhs_k != rhs_k {
                let lhs_info = self.node_debug_info(lhs_id);
                let rhs_info = self.node_debug_info(rhs_id);
                return Err(OnnxError::InvalidModel(format!(
                    "MatMul node {} dimension mismatch: {:?} x {:?} (lhs {:?}, rhs {:?})",
                    node_id, lhs_shape, rhs_shape, lhs_info, rhs_info
                )));
            }

            let batch_dims = if rhs_batch.is_empty() {
                lhs_batch.clone()
            } else if lhs_batch == rhs_batch {
                lhs_batch.clone()
            } else {
                let lhs_info = self.node_debug_info(lhs_id);
                let rhs_info = self.node_debug_info(rhs_id);
                return Err(OnnxError::InvalidModel(format!(
                    "MatMul node {} batch dimension mismatch: {:?} x {:?} (lhs {:?}, rhs {:?})",
                    node_id, lhs_shape, rhs_shape, lhs_info, rhs_info
                )));
            };

            let batch_size: usize = if batch_dims.is_empty() {
                1
            } else {
                batch_dims.iter().product()
            };

            let (lhs_vec, mut rhs_vec_fallback) = {
                let lhs = self.get_tensor(lhs_id)?;
                let rhs = self.get_tensor(rhs_id)?;
                let lhs_vec = self.contiguous_vec(&lhs.data)?;
                let rhs_vec_fallback = if rhs_storage.is_some() {
                    None
                } else {
                    Some(self.contiguous_vec(&rhs.data)?)
                };
                (lhs_vec, rhs_vec_fallback)
            };
            if rhs_storage.is_none() {
                if let (Some(weight_id), Some(rhs_vec)) = (rhs_weight_id, rhs_vec_fallback.take())
                {
                    rhs_storage = Some(self.runtime_pack_matmul_rhs(weight_id, rhs_vec)?);
                }
            }

            let lhs_stride = lhs_m * lhs_k;
            let rhs_stride = rhs_k * rhs_n;
            let out_stride = lhs_m * rhs_n;

            let rhs_vec_fallback = if rhs_storage.is_none() && rhs_vec_fallback.is_none() {
                let rhs = self.get_tensor(rhs_id)?;
                Some(self.contiguous_vec(&rhs.data)?)
            } else {
                rhs_vec_fallback
            };
            let rhs_buffer = match (&rhs_storage, &rhs_vec_fallback) {
                (Some(storage), _) => Some(storage.as_slice()),
                (None, Some(vec)) => Some(vec.as_slice()),
                (None, None) => None,
            }
            .ok_or_else(|| OnnxError::InvalidModel("MatMul rhs data missing".into()))?;

            let output = if self.backend.is_some() && batch_size == 1 {
                self.matmul_2d_backend(&lhs_vec, rhs_buffer, lhs_m, lhs_k, rhs_n)?
            } else {
                let mut output = vec![0.0f32; batch_size * out_stride];
                output
                    .par_chunks_mut(out_stride)
                    .enumerate()
                    .for_each(|(b, out_plane)| {
                        let lhs_offset = b * lhs_stride;
                        let rhs_offset = if rhs_batch.is_empty() {
                            0
                        } else {
                            b * rhs_stride
                        };

                        let lhs_slice = &lhs_vec[lhs_offset..lhs_offset + lhs_stride];
                        let rhs_slice = &rhs_buffer[rhs_offset..rhs_offset + rhs_stride];

                        Self::matmul_2d_sgemm_into(
                            lhs_slice,
                            rhs_slice,
                            lhs_m,
                            lhs_k,
                            rhs_n,
                            out_plane,
                        );
                    });
                output
            };

            let mut out_shape = batch_dims;
            out_shape.push(lhs_m);
            out_shape.push(rhs_n);

            ArrayD::from_shape_vec(IxDyn(&out_shape), output)
                .map_err(|e| OnnxError::InvalidModel(format!("MatMul output shape error: {}", e)))?
        } else {
            return Err(OnnxError::unsupported_op(
                format!("MatMul with shapes {:?}x{:?}", lhs_shape, rhs_shape),
                0,
            ));
        };

        Ok(Tensor { data: result })
    }

    fn split_batch_matmul_shape(
        &self,
        shape: &[usize],
    ) -> Result<(Vec<usize>, usize, usize)> {
        if shape.len() < 2 {
            return Err(OnnxError::InvalidModel(
                "MatMul expects tensors with at least 2 dims".into(),
            ));
        }
        let m = shape[shape.len() - 2];
        let k = shape[shape.len() - 1];
        let batch = shape[..shape.len() - 2].to_vec();
        Ok((batch, m, k))
    }

    fn contiguous_vec(&self, arr: &ArrayD<f32>) -> Result<Vec<f32>> {
        if arr.is_standard_layout() {
            return Ok(
                arr.as_slice()
                    .map(|slice| slice.to_vec())
                    .unwrap_or_else(|| arr.iter().copied().collect()),
            );
        }

        let shape = arr.shape();
        let owned = ArrayD::from_shape_vec(IxDyn(shape), arr.iter().copied().collect()).map_err(|e| {
            OnnxError::InvalidModel(format!("Failed to make contiguous: {}", e))
        })?;
        Ok(owned
            .as_slice()
            .map(|slice| slice.to_vec())
            .unwrap_or_else(|| owned.iter().copied().collect()))
    }

    fn packed_weight_vec(
        &mut self,
        weight_id: usize,
        kind: PackedWeightKind,
    ) -> Option<Arc<Vec<f32>>> {
        let entry = self.model.packed_weight_entry(weight_id, kind)?;
        if let Some(cached) = self.packed_weight_cache.get(&entry.id) {
            return Some(cached.clone());
        }

        let bytes = self.model.get_packed_weight(entry)?;
        if bytes.len() % 4 != 0 {
            return None;
        }
        let vec = bytes_to_f32(bytes);
        let arc = Arc::new(vec);
        self.packed_weight_cache.insert(entry.id, arc.clone());
        Some(arc)
    }

    fn runtime_pack_matmul_rhs(
        &mut self,
        weight_id: usize,
        rhs_vec: Vec<f32>,
    ) -> Result<Arc<Vec<f32>>> {
        if let Some(cached) = self.runtime_packed_matmul_rhs.get(&weight_id) {
            return Ok(cached.clone());
        }
        let arc = Arc::new(rhs_vec);
        self.runtime_packed_matmul_rhs
            .insert(weight_id, arc.clone());
        Ok(arc)
    }

    fn runtime_pack_conv2d_weight(
        &mut self,
        weight_id: usize,
        weight_vec: &[f32],
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
    ) -> Result<Arc<Vec<f32>>> {
        if let Some(cached) = self.runtime_packed_conv2d.get(&weight_id) {
            return Ok(cached.clone());
        }

        if weight_vec.len() != c_out * c_in * kh * kw {
            return Err(OnnxError::InvalidModel(format!(
                "Conv2D weight size mismatch: expected {}, got {}",
                c_out * c_in * kh * kw,
                weight_vec.len()
            )));
        }
        let mut packed = vec![0.0f32; c_out * c_in * kh * kw];
        for oc in 0..c_out {
            let mut k_idx = 0;
            for ic in 0..c_in {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let src_idx = ((oc * c_in + ic) * kh + ky) * kw + kx;
                        packed[k_idx * c_out + oc] = weight_vec[src_idx];
                        k_idx += 1;
                    }
                }
            }
        }

        let arc = Arc::new(packed);
        self.runtime_packed_conv2d.insert(weight_id, arc.clone());
        Ok(arc)
    }

    fn pack_conv2d_weight_from_tensor(
        &self,
        weight_vec: &[f32],
        c_out: usize,
        c_in: usize,
        kh: usize,
        kw: usize,
    ) -> Result<Vec<f32>> {
        if weight_vec.len() != c_out * c_in * kh * kw {
            return Err(OnnxError::InvalidModel(format!(
                "Conv2D weight size mismatch: expected {}, got {}",
                c_out * c_in * kh * kw,
                weight_vec.len()
            )));
        }
        let mut packed = vec![0.0f32; c_out * c_in * kh * kw];
        for oc in 0..c_out {
            let mut k_idx = 0;
            for ic in 0..c_in {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let src_idx = ((oc * c_in + ic) * kh + ky) * kw + kx;
                        packed[k_idx * c_out + oc] = weight_vec[src_idx];
                        k_idx += 1;
                    }
                }
            }
        }
        Ok(packed)
    }

    fn matmul_2d_sgemm(
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; m * n];
        Self::matmul_2d_sgemm_into(lhs, rhs, m, k, n, &mut output);
        Ok(output)
    }

    fn matmul_2d_backend(
        &mut self,
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let backend = self.backend.as_mut().ok_or_else(|| {
            OnnxError::InternalError("Backend executor missing for MatMul".into())
        })?;
        let exec = &mut backend.exec;

        let mut buf_a: Buffer<f32> = exec
            .allocate(m * k)
            .map_err(|e| OnnxError::HologramError(format!("MatMul allocate A failed: {}", e)))?;
        let mut buf_b: Buffer<f32> = exec
            .allocate(k * n)
            .map_err(|e| OnnxError::HologramError(format!("MatMul allocate B failed: {}", e)))?;
        let mut buf_c: Buffer<f32> = exec
            .allocate(m * n)
            .map_err(|e| OnnxError::HologramError(format!("MatMul allocate C failed: {}", e)))?;

        buf_a
            .copy_from_slice(exec, lhs)
            .map_err(|e| OnnxError::HologramError(format!("MatMul copy A failed: {}", e)))?;
        buf_b
            .copy_from_slice(exec, rhs)
            .map_err(|e| OnnxError::HologramError(format!("MatMul copy B failed: {}", e)))?;

        hologram_core::ops::linalg::gemm(exec, &buf_a, &buf_b, &mut buf_c, m, k, n)
            .map_err(|e| OnnxError::HologramError(format!("MatMul GEMM failed: {}", e)))?;

        let mut output = vec![0.0f32; m * n];
        buf_c
            .copy_to_slice(exec, &mut output)
            .map_err(|e| OnnxError::HologramError(format!("MatMul copy C failed: {}", e)))?;
        Ok(output)
    }

    fn matmul_2d_sgemm_into(
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
        output: &mut [f32],
    ) {
        let k_i = k as isize;
        let n_i = n as isize;

        unsafe {
            sgemm(
                m,
                k,
                n,
                1.0,
                lhs.as_ptr(),
                k_i,
                1,
                rhs.as_ptr(),
                n_i,
                1,
                0.0,
                output.as_mut_ptr(),
                n_i,
                1,
            );
        }
    }

    /// Execute reshape
    fn execute_reshape(
        &self,
        node_id: usize,
        input_id: usize,
        shape: &[DimSpec],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let input_shape = input.shape();
        let input_size: usize = input_shape.iter().product();

        let mut new_shape = Vec::with_capacity(shape.len());
        let mut symbolic_indices = Vec::new();

        for (i, dim) in shape.iter().enumerate() {
            match dim {
                DimSpec::Concrete(n) => {
                    if *n == 0 {
                        let size = *input_shape.get(i).ok_or_else(|| {
                            OnnxError::InvalidModel(format!(
                                "Reshape dimension {} is out of range for input shape {:?}",
                                i, input_shape
                            ))
                        })?;
                        new_shape.push(size);
                    } else {
                        new_shape.push(*n);
                    }
                }
                DimSpec::Symbolic(s) => {
                    if let Ok(n) = s.parse::<usize>() {
                        new_shape.push(n);
                    } else {
                        symbolic_indices.push(i);
                        new_shape.push(0);
                    }
                }
            }
        }

        if !symbolic_indices.is_empty() {
            let concrete_product: usize = new_shape.iter().filter(|&&x| x != 0).product();
            if symbolic_indices.len() == 1 {
                let idx = symbolic_indices[0];
                if concrete_product == 0 || !input_size.is_multiple_of(concrete_product) {
                    return Err(OnnxError::InvalidModel(format!(
                        "Reshape cannot infer dimension for shape {:?} with input size {}",
                        new_shape, input_size
                    )));
                }
                new_shape[idx] = input_size / concrete_product;
            } else if new_shape.len() == 2 {
                let batch = input_shape.first().copied().unwrap_or(1);
                new_shape[0] = batch;
                new_shape[1] = input_size / batch;
            } else if new_shape.len() == input_shape.len() {
                for &idx in &symbolic_indices {
                    new_shape[idx] = input_shape[idx];
                }
            } else {
                return Err(OnnxError::InvalidModel(format!(
                    "Reshape with symbolic dims {:?} cannot be inferred from input {:?}",
                    new_shape, input_shape
                )));
            }
        }

        let new_size: usize = new_shape.iter().product();
        if input_size != new_size {
            warn!(
                "Reshape node {} size mismatch: input {:?} ({}) -> target {:?} ({})",
                node_id, input_shape, input_size, new_shape, new_size
            );
            return Err(OnnxError::InvalidModel(format!(
                "Reshape node {} size mismatch: input {:?} ({}) -> target {:?} ({})",
                node_id, input_shape, input_size, new_shape, new_size
            )));
        }

        self.reshape_with_shape(input, &new_shape)
    }

    /// Execute transpose
    fn execute_transpose(&self, input_id: usize, perm: &[usize]) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;

        let result = if perm.is_empty() {
            // Reverse all dimensions
            let ndim = input.shape().len();
            let rev_perm: Vec<usize> = (0..ndim).rev().collect();
            input.data.clone().permuted_axes(IxDyn(&rev_perm))
        } else {
            input.data.clone().permuted_axes(IxDyn(perm))
        };

        Ok(Tensor { data: result })
    }

    /// Execute reduce operation
    fn execute_reduce(
        &self,
        op: &str,
        input_id: usize,
        axes: &[i32],
        keepdims: bool,
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let ndim = input.shape().len();

        // Normalize negative axes
        let axes: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (ndim as i32 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();

        if axes.iter().any(|&axis| axis >= ndim) {
            return Err(OnnxError::InvalidModel(format!(
                "Reduce axis out of bounds: axes {:?} for input shape {:?}",
                axes,
                input.shape()
            )));
        }

        // Sort axes in descending order for proper reduction
        let mut sorted_axes = axes.clone();
        sorted_axes.sort_by(|a, b| b.cmp(a));

        let mut result = input.data.clone();

        for axis in sorted_axes {
            if axis >= result.ndim() {
                return Err(OnnxError::InvalidModel(format!(
                    "Reduce axis {} out of bounds for shape {:?}",
                    axis,
                    result.shape()
                )));
            }
            result = match op {
                "sum" => result.sum_axis(Axis(axis)),
                "mean" => result.mean_axis(Axis(axis)).ok_or_else(|| {
                    OnnxError::InvalidModel(format!(
                        "Reduce mean failed for axis {} on shape {:?}",
                        axis,
                        result.shape()
                    ))
                })?,
                "max" => result.map_axis(Axis(axis), |row| {
                    row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                }),
                "min" => result.map_axis(Axis(axis), |row| {
                    row.iter().fold(f32::INFINITY, |a, &b| a.min(b))
                }),
                "prod" => result.map_axis(Axis(axis), |row| row.iter().product()),
                "argmax" => result.map_axis(Axis(axis), |row| {
                    row.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                }),
                "argmin" => result.map_axis(Axis(axis), |row| {
                    row.iter()
                        .enumerate()
                        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                }),
                _ => return Err(OnnxError::unsupported_op(format!("Reduce::{}", op), 0)),
            };
        }

        // Handle keepdims
        if keepdims {
            let mut new_shape: Vec<usize> = input.shape().to_vec();
            for &axis in &axes {
                new_shape[axis] = 1;
            }
            result = result
                .into_shape_with_order(IxDyn(&new_shape))
                .map_err(|e| OnnxError::InvalidModel(format!("Reshape failed: {}", e)))?;
        }

        Ok(Tensor { data: result })
    }

    /// Execute softmax
    fn execute_softmax(&self, input_id: usize, axis: i32) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let ndim = input.shape().len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };

        // Compute max along axis for numerical stability
        let max_vals = input.data.map_axis(Axis(axis), |row| {
            row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        });

        // Subtract max and exponentiate
        let mut exp_vals = input.data.clone();
        for (i, mut slice) in exp_vals.axis_iter_mut(Axis(axis)).enumerate() {
            let max_val = if axis == 0 {
                max_vals[[i]]
            } else {
                // This is simplified - proper implementation needs proper broadcasting
                max_vals.iter().next().copied().unwrap_or(0.0)
            };
            slice.mapv_inplace(|x| (x - max_val).exp());
        }

        // Sum and normalize
        let sum_vals = exp_vals.sum_axis(Axis(axis));
        for (i, mut slice) in exp_vals.axis_iter_mut(Axis(axis)).enumerate() {
            let sum_val = if axis == 0 {
                sum_vals[[i]]
            } else {
                sum_vals.iter().next().copied().unwrap_or(1.0)
            };
            slice.mapv_inplace(|x| x / sum_val);
        }

        Ok(Tensor { data: exp_vals })
    }

    /// Execute im2col (image to column transformation for convolution)
    fn execute_im2col(
        &self,
        input_id: usize,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Im2Col expects 4D input, got {:?}",
                shape
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = (kernel[0], kernel[1]);
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

        let col_vec = self.im2col_to_vec(
            input,
            h_out,
            w_out,
            (kh, kw),
            (sh, sw),
            (ph, pw),
            (dh, dw),
        )?;
        let col_size = c * kh * kw;
        let col = Array::from_shape_vec((n * h_out * w_out, col_size), col_vec).map_err(|e| {
            OnnxError::InvalidModel(format!("Im2Col output shape error: {}", e))
        })?;

        Ok(Tensor {
            data: col.into_dyn(),
        })
    }

    fn im2col_to_vec(
        &self,
        input: &Tensor,
        h_out: usize,
        w_out: usize,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Vec<f32>> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Im2Col expects 4D input, got {:?}",
                shape
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = kernel;
        let (sh, sw) = stride;
        let (ph, pw) = padding;
        let (dh, dw) = dilation;

        let col_size = c * kh * kw;
        let rows = n * h_out * w_out;
        let mut col = vec![0.0f32; rows * col_size];

        col.par_chunks_mut(col_size)
            .enumerate()
            .for_each(|(row, out_row)| {
                let batch = row / (h_out * w_out);
                let idx = row % (h_out * w_out);
                let y = idx / w_out;
                let x = idx % w_out;

                let mut col_idx = 0;
                for channel in 0..c {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let in_y = y * sh + ky * dh;
                            let in_x = x * sw + kx * dw;

                            let val = if in_y >= ph
                                && in_y < h + ph
                                && in_x >= pw
                                && in_x < w + pw
                            {
                                input.data[[batch, channel, in_y - ph, in_x - pw]]
                            } else {
                                0.0
                            };

                            out_row[col_idx] = val;
                            col_idx += 1;
                        }
                    }
                }
            });

        Ok(col)
    }

    /// Execute unfold (similar to im2col for pooling)
    fn execute_unfold(
        &self,
        input_id: usize,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Unfold expects 4D input, got {:?}",
                shape
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = (kernel[0], kernel[1]);
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let mut unfolded = Array::zeros((n, c, h_out, w_out, kh, kw));

        for batch in 0..n {
            for channel in 0..c {
                for y in 0..h_out {
                    for x in 0..w_out {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let in_y = y * sh + ky;
                                let in_x = x * sw + kx;

                                let val =
                                    if in_y >= ph && in_y < h + ph && in_x >= pw && in_x < w + pw {
                                        input.data[[batch, channel, in_y - ph, in_x - pw]]
                                    } else {
                                        0.0
                                    };

                                unfolded[[batch, channel, y, x, ky, kx]] = val;
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor {
            data: unfolded.into_dyn(),
        })
    }

    /// Execute concat
    fn execute_concat(&self, input_ids: &[usize], axis: i32) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(OnnxError::InvalidModel(
                "Concat requires at least one input".into(),
            ));
        }

        let first = self.get_tensor(input_ids[0])?;
        let ndim = first.shape().len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };

        let views: Vec<_> = input_ids
            .iter()
            .map(|&id| self.get_tensor(id).map(|t| t.data.view()))
            .collect::<Result<Vec<_>>>()?;

        let result = ndarray::concatenate(Axis(axis), &views)
            .map_err(|e| OnnxError::InvalidModel(format!("Concat failed: {}", e)))?;

        Ok(Tensor { data: result })
    }

    /// Execute gather
    fn execute_gather(&self, input_id: usize, indices_id: usize, axis: i32) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let indices = self.get_tensor(indices_id)?;
        let ndim = input.shape().len();
        let axis = if axis < 0 {
            (ndim as i32 + axis) as usize
        } else {
            axis as usize
        };

        // Simplified gather - works for common cases
        let idx_flat: Vec<usize> = indices.data.iter().map(|&x| x as usize).collect();

        let result = input.data.select(Axis(axis), &idx_flat);

        Ok(Tensor { data: result })
    }

    /// Execute slice
    fn execute_slice(
        &self,
        input_id: usize,
        ranges: &[(Option<i64>, Option<i64>, Option<i64>)],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let shape = input.shape();

        // Build slice ranges for each dimension
        let mut slice_ranges: Vec<(usize, usize, usize)> = Vec::new();
        for (i, &(start, end, step)) in ranges.iter().enumerate() {
            let dim_size = shape.get(i).copied().unwrap_or(1) as i64;

            let start = start.unwrap_or(0);
            let end = end.unwrap_or(dim_size);
            let step = step.unwrap_or(1);

            // Normalize negative indices
            let start = if start < 0 {
                (dim_size + start) as usize
            } else {
                start as usize
            };
            let end = if end < 0 {
                (dim_size + end) as usize
            } else {
                end as usize
            };
            let step = step as usize;

            slice_ranges.push((start, end, step));
        }

        // Apply slices using select for each dimension
        let mut result = input.data.clone();

        for (i, &(start, end, step)) in slice_ranges.iter().enumerate() {
            // Create indices for this dimension
            let indices: Vec<usize> = (start..end).step_by(step).collect();
            if !indices.is_empty() && indices.len() < result.shape()[i] {
                result = result.select(Axis(i), &indices);
            }
        }

        Ok(Tensor { data: result })
    }

    /// Execute broadcast
    fn execute_broadcast(&self, input_id: usize, shape: &[DimSpec]) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;

        let target_shape: Vec<usize> = shape
            .iter()
            .map(|d| match d {
                DimSpec::Concrete(n) => *n,
                DimSpec::Symbolic(_) => 1,
            })
            .collect();

        let result = input
            .data
            .broadcast(IxDyn(&target_shape))
            .ok_or_else(|| OnnxError::InvalidModel("Broadcast failed".into()))?
            .into_owned();

        Ok(Tensor { data: result })
    }

    /// Execute select (where/conditional)
    fn execute_select(&self, cond_id: usize, true_id: usize, false_id: usize) -> Result<Tensor> {
        let cond = self.get_tensor(cond_id)?;
        let on_true = self.get_tensor(true_id)?;
        let on_false = self.get_tensor(false_id)?;

        let result = ndarray::Zip::from(&cond.data)
            .and(&on_true.data)
            .and(&on_false.data)
            .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f });

        Ok(Tensor { data: result })
    }

    /// Execute conv2d
    #[allow(clippy::too_many_arguments)]
    fn execute_conv2d(
        &mut self,
        input_id: usize,
        weight_id: usize,
        bias_id: Option<usize>,
        stride: &[usize],
        padding: &[usize],
        dilation: &[usize],
        groups: usize,
    ) -> Result<Tensor> {
        let (input_shape, weight_shape) = {
            let input = self.get_tensor(input_id)?;
            let weight = self.get_tensor(weight_id)?;
            (input.shape().to_vec(), weight.shape().to_vec())
        };

        if input_shape.len() != 4 || weight_shape.len() != 4 {
            return Err(OnnxError::InvalidModel("Conv2D expects 4D tensors".into()));
        }

        let (n, _c_in, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (c_out, _c_in_k, kh, kw) = (
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        );

        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);
        let (dh, dw) = (dilation[0], dilation[1]);

        if groups == 0 || c_out % groups != 0 {
            return Err(OnnxError::InvalidModel(format!(
                "Conv2D invalid groups: {} for c_out {}",
                groups, c_out
            )));
        }

        let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
        let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

        let bias_values = if let Some(bias_id) = bias_id {
            let bias = self.get_tensor(bias_id)?;
            let bias_slice = bias.data.as_slice().ok_or_else(|| {
                OnnxError::InvalidModel("Conv2D bias must be contiguous".into())
            })?;
            if bias_slice.len() != c_out {
                return Err(OnnxError::InvalidModel(format!(
                    "Conv2D bias length {} does not match c_out {}",
                    bias_slice.len(),
                    c_out
                )));
            }
            Some(bias_slice.to_vec())
        } else {
            None
        };

        if groups == 1 && dh == 1 && dw == 1 {
            let weight_vec = {
                let weight = self.get_tensor(weight_id)?;
                self.contiguous_vec(&weight.data)?
            };
            let m = n * h_out * w_out;
            let k = _c_in_k * kh * kw;

            let weight_entry_id = self.constant_weight_ids.get(&weight_id).copied();
            let weight_t = if let Some(weight_id) = weight_entry_id {
                if let Some(packed) = self.packed_weight_vec(weight_id, PackedWeightKind::Conv2dIm2Col)
                {
                    packed
                } else {
                    self.runtime_pack_conv2d_weight(
                        weight_id,
                        &weight_vec,
                        c_out,
                        _c_in_k,
                        kh,
                        kw,
                    )?
                }
            } else {
                Arc::new(self.pack_conv2d_weight_from_tensor(
                    &weight_vec,
                    c_out,
                    _c_in_k,
                    kh,
                    kw,
                )?)
            };
            if weight_t.len() != k * c_out {
                return Err(OnnxError::InvalidModel(format!(
                    "Conv2D packed weight size mismatch: expected {}, got {}",
                    k * c_out,
                    weight_t.len()
                )));
            }

            let input = self.get_tensor(input_id)?;
            let col_vec = self.im2col_to_vec(
                input,
                h_out,
                w_out,
                (kh, kw),
                (sh, sw),
                (ph, pw),
                (dh, dw),
            )?;

            let out_vec = if self.backend.is_some() {
                self.matmul_2d_backend(&col_vec, weight_t.as_slice(), m, k, c_out)?
            } else {
                let mut out_vec = vec![0.0f32; m * c_out];
                Self::matmul_2d_sgemm_into(
                    &col_vec,
                    weight_t.as_slice(),
                    m,
                    k,
                    c_out,
                    &mut out_vec,
                );
                out_vec
            };

            let plane = h_out * w_out;
            let mut output = vec![0.0f32; n * c_out * plane];
            output
                .par_chunks_mut(plane)
                .enumerate()
                .for_each(|(index, out_plane)| {
                    let batch = index / c_out;
                    let oc = index % c_out;
                    let bias_val = bias_values
                        .as_ref()
                        .map(|bias| bias[oc])
                        .unwrap_or(0.0);
                    for y in 0..h_out {
                        for x in 0..w_out {
                            let row = batch * plane + y * w_out + x;
                            out_plane[y * w_out + x] = out_vec[row * c_out + oc] + bias_val;
                        }
                    }
                });

            let output = Array::from_shape_vec((n, c_out, h_out, w_out), output).map_err(|e| {
                OnnxError::InvalidModel(format!("Conv2D output shape error: {}", e))
            })?;

            return Ok(Tensor {
                data: output.into_dyn(),
            });
        }

        let input = self.get_tensor(input_id)?;
        let weight = self.get_tensor(weight_id)?;

        let plane = h_out * w_out;
        let mut output = vec![0.0f32; n * c_out * plane];

        output
            .par_chunks_mut(plane)
            .enumerate()
            .for_each(|(index, out_plane)| {
                let batch = index / c_out;
                let oc = index % c_out;
                let oc_group = oc / (c_out / groups);
                let ic_start = oc_group * _c_in_k;
                let ic_end = ic_start + _c_in_k;
                let bias_val = bias_values
                    .as_ref()
                    .map(|bias| bias[oc])
                    .unwrap_or(0.0);

                for y in 0..h_out {
                    for x in 0..w_out {
                        let mut sum = bias_val;

                        for ic in ic_start..ic_end {
                            let ic_k = ic - ic_start;
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let in_y = y * sh + ky * dh;
                                    let in_x = x * sw + kx * dw;

                                    if in_y >= ph
                                        && in_y < h + ph
                                        && in_x >= pw
                                        && in_x < w + pw
                                    {
                                        let val = input.data[[batch, ic, in_y - ph, in_x - pw]];
                                        let w_val = weight.data[[oc, ic_k, ky, kx]];
                                        sum += val * w_val;
                                    }
                                }
                            }
                        }

                        out_plane[y * w_out + x] = sum;
                    }
                }
            });

        let output = Array::from_shape_vec((n, c_out, h_out, w_out), output).map_err(|e| {
            OnnxError::InvalidModel(format!("Conv2D output shape error: {}", e))
        })?;

        Ok(Tensor {
            data: output.into_dyn(),
        })
    }

    /// Execute maxpool
    fn execute_maxpool(
        &self,
        input_id: usize,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(OnnxError::InvalidModel("MaxPool expects 4D input".into()));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = (kernel[0], kernel[1]);
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let mut output = Array::zeros((n, c, h_out, w_out));

        for batch in 0..n {
            for channel in 0..c {
                for y in 0..h_out {
                    for x in 0..w_out {
                        let mut max_val = f32::NEG_INFINITY;

                        for ky in 0..kh {
                            for kx in 0..kw {
                                let in_y = y * sh + ky;
                                let in_x = x * sw + kx;

                                if in_y >= ph && in_y < h + ph && in_x >= pw && in_x < w + pw {
                                    let val = input.data[[batch, channel, in_y - ph, in_x - pw]];
                                    max_val = max_val.max(val);
                                }
                            }
                        }

                        output[[batch, channel, y, x]] = if max_val == f32::NEG_INFINITY {
                            0.0
                        } else {
                            max_val
                        };
                    }
                }
            }
        }

        Ok(Tensor {
            data: output.into_dyn(),
        })
    }

    /// Execute avgpool
    fn execute_avgpool(
        &self,
        input_id: usize,
        kernel: &[usize],
        stride: &[usize],
        padding: &[usize],
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let shape = input.shape();

        if shape.len() != 4 {
            return Err(OnnxError::InvalidModel("AvgPool expects 4D input".into()));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (kh, kw) = (kernel[0], kernel[1]);
        let (sh, sw) = (stride[0], stride[1]);
        let (ph, pw) = (padding[0], padding[1]);

        let h_out = (h + 2 * ph - kh) / sh + 1;
        let w_out = (w + 2 * pw - kw) / sw + 1;

        let mut output = Array::zeros((n, c, h_out, w_out));

        for batch in 0..n {
            for channel in 0..c {
                for y in 0..h_out {
                    for x in 0..w_out {
                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for ky in 0..kh {
                            for kx in 0..kw {
                                let in_y = y * sh + ky;
                                let in_x = x * sw + kx;

                                if in_y >= ph && in_y < h + ph && in_x >= pw && in_x < w + pw {
                                    sum += input.data[[batch, channel, in_y - ph, in_x - pw]];
                                    count += 1;
                                }
                            }
                        }

                        output[[batch, channel, y, x]] =
                            if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Ok(Tensor {
            data: output.into_dyn(),
        })
    }

    /// Execute batchnorm
    fn execute_batchnorm(
        &self,
        input_id: usize,
        scale_id: usize,
        bias_id: usize,
        mean_id: usize,
        var_id: usize,
        epsilon: f32,
    ) -> Result<Tensor> {
        let input = self.get_tensor(input_id)?;
        let scale = self.get_tensor(scale_id)?;
        let bias = self.get_tensor(bias_id)?;
        let mean = self.get_tensor(mean_id)?;
        let var = self.get_tensor(var_id)?;

        let mut result = input.data.clone();
        let shape = result.shape().to_vec();

        if shape.len() >= 2 {
            let c = shape[1];
            for channel in 0..c {
                let s = scale.data[[channel]];
                let b = bias.data[[channel]];
                let m = mean.data[[channel]];
                let v = var.data[[channel]];
                let inv_std = 1.0 / (v + epsilon).sqrt();

                for batch in 0..shape[0] {
                    if shape.len() == 4 {
                        for y in 0..shape[2] {
                            for x in 0..shape[3] {
                                let val = result[[batch, channel, y, x]];
                                result[[batch, channel, y, x]] = (val - m) * inv_std * s + b;
                            }
                        }
                    } else if shape.len() == 2 {
                        let val = result[[batch, channel]];
                        result[[batch, channel]] = (val - m) * inv_std * s + b;
                    }
                }
            }
        }

        Ok(Tensor { data: result })
    }

    /// Execute cast
    fn execute_cast(&self, input_id: usize, _dtype: &str) -> Result<Tensor> {
        // For now, just pass through since we store everything as f32
        let input = self.get_tensor(input_id)?;
        Ok(Tensor {
            data: input.data.clone(),
        })
    }

    /// Execute stack
    fn execute_stack(&self, input_ids: &[usize], axis: i32) -> Result<Tensor> {
        if input_ids.is_empty() {
            return Err(OnnxError::InvalidModel(
                "Stack requires at least one input".into(),
            ));
        }

        let first = self.get_tensor(input_ids[0])?;
        let ndim = first.shape().len();
        let axis = if axis < 0 {
            (ndim as i32 + 1 + axis) as usize
        } else {
            axis as usize
        };

        let views: Vec<_> = input_ids
            .iter()
            .map(|&id| {
                self.get_tensor(id)
                    .map(|t| t.data.clone().insert_axis(Axis(axis)))
            })
            .collect::<Result<Vec<_>>>()?;

        let view_refs: Vec<_> = views.iter().map(|v| v.view()).collect();

        let result = ndarray::concatenate(Axis(axis), &view_refs)
            .map_err(|e| OnnxError::InvalidModel(format!("Stack failed: {}", e)))?;

        Ok(Tensor { data: result })
    }

    /// Execute vstack
    fn execute_vstack(&self, input_ids: &[usize]) -> Result<Tensor> {
        self.execute_concat(input_ids, 0)
    }

    /// Execute hstack
    fn execute_hstack(&self, input_ids: &[usize]) -> Result<Tensor> {
        self.execute_concat(input_ids, 1)
    }

    /// Execute runtime call operation
    fn execute_call(&self, node_id: usize, func: &str, args: &[usize]) -> Result<Tensor> {
        match func {
            "onnx.Shape" => self.execute_shape(args),
            "onnx.ConstantOfShape" => self.execute_constant_of_shape(args),
            "onnx.GroupNormalization" => self.execute_group_normalization(args),
            "onnx.Reshape" => self.execute_dynamic_reshape(node_id, args),
            "onnx.Resize" | "onnx.Upsample" => self.execute_resize(node_id, args),
            _ => Err(OnnxError::unsupported_op(format!("Call::{}", func), 0)),
        }
    }

    fn execute_dynamic_reshape(&self, node_id: usize, args: &[usize]) -> Result<Tensor> {
        if args.len() != 2 {
            return Err(OnnxError::InvalidModel(format!(
                "Reshape expects 2 inputs, got {}",
                args.len()
            )));
        }

        let input = self.get_tensor(args[0])?;
        let shape_tensor = self.get_tensor(args[1])?;
        let shape_values = self.parse_shape_tensor(shape_tensor)?;
        let new_shape = self
            .resolve_reshape_dims(input.shape(), &shape_values)
            .map_err(|e| {
                OnnxError::InvalidModel(format!(
                    "Reshape node {} failed to resolve dims for input {:?} shape {:?}: {}",
                    node_id,
                    input.shape(),
                    shape_values,
                    e
                ))
            })?;
        let input_size: usize = input.shape().iter().product();
        let new_size: usize = new_shape.iter().product();
        if input_size != new_size {
            warn!(
                "Reshape node {} size mismatch: input {:?} ({}) -> target {:?} ({}) (shape={:?})",
                node_id,
                input.shape(),
                input_size,
                new_shape,
                new_size,
                shape_values
            );
            return Err(OnnxError::InvalidModel(format!(
                "Reshape node {} size mismatch: input {:?} ({}) -> target {:?} ({}) (shape={:?})",
                node_id,
                input.shape(),
                input_size,
                new_shape,
                new_size,
                shape_values
            )));
        }

        self.reshape_with_shape(input, &new_shape)
    }

    fn parse_shape_tensor(&self, shape_tensor: &Tensor) -> Result<Vec<i64>> {
        shape_tensor
            .data
            .iter()
            .map(|&val| {
                if (val - val.round()).abs() > 1e-6 {
                    return Err(OnnxError::InvalidModel(format!(
                        "Reshape expects integer dims, got {}",
                        val
                    )));
                }
                Ok(val as i64)
            })
            .collect()
    }

    fn execute_resize(&self, node_id: usize, args: &[usize]) -> Result<Tensor> {
        if args.is_empty() || args.len() > 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Resize expects 1-4 inputs, got {}",
                args.len()
            )));
        }

        let input = self.get_tensor(args[0])?;
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Resize expects 4D input, got shape {:?}",
                input_shape
            )));
        }

        let scales_id = if args.len() == 2 {
            Some(args[1])
        } else if args.len() >= 3 {
            Some(args[2])
        } else {
            None
        };
        let sizes_id = if args.len() == 4 { Some(args[3]) } else { None };

        let sizes = if let Some(sizes_id) = sizes_id {
            let sizes_tensor = self.get_tensor(sizes_id)?;
            Some(self.parse_shape_tensor(sizes_tensor)?)
        } else {
            None
        };

        let scales = if sizes.is_none() {
            if let Some(scales_id) = scales_id {
                let scales_tensor = self.get_tensor(scales_id)?;
                Some(
                    scales_tensor
                        .data
                        .iter()
                        .map(|&v| v as f32)
                        .collect::<Vec<f32>>(),
                )
            } else {
                None
            }
        } else {
            None
        };

        let output_shape: Vec<usize> = if let Some(sizes) = sizes {
            sizes
                .iter()
                .map(|&s| {
                    if s <= 0 {
                        Err(OnnxError::InvalidModel(format!(
                            "Resize node {} invalid size {}",
                            node_id, s
                        )))
                    } else {
                        Ok(s as usize)
                    }
                })
                .collect::<Result<_>>()?
        } else if let Some(scales) = scales {
            if scales.len() != input_shape.len() {
                return Err(OnnxError::InvalidModel(format!(
                    "Resize node {} scales length {} does not match input rank {}",
                    node_id,
                    scales.len(),
                    input_shape.len()
                )));
            }
            input_shape
                .iter()
                .zip(scales.iter())
                .map(|(&dim, &scale)| {
                    let scaled = (dim as f32 * scale).round();
                    if scaled <= 0.0 {
                        1
                    } else {
                        scaled as usize
                    }
                })
                .collect()
        } else {
            return Err(OnnxError::InvalidModel(format!(
                "Resize node {} missing scales or sizes input",
                node_id
            )));
        };

        if output_shape.len() != 4 {
            return Err(OnnxError::InvalidModel(format!(
                "Resize node {} expects 4D output, got {:?}",
                node_id, output_shape
            )));
        }

        let (n, c, h, w) = (input_shape[0], input_shape[1], input_shape[2], input_shape[3]);
        let (out_h, out_w) = (output_shape[2], output_shape[3]);
        let scale_h = if h == 0 { 1.0 } else { out_h as f32 / h as f32 };
        let scale_w = if w == 0 { 1.0 } else { out_w as f32 / w as f32 };

        let mut output = Array::zeros((n, c, out_h, out_w));
        for b in 0..n {
            for ch in 0..c {
                for y in 0..out_h {
                    let in_y = ((y as f32) / scale_h).floor() as usize;
                    let src_y = in_y.min(h.saturating_sub(1));
                    for x in 0..out_w {
                        let in_x = ((x as f32) / scale_w).floor() as usize;
                        let src_x = in_x.min(w.saturating_sub(1));
                        output[[b, ch, y, x]] = input.data[[b, ch, src_y, src_x]];
                    }
                }
            }
        }

        Ok(Tensor {
            data: output.into_dyn(),
        })
    }

    fn resolve_reshape_dims(&self, input_shape: &[usize], dims: &[i64]) -> Result<Vec<usize>> {
        if dims.is_empty() {
            return Err(OnnxError::InvalidModel(
                "Reshape requires non-empty shape".to_string(),
            ));
        }

        let resolved_dims: Vec<i64> = if dims.len() == 1 && dims[0] == 0 {
            input_shape.iter().map(|&d| d as i64).collect()
        } else {
            dims.to_vec()
        };

        let input_size: usize = input_shape.iter().product();
        let mut new_shape = Vec::with_capacity(resolved_dims.len());
        let mut infer_index: Option<usize> = None;
        let mut known_product: usize = 1;

        for (idx, &dim) in resolved_dims.iter().enumerate() {
            match dim {
                0 => {
                    let size = *input_shape.get(idx).ok_or_else(|| {
                        OnnxError::InvalidModel(format!(
                            "Reshape dimension {} is out of range for input shape {:?}",
                            idx, input_shape
                        ))
                    })?;
                    new_shape.push(size);
                    known_product = known_product.saturating_mul(size);
                }
                -1 => {
                    if infer_index.is_some() {
                        return Err(OnnxError::InvalidModel(
                            "Reshape can only infer one dimension".to_string(),
                        ));
                    }
                    infer_index = Some(idx);
                    new_shape.push(1);
                }
                n if n > 0 => {
                    let size = usize::try_from(n).map_err(|_| {
                        OnnxError::InvalidModel(format!("Invalid reshape dim {}", n))
                    })?;
                    new_shape.push(size);
                    known_product = known_product.saturating_mul(size);
                }
                _ => {
                    return Err(OnnxError::InvalidModel(format!(
                        "Reshape expects dims >= -1, got {}",
                        dim
                    )));
                }
            }
        }

        if let Some(idx) = infer_index {
            if known_product == 0 {
                return Err(OnnxError::InvalidModel(
                    "Reshape cannot infer dimension from zero-sized shape".to_string(),
                ));
            }
            if !input_size.is_multiple_of(known_product) {
                return Err(OnnxError::InvalidModel(format!(
                    "Reshape cannot infer dimension: input size {} not divisible by {}",
                    input_size, known_product
                )));
            }
            new_shape[idx] = input_size / known_product;
        } else if input_size != known_product {
            return Err(OnnxError::InvalidModel(format!(
                "Reshape size mismatch: input has {} elements, target has {}",
                input_size, known_product
            )));
        }

        Ok(new_shape)
    }

    fn reshape_with_shape(&self, input: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
        let input_shape = input.shape();
        let input_size: usize = input_shape.iter().product();
        let new_size: usize = new_shape.iter().product();

        if input_size != new_size {
            warn!(
                "Reshape size mismatch: input {:?} ({}) -> target {:?} ({})",
                input_shape, input_size, new_shape, new_size
            );
            return Err(OnnxError::InvalidModel(format!(
                "Reshape size mismatch: input {:?} ({}) -> target {:?} ({})",
                input_shape, input_size, new_shape, new_size
            )));
        }

        let contiguous = if input.data.is_standard_layout() {
            input.data.clone()
        } else {
            ArrayD::from_shape_vec(IxDyn(input_shape), input.data.iter().copied().collect())
                .map_err(|e| OnnxError::InvalidModel(format!("Failed to make contiguous: {}", e)))?
        };

        let reshaped = contiguous
            .into_shape_with_order(IxDyn(new_shape))
            .map_err(|e| OnnxError::InvalidModel(format!("Reshape failed: {}", e)))?;

        Ok(Tensor { data: reshaped })
    }

    fn execute_shape(&self, args: &[usize]) -> Result<Tensor> {
        if args.len() != 1 {
            return Err(OnnxError::InvalidModel(format!(
                "Shape expects 1 input, got {}",
                args.len()
            )));
        }
        let input = self.get_tensor(args[0])?;
        let shape: Vec<f32> = input.shape().iter().map(|&d| d as f32).collect();
        Tensor::from_vec(shape.clone(), &[shape.len()])
    }

    fn execute_constant_of_shape(&self, args: &[usize]) -> Result<Tensor> {
        if args.len() != 1 {
            return Err(OnnxError::InvalidModel(format!(
                "ConstantOfShape expects 1 input, got {}",
                args.len()
            )));
        }
        let shape_tensor = self.get_tensor(args[0])?;
        let mut shape = Vec::new();
        for &val in shape_tensor.data.iter() {
            if (val - val.round()).abs() > 1e-6 {
                return Err(OnnxError::InvalidModel(format!(
                    "ConstantOfShape expects integer dims, got {}",
                    val
                )));
            }
            let dim = val as i64;
            if dim < 0 {
                return Err(OnnxError::InvalidModel(format!(
                    "ConstantOfShape expects non-negative dims, got {}",
                    dim
                )));
            }
            shape.push(dim as usize);
        }
        Ok(Tensor::zeros(&shape))
    }

    fn execute_group_normalization(&self, args: &[usize]) -> Result<Tensor> {
        if args.len() != 5 {
            return Err(OnnxError::InvalidModel(format!(
                "GroupNormalization expects 5 inputs, got {}",
                args.len()
            )));
        }

        let input = self.get_tensor(args[0])?;
        let scale = self.get_tensor(args[1])?;
        let bias = self.get_tensor(args[2])?;
        let num_groups = self.get_scalar(args[3])? as usize;
        let epsilon = self.get_scalar(args[4])?;

        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(OnnxError::InvalidModel(format!(
                "GroupNormalization expects rank >= 2, got {:?}",
                input_shape
            )));
        }

        let n = input_shape[0];
        let c = input_shape[1];
        if num_groups == 0 || !c.is_multiple_of(num_groups) {
            return Err(OnnxError::InvalidModel(format!(
                "GroupNormalization expects C divisible by num_groups, got C={}, groups={}",
                c, num_groups
            )));
        }

        let group_size = c / num_groups;
        let spatial = if input_shape.len() > 2 {
            input_shape[2..].iter().product()
        } else {
            1
        };

        let reshaped = input
            .data
            .clone()
            .into_shape_with_order(IxDyn(&[n, num_groups, group_size, spatial]))
            .map_err(|e| OnnxError::InvalidModel(format!("GroupNorm reshape failed: {}", e)))?;

        let mean = reshaped
            .mean_axis(Axis(3))
            .ok_or_else(|| OnnxError::InvalidModel("GroupNorm mean axis 3 failed".into()))?;
        let mean = mean
            .mean_axis(Axis(2))
            .ok_or_else(|| OnnxError::InvalidModel("GroupNorm mean axis 2 failed".into()))?;
        let mean = mean.insert_axis(Axis(2)).insert_axis(Axis(3));

        let centered = &reshaped - &mean;
        let centered_sq = centered.mapv(|x| x * x);

        let var = centered_sq
            .mean_axis(Axis(3))
            .ok_or_else(|| OnnxError::InvalidModel("GroupNorm var axis 3 failed".into()))?;
        let var = var
            .mean_axis(Axis(2))
            .ok_or_else(|| OnnxError::InvalidModel("GroupNorm var axis 2 failed".into()))?;
        let var = var.insert_axis(Axis(2)).insert_axis(Axis(3));

        let std = var.mapv(|x| (x + epsilon).sqrt());
        let normalized = &centered / &std;
        let normalized = normalized
            .into_shape_with_order(IxDyn(input_shape))
            .map_err(|e| {
                OnnxError::InvalidModel(format!("GroupNorm reshape back failed: {}", e))
            })?;

        let scale_b = self.broadcast_param(scale, input_shape, c)?;
        let bias_b = self.broadcast_param(bias, input_shape, c)?;

        let scaled = &normalized * &scale_b;
        let result = &scaled + &bias_b;

        Ok(Tensor { data: result })
    }

    fn broadcast_param(
        &self,
        param: &Tensor,
        input_shape: &[usize],
        channels: usize,
    ) -> Result<ArrayD<f32>> {
        let input_rank = input_shape.len();
        let mut target = vec![1; input_rank];
        target[1] = channels;

        let mut param_data = param.data.clone();
        if param_data.ndim() == 1 {
            if param_data.len() != channels {
                return Err(OnnxError::InvalidModel(format!(
                    "Param length {} does not match channels {}",
                    param_data.len(),
                    channels
                )));
            }
            param_data = param_data
                .into_shape_with_order(IxDyn(&target))
                .map_err(|e| OnnxError::InvalidModel(format!("Param reshape failed: {}", e)))?;
        } else if param_data.ndim() == 2 && input_rank > 2 {
            let shape = param_data.shape();
            if shape != [1, channels] {
                return Err(OnnxError::InvalidModel(format!(
                    "Param shape {:?} not broadcastable to channels {}",
                    shape, channels
                )));
            }
            param_data = param_data
                .into_shape_with_order(IxDyn(&target))
                .map_err(|e| OnnxError::InvalidModel(format!("Param reshape failed: {}", e)))?;
        }

        let view = param_data.broadcast(IxDyn(input_shape)).ok_or_else(|| {
            OnnxError::InvalidModel(format!(
                "Param shape {:?} not broadcastable to input {:?}",
                param_data.shape(),
                input_shape
            ))
        })?;
        Ok(view.to_owned())
    }

    fn get_scalar(&self, id: usize) -> Result<f32> {
        let tensor = self.get_tensor(id)?;
        if tensor.data.len() != 1 {
            return Err(OnnxError::InvalidModel(format!(
                "Expected scalar tensor for node {}, got shape {:?}",
                id,
                tensor.shape()
            )));
        }
        Ok(*tensor.data.iter().next().unwrap_or(&0.0))
    }

    /// Get a tensor by node ID
    fn get_tensor(&self, id: usize) -> Result<&Tensor> {
        self.tensors
            .get(&id)
            .ok_or_else(|| OnnxError::InvalidModel(format!("Tensor for node {} not found", id)))
    }
}

fn build_constant_weight_map(model: &HoloModel) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for node in &model.graph.nodes {
        if let SerNodeKind::Constant { weight_id } = node.node {
            map.insert(node.id, weight_id);
        }
    }
    map
}

// Helper functions for byte conversion

fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn bytes_to_f64_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(8)
        .map(|chunk| {
            let val = f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            val as f32
        })
        .collect()
}

fn bytes_to_i64_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(8)
        .map(|chunk| {
            let val = i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            val as f32
        })
        .collect()
}

fn bytes_to_i32_as_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| {
            let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            val as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data.clone(), &[2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.to_vec(), data);
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(tensor.shape(), &[2, 3, 4]);
        assert!(tensor.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_bytes_to_f32() {
        let val = 2.5f32;
        let bytes = val.to_le_bytes();
        let result = bytes_to_f32(&bytes);
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5).abs() < 1e-6);
    }
}

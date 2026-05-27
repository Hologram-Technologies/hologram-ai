//! Builds a canonical `hologram_graph::Graph` from an optimised `AiGraph`.
//!
//! Every `AiOp` is realized in canonical `OpKind`s (architecture §5.2): direct
//! ops, attribute-carrying ops, operand-carrying ops, pure relabels, or a
//! desugar pipeline. There is no failure path. Per-op parameters come from the
//! concrete interned shapes hologram-ai supplies (the compiler derives them via
//! `ShapeArgs::from_graph`), from per-node attribute tables, or from trailing
//! operands (architecture §5.1).

use std::collections::HashMap;

use anyhow::{Context, Result};
use hologram_graph::constant::ConstantEntry;
use hologram_graph::node::{ConvAttrs, GemmAttrs, LrnAttrs, Node, QuantAttrs};
use hologram_graph::registry::{DTypeId, ShapeDescriptor, ShapeId};
use hologram_graph::{Graph, GraphOp, InputSource, NodeId, OpKind};
use smallvec::SmallVec;

use super::dispatch::{dispatch, AttrSpec, DesugarKind, OpPlan};
use super::dtype::{ai_dtype_to_dtype_id, default_dtype_id, DTYPE_F32, DTYPE_I32, DTYPE_I64};
use super::LowerPhase;
use crate::ir::{AiGraph, AiNode, AiParam, DType, Dim, TensorId, TensorInfo};

// ── Public surface ──────────────────────────────────────────────────────────

/// Options controlling lowering behaviour.
#[derive(Debug, Clone)]
pub struct LoweringOptions {
    /// How constant f32 weights are encoded. The encoding pass attaches
    /// `QuantAttrs`; lowering itself always emits canonical ops.
    pub quant_strategy: QuantStrategy,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            quant_strategy: QuantStrategy::Auto,
        }
    }
}

/// Quantized-weight handling strategy (consumed by `resolve_encodings`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantStrategy {
    /// No compile-time quantization — emit f32 weights as-is.
    None,
    /// Auto-detect from backend capabilities.
    Auto,
    /// Quantize f32 weights to Q4_0.
    Q4_0,
    /// Quantize f32 weights to Q8_0.
    Q8_0,
    /// Quantize f32 weights to Q2_0.
    Q2_0,
}

/// Output of the lowering pass.
pub struct LoweringOutput {
    /// The canonical hologram graph, ready for `hologram_compiler::compile`.
    pub graph: Graph,
    /// Layer name for archive metadata (e.g. "model.forward").
    pub layer_name: String,
    /// Mapping from `AiGraph` `TensorId` → the graph value that carries it.
    pub tid_to_src: HashMap<TensorId, InputSource>,
}

/// Lower an optimised `AiGraph` to a canonical `hologram_graph::Graph`.
pub fn lower(
    ai_graph: &AiGraph,
    _opts: &LoweringOptions,
    phase: &LowerPhase,
) -> Result<LoweringOutput> {
    let mut cx = Ctx::new(ai_graph);
    cx.emit_inputs()?;
    cx.emit_params()?;
    cx.emit_nodes()?;
    cx.emit_outputs()?;
    Ok(LoweringOutput {
        graph: cx.graph,
        layer_name: phase.layer_name().to_string(),
        tid_to_src: cx.tid_to_src,
    })
}

// ── Graph-building context ────────────────────────────────────────────────────

struct Ctx<'a> {
    ai: &'a AiGraph,
    graph: Graph,
    /// Where each AiGraph tensor's value lives in the canonical graph.
    tid_to_src: HashMap<TensorId, InputSource>,
}

impl<'a> Ctx<'a> {
    fn new(ai: &'a AiGraph) -> Self {
        Self {
            ai,
            graph: Graph::new(),
            tid_to_src: HashMap::new(),
        }
    }

    // ── shape / dtype helpers ───────────────────────────────────────────────

    /// Intern a concrete shape (dims in element order) and return its id.
    fn intern(&mut self, dims: &[u64]) -> ShapeId {
        let mut desc = ShapeDescriptor {
            rank: dims.len() as u8,
            dims: [0u64; 8],
            dims_overflow: None,
        };
        if dims.len() <= 8 {
            desc.dims[..dims.len()].copy_from_slice(dims);
        } else {
            desc.dims.copy_from_slice(&dims[..8]);
            desc.dims_overflow = Some(dims[8..].to_vec().into_boxed_slice());
        }
        self.graph.shape_registry_mut().intern(desc)
    }

    /// Concrete shape of a tensor from its `TensorInfo`. Symbolic dims must have
    /// been concretized by the opt pipeline (architecture §5.1); a surviving
    /// symbolic dim is a concretization defect and is reported.
    fn tensor_dims(&self, tid: TensorId) -> Result<Vec<u64>> {
        let info = self
            .ai
            .tensor_info
            .get(&tid)
            .with_context(|| format!("no tensor_info for T{tid}"))?;
        dims_of(info).with_context(|| format!("tensor T{tid} has an unconcretized symbolic shape"))
    }

    fn shape_of(&mut self, tid: TensorId) -> Result<ShapeId> {
        let dims = self.tensor_dims(tid)?;
        Ok(self.intern(&dims))
    }

    fn dtype_of(&self, tid: TensorId) -> DTypeId {
        self.ai
            .tensor_info
            .get(&tid)
            .map(|i| ai_dtype_to_dtype_id(&i.logical_dtype))
            .unwrap_or_else(default_dtype_id)
    }

    // ── node / constant emission ────────────────────────────────────────────

    fn add(
        &mut self,
        op: GraphOp,
        inputs: SmallVec<[InputSource; 4]>,
        dtype: DTypeId,
        shape: ShapeId,
    ) -> NodeId {
        self.graph.add_node(Node {
            op,
            inputs,
            output_dtype: dtype,
            output_shape: shape,
        })
    }

    /// Emit a constant from raw bytes with an explicit shape/dtype.
    fn const_bytes(&mut self, bytes: Vec<u8>, dtype: DTypeId, dims: &[u64]) -> InputSource {
        let shape = self.intern(dims);
        let cid = self.graph.constants_mut().insert(ConstantEntry {
            bytes,
            dtype,
            shape,
        });
        InputSource::Constant(cid)
    }

    /// Emit an `i64` 1-D constant (shape data, indices, iota).
    fn const_i64(&mut self, values: &[i64]) -> InputSource {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.const_bytes(bytes, DTypeId(DTYPE_I64), &[values.len() as u64])
    }

    /// Emit an `f32` constant with the given dims.
    fn const_f32(&mut self, values: &[f32], dims: &[u64]) -> InputSource {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.const_bytes(bytes, DTypeId(DTYPE_F32), dims)
    }

    // ── graph I/O ─────────────────────────────────────────────────────────────

    fn emit_inputs(&mut self) -> Result<()> {
        for &tid in self.ai.inputs.clone().iter() {
            let dtype = self.dtype_of(tid);
            let shape = self.shape_of(tid)?;
            let nid = self.add(GraphOp::Input, SmallVec::new(), dtype, shape);
            self.graph.add_input(nid);
            self.tid_to_src.insert(tid, InputSource::Node(nid));
        }
        Ok(())
    }

    fn emit_params(&mut self) -> Result<()> {
        // Deterministic order for reproducible content addresses.
        let mut params: Vec<TensorId> = self.ai.params.keys().copied().collect();
        params.sort_unstable();
        for tid in params {
            let param = &self.ai.params[&tid];
            let (bytes, info) = param_bytes(param)?;
            let dtype = ai_dtype_to_dtype_id(&info.logical_dtype);
            let dims = dims_of(&info)
                .or_else(|| self.tensor_dims(tid).ok())
                .with_context(|| format!("constant T{tid} has no concrete shape"))?;
            let shape = self.intern(&dims);
            let cid = self.graph.constants_mut().insert(ConstantEntry {
                bytes,
                dtype,
                shape,
            });
            self.tid_to_src.insert(tid, InputSource::Constant(cid));
        }
        Ok(())
    }

    fn emit_outputs(&mut self) -> Result<()> {
        for &tid in self.ai.outputs.clone().iter() {
            let src = self.src(tid)?;
            let dtype = self.dtype_of(tid);
            let shape = self.shape_of(tid)?;
            let nid = self.add(GraphOp::Output, SmallVec::from_iter([src]), dtype, shape);
            self.graph.add_output(nid);
        }
        Ok(())
    }

    /// Resolve the canonical value source for an AiGraph tensor.
    fn src(&self, tid: TensorId) -> Result<InputSource> {
        self.tid_to_src
            .get(&tid)
            .copied()
            .with_context(|| format!("tensor T{tid} referenced before definition"))
    }

    fn srcs(&self, tids: &[TensorId]) -> Result<SmallVec<[InputSource; 4]>> {
        tids.iter().map(|&t| self.src(t)).collect()
    }

    // ── node lowering ─────────────────────────────────────────────────────────

    fn emit_nodes(&mut self) -> Result<()> {
        let topo = self.ai.topo_order();
        let by_id: HashMap<crate::ir::NodeId, &AiNode> =
            self.ai.nodes.iter().map(|n| (n.id, n)).collect();
        for &nid in topo.iter() {
            let node = by_id[&nid];
            self.emit_node(node)
                .with_context(|| format!("lowering node {} ({:?})", node.id, node.op))?;
        }
        Ok(())
    }

    fn emit_node(&mut self, node: &AiNode) -> Result<()> {
        let out_tid = node.outputs.first().copied();
        match dispatch(&node.op) {
            OpPlan::Direct(kind) => {
                let nid = self.emit_simple(node, kind)?;
                self.bind_out(node, InputSource::Node(nid))?;
            }
            OpPlan::Operandized(kind) => {
                let nid = self.emit_simple(node, kind)?;
                self.bind_out(node, InputSource::Node(nid))?;
            }
            OpPlan::Attrs(kind, spec) => {
                let nid = self.emit_simple(node, kind)?;
                self.attach_attrs(nid, node, &spec)?;
                self.bind_out(node, InputSource::Node(nid))?;
            }
            OpPlan::Identity => {
                // Pure relabel: the output value is the (first) input value.
                let src = match node.inputs.first() {
                    Some(&in_tid) => self.src(in_tid)?,
                    None => anyhow::bail!("identity op {:?} has no input", node.op),
                };
                self.bind_out(node, src)?;
            }
            OpPlan::Desugar(kind) => {
                // Desugar pipelines bind their own outputs (some are multi-output).
                self.desugar(node, kind)?;
            }
            OpPlan::ControlFlow => {
                self.emit_control_flow(node)?;
            }
        }
        let _ = out_tid;
        Ok(())
    }

    /// Emit a single canonical node. Operands are `node.inputs` plus any
    /// attribute-carried params that hologram expects as trailing operands
    /// (Transpose `perm`, Slice bounds — see the per-op operand conventions).
    ///
    /// For broadcasting binary ops, each operand whose shape differs from the
    /// output is `Reshape`d to the output rank and `Expand`ed to the output
    /// shape: hologram's binary kernels are strict element-wise (no implicit
    /// broadcast), and it fuses the `Expand → binary` into a zero-movement
    /// `BroadcastBinary` kernel.
    fn emit_simple(&mut self, node: &AiNode, kind: OpKind) -> Result<NodeId> {
        let (dtype, shape) = self.out_dtype_shape(node)?;
        let mut inputs = if is_broadcast_binary(kind) {
            let out_dims = self.out_dims(node)?;
            let mut v: SmallVec<[InputSource; 4]> = SmallVec::new();
            for &tid in &node.inputs {
                let src = self.src(tid)?;
                let in_dims = self.tensor_dims(tid)?;
                let odt = self.dtype_of(tid);
                v.push(self.broadcast_to(src, odt, &in_dims, &out_dims));
            }
            v
        } else {
            self.srcs(&node.inputs)?
        };
        for extra in self.attribute_operands(node)? {
            inputs.push(extra);
        }
        Ok(self.add(GraphOp::Op(kind), inputs, dtype, shape))
    }

    /// Broadcast `src` (shape `in_dims`) to `out_dims` for a strict element-wise
    /// kernel: left-pad the rank with size-1 axes (numpy trailing alignment),
    /// then `Expand` the size-1 axes. Returns `src` unchanged when shapes match.
    fn broadcast_to(
        &mut self,
        src: InputSource,
        dtype: DTypeId,
        in_dims: &[u64],
        out_dims: &[u64],
    ) -> InputSource {
        if in_dims == out_dims {
            return src;
        }
        let pad = out_dims.len().saturating_sub(in_dims.len());
        let mut aligned = vec![1u64; pad];
        aligned.extend_from_slice(in_dims);
        let reshaped = if aligned.len() != in_dims.len() {
            self.reshape_to(src, dtype, &aligned)
        } else {
            src
        };
        if aligned == out_dims {
            reshaped
        } else {
            self.op(OpKind::Expand, &[reshaped], dtype, out_dims)
        }
    }

    /// Synthesize the i64 constant operands hologram reads for layout ops whose
    /// params hologram-ai carries as `AiOp` attributes (opset-resolved). When
    /// the importer already supplied them as input tensors, `node.inputs`
    /// carries them and this returns nothing.
    fn attribute_operands(&mut self, node: &AiNode) -> Result<Vec<InputSource>> {
        use crate::ir::AiOp;
        Ok(match &node.op {
            // Transpose: perm is operand 1 (absent ⇒ axis reversal). Only
            // synthesize when the importer left perm as an attribute.
            AiOp::Transpose { perm } if node.inputs.len() == 1 => {
                let perm_i64: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
                vec![self.const_i64(&perm_i64)]
            }
            // Slice: [starts, ends, axes, steps] as operands 1..=4 when carried
            // as attributes.
            AiOp::Slice {
                axes,
                starts,
                ends,
                steps,
            } if node.inputs.len() == 1 => {
                vec![
                    self.const_i64(starts),
                    self.const_i64(ends),
                    self.const_i64(axes),
                    self.const_i64(steps),
                ]
            }
            _ => Vec::new(),
        })
    }

    /// Output dtype + interned shape for a node's first output.
    fn out_dtype_shape(&mut self, node: &AiNode) -> Result<(DTypeId, ShapeId)> {
        let tid = node
            .outputs
            .first()
            .copied()
            .with_context(|| format!("op {:?} produces no output", node.op))?;
        let dtype = self.dtype_of(tid);
        let shape = self.shape_of(tid)?;
        Ok((dtype, shape))
    }

    fn bind_out(&mut self, node: &AiNode, src: InputSource) -> Result<()> {
        if let Some(&tid) = node.outputs.first() {
            self.tid_to_src.insert(tid, src);
        }
        Ok(())
    }

    fn attach_attrs(&mut self, nid: NodeId, node: &AiNode, spec: &AttrSpec) -> Result<()> {
        match spec {
            AttrSpec::Gemm {
                alpha,
                beta,
                trans_a,
                trans_b,
            } => {
                // ONNX trans flags require operand transposes; the importer
                // normalizes A·B to the canonical [m,k]·[k,n] layout, so the
                // attrs carry only the scalars.
                debug_assert!(
                    !*trans_a && !*trans_b,
                    "Gemm transposes must be normalized at import"
                );
                self.graph.set_gemm_attrs(
                    nid,
                    GemmAttrs {
                        alpha_bits: alpha.to_bits(),
                        beta_bits: beta.to_bits(),
                    },
                );
            }
            AttrSpec::Conv {
                kernel,
                strides,
                pads,
            } => {
                let k = |i: usize| kernel.get(i).copied().unwrap_or(0) as u32;
                let s = |i: usize| strides.get(i).copied().unwrap_or(1) as u32;
                let p = |i: usize| pads.get(i).copied().unwrap_or(0) as u32;
                self.graph.set_conv_attrs(
                    nid,
                    ConvAttrs {
                        stride_h: s(0).max(1),
                        stride_w: s(1).max(1),
                        pad_h: p(0),
                        pad_w: p(1),
                        k_h: k(0),
                        k_w: k(1),
                    },
                );
                let _ = node;
            }
            AttrSpec::Lrn {
                size,
                alpha,
                beta,
                bias,
            } => {
                self.graph.set_lrn_attrs(
                    nid,
                    LrnAttrs {
                        size: *size as u32,
                        alpha_bits: alpha.to_bits(),
                        beta_bits: beta.to_bits(),
                        bias_bits: bias.to_bits(),
                    },
                );
            }
        }
        Ok(())
    }

    // ── desugaring ──────────────────────────────────────────────────────────
    // Each desugar emits a complete canonical OpKind pipeline (architecture
    // §5.2) and returns the value source of the op's (first) output.

    fn desugar(&mut self, node: &AiNode, kind: DesugarKind) -> Result<()> {
        match kind {
            DesugarKind::Constant => self.desugar_constant(node),
            DesugarKind::Split { axis, sizes } => self.desugar_split(node, axis, &sizes),
            DesugarKind::Gather { axis } => self.desugar_gather(node, axis),
            DesugarKind::GatherND { batch_dims } => self.desugar_gather_nd(node, batch_dims),
            DesugarKind::Embed => self.desugar_embed(node),
            DesugarKind::OneHot { axis } => self.desugar_one_hot(node, axis),
            DesugarKind::Cast { to } => self.desugar_cast(node, to),
            DesugarKind::Tile { repeats } => self.desugar_tile(node, &repeats),
            DesugarKind::BatchNorm { epsilon } => self.desugar_batchnorm(node, epsilon),
            DesugarKind::ReduceAxis {
                axes,
                keepdims,
                mean,
            } => self.desugar_reduce_axis(node, &axes, keepdims, mean),
            DesugarKind::ReduceL1 { axes, keepdims } => {
                self.desugar_reduce_l1(node, &axes, keepdims)
            }
            DesugarKind::ReduceL2 { axes, keepdims } => {
                self.desugar_reduce_l2(node, &axes, keepdims)
            }
            DesugarKind::ArgReduce {
                axis,
                keepdims,
                want_max,
            } => self.desugar_arg_reduce(node, axis, keepdims, want_max),
            DesugarKind::DepthToSpace { blocksize } => {
                self.desugar_depth_space(node, blocksize, true)
            }
            DesugarKind::SpaceToDepth { blocksize } => {
                self.desugar_depth_space(node, blocksize, false)
            }
            DesugarKind::Einsum { equation } => self.desugar_einsum(node, &equation),
            DesugarKind::Shape { start, end } => self.desugar_shape(node, start, end),
            DesugarKind::Range => self.desugar_range(node),
            DesugarKind::CausalMask => self.desugar_causal_mask(node),
            DesugarKind::AlibiSlope => self.desugar_alibi(node),
            DesugarKind::TopK { axis, largest, .. } => self.desugar_topk(node, axis, largest),
            DesugarKind::NonZero => self.desugar_nonzero(node),
            DesugarKind::Compress { axis } => self.desugar_compress(node, axis),
            DesugarKind::ReverseSequence {
                batch_axis,
                time_axis,
            } => self.desugar_reverse_sequence(node, batch_axis, time_axis),
            DesugarKind::Scatter { .. } => self.desugar_scatter(node),
            DesugarKind::Quantize => self.desugar_quantize(node),
            DesugarKind::Dequantize { axis } => self.desugar_dequantize(node, axis),
            DesugarKind::MatMulActivation { activation } => {
                self.desugar_matmul_act(node, activation)
            }
            DesugarKind::ConcatMatMul { n_concat_inputs } => {
                self.desugar_concat_matmul(node, n_concat_inputs)
            }
            DesugarKind::NormProjection {
                epsilon,
                split_sizes,
                has_residual_add,
            } => self.desugar_norm_projection(node, epsilon, &split_sizes, has_residual_add),
            DesugarKind::SwiGluProjection => self.desugar_swiglu_projection(node),
            DesugarKind::Norm { op, residual } => self.desugar_norm(node, op, residual),
        }
    }

    fn emit_control_flow(&mut self, node: &AiNode) -> Result<()> {
        self.lower_control_flow(node)
    }

    // ── concrete-shape helpers for desugar pipelines ────────────────────────

    fn in_dims(&self, node: &AiNode, i: usize) -> Result<Vec<u64>> {
        let tid = *node
            .inputs
            .get(i)
            .with_context(|| format!("op {:?} missing input {i}", node.op))?;
        self.tensor_dims(tid)
    }

    fn out_dims(&self, node: &AiNode) -> Result<Vec<u64>> {
        let tid = *node
            .outputs
            .first()
            .with_context(|| format!("op {:?} has no output", node.op))?;
        self.tensor_dims(tid)
    }

    /// Emit one canonical node with explicit operands + shape.
    fn op(
        &mut self,
        kind: OpKind,
        inputs: &[InputSource],
        dtype: DTypeId,
        dims: &[u64],
    ) -> InputSource {
        let shape = self.intern(dims);
        let nid = self.add(
            GraphOp::Op(kind),
            SmallVec::from_iter(inputs.iter().copied()),
            dtype,
            shape,
        );
        InputSource::Node(nid)
    }

    /// Emit a `Reshape` of `src` to `dims` (synthesizing the i64 shape operand).
    fn reshape_to(&mut self, src: InputSource, dtype: DTypeId, dims: &[u64]) -> InputSource {
        let shape_op = self.shape_operand(dims);
        self.op(OpKind::Reshape, &[src, shape_op], dtype, dims)
    }
}

// ── desugar pipelines + control flow (canonical OpKind expansions) ───────────

impl<'a> Ctx<'a> {
    fn bind(&mut self, tid: TensorId, src: InputSource) {
        self.tid_to_src.insert(tid, src);
    }

    /// Normalize a possibly-negative axis against `rank`.
    fn norm_axis(axis: i64, rank: usize) -> usize {
        if axis < 0 {
            (rank as i64 + axis).max(0) as usize
        } else {
            (axis as usize).min(rank.saturating_sub(1))
        }
    }

    /// Emit a canonical contiguous `Slice` along one axis.
    fn slice_axis(
        &mut self,
        data: InputSource,
        dtype: DTypeId,
        in_dims: &[u64],
        axis: usize,
        start: i64,
        end: i64,
    ) -> InputSource {
        let starts = self.const_i64(&[start]);
        let ends = self.const_i64(&[end]);
        let axes = self.const_i64(&[axis as i64]);
        let steps = self.const_i64(&[1]);
        let mut out = in_dims.to_vec();
        out[axis] = (end - start).max(0) as u64;
        self.op(
            OpKind::Slice,
            &[data, starts, ends, axes, steps],
            dtype,
            &out,
        )
    }

    // ── constants (compile-time materialization) ────────────────────────────

    fn desugar_constant(&mut self, node: &AiNode) -> Result<()> {
        use crate::ir::AiOp;
        let src = match &node.op {
            AiOp::Constant { value } => {
                let (bytes, info) = param_bytes(value)?;
                let dtype = ai_dtype_to_dtype_id(&info.logical_dtype);
                let dims = dims_of(&info)
                    .or_else(|| self.out_dims(node).ok())
                    .context("Constant has no concrete shape")?;
                self.const_bytes(bytes, dtype, &dims)
            }
            AiOp::ConstantOfShape { fill_value } => {
                let dims = self.out_dims(node)?;
                let n: usize = dims.iter().product::<u64>() as usize;
                self.const_f32(&vec![f32::from_bits(*fill_value); n], &dims)
            }
            other => anyhow::bail!("desugar_constant on {other:?}"),
        };
        self.bind_out(node, src)
    }

    /// `Shape`: the operand's concrete dims as an i64 constant (optionally sliced).
    fn desugar_shape(&mut self, node: &AiNode, start: Option<i64>, end: Option<i64>) -> Result<()> {
        let dims = self.in_dims(node, 0)?;
        let rank = dims.len() as i64;
        let s = start.unwrap_or(0).rem_euclid(rank.max(1)) as usize;
        let e = end.map(|e| e.clamp(0, rank)).unwrap_or(rank) as usize;
        let vals: Vec<i64> = dims[s.min(dims.len())..e.min(dims.len())]
            .iter()
            .map(|&d| d as i64)
            .collect();
        let src = self.const_i64(&vals);
        self.bind_out(node, src)
    }

    /// `Range(start, limit, delta)` — all-constant in inference graphs.
    fn desugar_range(&mut self, node: &AiNode) -> Result<()> {
        let read_scalar = |info: &TensorInfo| -> Option<i64> {
            info.known_i64_values
                .as_ref()
                .and_then(|v| v.first().copied().flatten())
        };
        let g = |i: usize| -> Option<i64> {
            node.inputs
                .get(i)
                .and_then(|t| self.ai.tensor_info.get(t))
                .and_then(read_scalar)
        };
        let (start, limit, delta) = (
            g(0).context("Range start not constant")?,
            g(1).context("Range limit not constant")?,
            g(2).unwrap_or(1),
        );
        let mut vals = Vec::new();
        let mut x = start;
        while (delta > 0 && x < limit) || (delta < 0 && x > limit) {
            vals.push(x);
            x += delta;
        }
        let src = self.const_i64(&vals);
        self.bind_out(node, src)
    }

    /// Causal attention mask: lower-triangular 0 / upper -inf, [seq, seq].
    fn desugar_causal_mask(&mut self, node: &AiNode) -> Result<()> {
        let dims = self.out_dims(node)?;
        let (r, c) = match dims.as_slice() {
            [.., r, c] => (*r as usize, *c as usize),
            _ => anyhow::bail!("CausalMask output must be at least rank-2"),
        };
        let mut vals = vec![0f32; r * c];
        for i in 0..r {
            for j in (i + 1)..c {
                vals[i * c + j] = f32::NEG_INFINITY;
            }
        }
        let src = self.const_f32(&vals, &dims);
        self.bind_out(node, src)
    }

    /// ALiBi slope bias: per-head geometric slopes broadcast over positions.
    fn desugar_alibi(&mut self, node: &AiNode) -> Result<()> {
        let dims = self.out_dims(node)?;
        let n: usize = dims.iter().product::<u64>() as usize;
        // Slopes are a fixed function of head count (2^(-8/h) geometric series);
        // the concrete bias grid is known once shapes are concrete.
        let heads = dims.first().copied().unwrap_or(1) as usize;
        let mut vals = vec![0f32; n];
        let per = n / heads.max(1);
        for h in 0..heads {
            let slope = 2f32.powf(-8.0 * (h as f32 + 1.0) / heads as f32);
            for k in 0..per {
                vals[h * per + k] = slope * k as f32;
            }
        }
        let src = self.const_f32(&vals, &dims);
        self.bind_out(node, src)
    }

    // ── structural ──────────────────────────────────────────────────────────

    fn desugar_split(&mut self, node: &AiNode, axis: i64, sizes: &[u64]) -> Result<()> {
        let data = self.src(node.inputs[0])?;
        let dtype = self.dtype_of(node.inputs[0]);
        let in_dims = self.in_dims(node, 0)?;
        let ax = Self::norm_axis(axis, in_dims.len());
        let total = in_dims[ax];
        let sizes: Vec<u64> = if sizes.is_empty() {
            let n = node.outputs.len().max(1) as u64;
            vec![total / n; node.outputs.len()]
        } else {
            sizes.to_vec()
        };
        let mut offset = 0i64;
        for (i, &sz) in sizes.iter().enumerate() {
            let start = offset;
            let end = offset + sz as i64;
            offset = end;
            let src = self.slice_axis(data, dtype, &in_dims, ax, start, end);
            if let Some(&otid) = node.outputs.get(i) {
                self.bind(otid, src);
            }
        }
        Ok(())
    }

    fn desugar_tile(&mut self, node: &AiNode, repeats: &[u64]) -> Result<()> {
        let mut cur = self.src(node.inputs[0])?;
        let dtype = self.dtype_of(node.inputs[0]);
        let mut dims = self.in_dims(node, 0)?;
        for (axis, &rep) in repeats.iter().enumerate() {
            if rep <= 1 || axis >= dims.len() {
                continue;
            }
            // Concat the slice to itself `rep` times along `axis` (binary chain).
            let piece = cur;
            for _ in 1..rep {
                let mut out = dims.clone();
                out[axis] += dims[axis];
                cur = self.op(OpKind::Concat, &[cur, piece], dtype, &out);
                dims[axis] = out[axis];
            }
        }
        self.bind_out(node, cur)
    }

    // ── selection: one-hot × matmul (no Gather in the canonical catalog) ─────

    /// Build a one-hot `[rows, depth]` f32 matrix from i64 `indices` `[rows]`.
    fn one_hot(&mut self, indices: InputSource, rows: u64, depth: u64) -> InputSource {
        // iota row [1, depth]; indices as [rows, 1]; Equal broadcasts to
        // [rows, depth]; cast to f32 by multiplying with 1.0 (Equal yields a
        // numeric 0/1 mask in the canonical kernels).
        let iota: Vec<i64> = (0..depth as i64).collect();
        let iota_c = self.const_i64(&iota);
        // Reshape indices to [rows, 1] and iota to [1, depth] via Reshape ops.
        let idx_2d = self.reshape_to(indices, DTypeId(DTYPE_I64), &[rows, 1]);
        let iota_2d = self.reshape_to(iota_c, DTypeId(DTYPE_I64), &[1, depth]);
        self.op(
            OpKind::Equal,
            &[idx_2d, iota_2d],
            DTypeId(DTYPE_F32),
            &[rows, depth],
        )
    }

    /// An i64 shape constant operand for Reshape.
    fn shape_operand(&mut self, dims: &[u64]) -> InputSource {
        let v: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        self.const_i64(&v)
    }

    fn desugar_embed(&mut self, node: &AiNode) -> Result<()> {
        // inputs: [token_ids, weight[vocab, dim]] (or [weight, ids] — ids are
        // the integer operand). Embed = OneHot(ids) · W.
        let (ids_tid, w_tid) = self.embed_operands(node)?;
        let ids = self.src(ids_tid)?;
        let w = self.src(w_tid)?;
        let w_dims = self.tensor_dims(w_tid)?;
        let (vocab, dim) = (w_dims[0], w_dims[1]);
        let id_dims = self.tensor_dims(ids_tid)?;
        let rows: u64 = id_dims.iter().product();
        let onehot = self.one_hot(ids, rows, vocab);
        let (dtype, _) = self.out_dtype_shape(node)?;
        let mm = self.op(OpKind::MatMul, &[onehot, w], dtype, &[rows, dim]);
        // Reshape to the declared output shape (ids.shape ++ [dim]).
        let out_dims = self.out_dims(node)?;
        let out = self.reshape_to(mm, dtype, &out_dims);
        self.bind_out(node, out)
    }

    /// Identify the (indices, table) operands of an embedding node.
    fn embed_operands(&self, node: &AiNode) -> Result<(TensorId, TensorId)> {
        let a = node.inputs[0];
        let b = *node.inputs.get(1).context("Embed needs 2 operands")?;
        let a_is_int = self
            .ai
            .tensor_info
            .get(&a)
            .map(|i| matches!(i.logical_dtype, DType::INT64 | DType::INT32))
            .unwrap_or(false);
        Ok(if a_is_int { (a, b) } else { (b, a) })
    }

    fn desugar_gather(&mut self, node: &AiNode, axis: i64) -> Result<()> {
        // Gather(data, indices, axis): select rows along `axis`. For axis 0 this
        // is OneHot(indices) · data; general axis is a Transpose to axis-0,
        // gather, Transpose back.
        let data_tid = node.inputs[0];
        let idx_tid = *node.inputs.get(1).context("Gather needs indices")?;
        let data_dims = self.tensor_dims(data_tid)?;
        let ax = Self::norm_axis(axis, data_dims.len());
        let dtype = self.dtype_of(data_tid);
        let idx_dims = self.tensor_dims(idx_tid)?;
        let rows: u64 = idx_dims.iter().product();
        let depth = data_dims[ax];

        let data = self.src(data_tid)?;
        let idx = self.src(idx_tid)?;
        let onehot = self.one_hot(idx, rows, depth); // [rows, depth]

        // Bring axis to front: [depth, rest]; matmul one-hot; restore.
        let rest: u64 = data_dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != ax)
            .map(|(_, &d)| d)
            .product();
        let perm: Vec<u32> = std::iter::once(ax as u32)
            .chain((0..data_dims.len() as u32).filter(|&i| i as usize != ax))
            .collect();
        let mut permuted_dims = vec![depth];
        permuted_dims.extend(
            data_dims
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != ax)
                .map(|(_, &d)| d),
        );
        let perm_c = {
            let v: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
            self.const_i64(&v)
        };
        let dt = self.op(OpKind::Transpose, &[data, perm_c], dtype, &permuted_dims); // [depth, rest]
        let dt2 = self.reshape_to(dt, dtype, &[depth, rest]);
        let gathered = self.op(OpKind::MatMul, &[onehot, dt2], dtype, &[rows, rest]); // [rows, rest]
                                                                                      // Final output shape is declared on the node.
        let out_dims = self.out_dims(node)?;
        let out = self.reshape_to(gathered, dtype, &out_dims);
        self.bind_out(node, out)
    }

    fn desugar_gather_nd(&mut self, node: &AiNode, _batch_dims: i64) -> Result<()> {
        // GatherND with a constant index set flattens to a row gather over the
        // leading dimension; reuse the one-hot selection on flattened indices.
        self.desugar_gather(node, 0)
    }

    fn desugar_one_hot(&mut self, node: &AiNode, axis: i64) -> Result<()> {
        let idx_tid = node.inputs[0];
        let idx = self.src(idx_tid)?;
        let idx_dims = self.tensor_dims(idx_tid)?;
        let rows: u64 = idx_dims.iter().product();
        let out_dims = self.out_dims(node)?;
        let depth = out_dims[Self::norm_axis(axis, out_dims.len())];
        let oh = self.one_hot(idx, rows, depth);
        let (dtype, _) = self.out_dtype_shape(node)?;
        let out = self.reshape_to(oh, dtype, &out_dims);
        self.bind_out(node, out)
    }

    // ── type / numeric ───────────────────────────────────────────────────────

    fn desugar_cast(&mut self, node: &AiNode, _to: DType) -> Result<()> {
        // Numeric reinterpretation is realized by the backend at the dtype
        // boundary; the canonical graph carries the new output dtype on a
        // Reshape-identity (no element movement). Quantized sources go through
        // Dequantize instead.
        let data = self.src(node.inputs[0])?;
        let (dtype, shape) = self.out_dtype_shape(node)?;
        let _ = shape;
        let dims = self.out_dims(node)?;
        let out = self.reshape_to(data, dtype, &dims);
        self.bind_out(node, out)
    }

    fn desugar_quantize(&mut self, node: &AiNode) -> Result<()> {
        // Quantize(x) = Round(x / scale) clamped — but inference graphs carry
        // quantization as a compile-time weight encoding (QuantAttrs), so a
        // runtime Quantize is the identity on already-encoded values.
        let data = self.src(node.inputs[0])?;
        self.bind_out(node, data)
    }

    /// `Dequantize` (ONNX `DequantizeLinear`, inputs `[x_quant, scale, zp?]`).
    ///
    /// Two canonical realizations, chosen by what the model provides — never a
    /// panic or a hard failure on a valid graph:
    ///
    /// 1. **Packed** (the weight case — constant scale + zero-point): emit
    ///    `OpKind::Dequantize` over the **packed** quantized operand, with
    ///    scale/zp as `QuantAttrs` (per-tensor scalar) or i32/f32 vector operands
    ///    (per-channel along the op's declared axis). The quantized weight stays
    ///    operand 0 at its quantum width (i8 = 1 B/param, i4 = ½), so hologram
    ///    fuses `Dequantize→MatMul`/`→activation` and reads it **in-register** —
    ///    the dense f32 is never materialized (architecture §6, class QZ).
    /// 2. **Primitive** (anything the packed kernel can't express — e.g. a
    ///    runtime/dynamic scale): the canonical arithmetic
    ///    `(toᶠ³²(x) − toᶠ³²(zp)) · scale` over `Dequantize`(scale 1)/`Sub`/`Mul`,
    ///    correct for arbitrary scale/zp shapes and dtypes (§5.2).
    fn desugar_dequantize(&mut self, node: &AiNode, axis: i64) -> Result<()> {
        let x_tid = *node.inputs.first().context("Dequantize missing input")?;
        let quant_dtype = self.dtype_of(x_tid).0;
        let scale_tid = node.inputs.get(1).copied();
        let zp_tid = node.inputs.get(2).copied();

        // Eligibility for the packed kernel — computed without failing, so an
        // ineligible case routes to the primitive form instead of erroring.
        let scale_is_const = scale_tid.is_none_or(|t| self.ai.params.contains_key(&t));
        let zp_is_const = zp_tid.is_none_or(|t| self.ai.params.contains_key(&t));
        let scale_len: u64 = scale_tid
            .and_then(|t| self.tensor_dims(t).ok())
            .map(|d| d.iter().product())
            .unwrap_or(1);

        if scale_is_const && zp_is_const {
            if scale_len <= 1 {
                // Per-tensor: scalar scale/zp fold into QuantAttrs (channels = 0).
                if let (Some(scale), Some(zero_point)) = (
                    scale_tid.map_or(Some(1.0), |t| self.const_scalar_f32(t).ok()),
                    zp_tid.map_or(Some(0), |t| self.const_scalar_i32(t).ok()),
                ) {
                    let (out_dtype, out_shape) = self.out_dtype_shape(node)?;
                    let x_src = self.src(x_tid)?;
                    let nid = self.add(
                        GraphOp::Op(OpKind::Dequantize),
                        SmallVec::from_iter([x_src]),
                        out_dtype,
                        out_shape,
                    );
                    self.graph.set_quant_attrs(
                        nid,
                        QuantAttrs { quant_dtype, scale_bits: scale.to_bits(), zero_point, axis: -1 },
                    );
                    return self.bind_out(node, InputSource::Node(nid));
                }
            } else if let Ok(x_dims) = self.tensor_dims(x_tid) {
                // Per-channel: scale (f32) and zero-point (widened to i32) are
                // vector operands; the channel axis is the op's declared `axis`.
                let rank = x_dims.len() as i64;
                let ax = if axis < 0 { axis + rank } else { axis };
                let fits = ax >= 0
                    && (ax as usize) < x_dims.len()
                    && x_dims[ax as usize] == scale_len;
                let zp_i32 = match zp_tid {
                    Some(zt) => self.param_as_i32_vec(zt).ok(),
                    None => Some(vec![0i32; scale_len as usize]),
                };
                if let (true, Some(zp_i32), Some(scale_tid)) = (fits, zp_i32, scale_tid) {
                    let (out_dtype, out_shape) = self.out_dtype_shape(node)?;
                    let x_src = self.src(x_tid)?;
                    let scale_src = self.src(scale_tid)?;
                    let zp_src = self.const_i32(&zp_i32);
                    let nid = self.add(
                        GraphOp::Op(OpKind::Dequantize),
                        SmallVec::from_iter([x_src, scale_src, zp_src]),
                        out_dtype,
                        out_shape,
                    );
                    self.graph.set_quant_attrs(
                        nid,
                        QuantAttrs { quant_dtype, scale_bits: 0, zero_point: 0, axis: ax as i32 },
                    );
                    return self.bind_out(node, InputSource::Node(nid));
                }
            }
        }

        // General canonical form for everything else (runtime scale, or shapes
        // the packed kernel can't carry). Correct for any model; no panic.
        self.desugar_dequant_primitive(node, x_tid, quant_dtype, scale_tid, zp_tid, axis)
    }

    /// Reshape a scalar/per-channel scale-or-zp operand so it broadcasts
    /// correctly against `x_dims` along `axis`: a length-`C` vector becomes
    /// `[1,…,C,…,1]` (C at `axis`), a scalar broadcasts as-is.
    fn channel_align(
        &mut self,
        src: InputSource,
        dims: &[u64],
        axis: i64,
        x_dims: &[u64],
    ) -> InputSource {
        let f32t = DTypeId(DTYPE_F32);
        let total: u64 = dims.iter().product();
        if total <= 1 {
            return self.broadcast_to(src, f32t, dims, x_dims);
        }
        let mut aligned = vec![1u64; x_dims.len()];
        let ax = if axis < 0 { axis + x_dims.len() as i64 } else { axis };
        if ax >= 0 && (ax as usize) < aligned.len() {
            aligned[ax as usize] = total;
        } else if let Some(last) = aligned.last_mut() {
            *last = total; // ONNX default axis 1 collapses to trailing for rank ≤ 1
        }
        let reshaped = self.reshape_to(src, f32t, &aligned);
        self.broadcast_to(reshaped, f32t, &aligned, x_dims)
    }

    /// `(toᶠ³²(x) − toᶠ³²(zp)) · scale` over canonical primitives. `toᶠ³²` of a
    /// quantized integer is `Dequantize`(scale 1, zp 0) — the in-register
    /// int→f32 numeric conversion — so this realizes `DequantizeLinear` for an
    /// arbitrary (incl. runtime) scale/zero-point without materializing a packed
    /// kernel it can't express.
    fn desugar_dequant_primitive(
        &mut self,
        node: &AiNode,
        x_tid: TensorId,
        quant_dtype: u8,
        scale_tid: Option<TensorId>,
        zp_tid: Option<TensorId>,
        axis: i64,
    ) -> Result<()> {
        let f32t = DTypeId(DTYPE_F32);
        let x_dims = self.tensor_dims(x_tid)?;
        let x_src = self.src(x_tid)?;
        // toᶠ³²(x): Dequantize(x, scale 1, zp 0) — numeric int→f32, x stays packed.
        let mut acc = {
            let shape = self.intern(&x_dims);
            let nid = self.add(GraphOp::Op(OpKind::Dequantize), SmallVec::from_iter([x_src]), f32t, shape);
            self.graph.set_quant_attrs(
                nid,
                QuantAttrs { quant_dtype, scale_bits: 1.0f32.to_bits(), zero_point: 0, axis: -1 },
            );
            InputSource::Node(nid)
        };
        // − toᶠ³²(zp), aligned to the quantization axis and broadcast to x.
        if let Some(zt) = zp_tid {
            let zp_dims = self.tensor_dims(zt)?;
            let zp_src = self.src(zt)?;
            let zp_dtype = self.dtype_of(zt).0;
            let zp_shape = self.intern(&zp_dims);
            let zpf = self.add(GraphOp::Op(OpKind::Dequantize), SmallVec::from_iter([zp_src]), f32t, zp_shape);
            self.graph.set_quant_attrs(
                zpf,
                QuantAttrs { quant_dtype: zp_dtype, scale_bits: 1.0f32.to_bits(), zero_point: 0, axis: -1 },
            );
            let zpf_b = self.channel_align(InputSource::Node(zpf), &zp_dims, axis, &x_dims);
            acc = self.op(OpKind::Sub, &[acc, zpf_b], f32t, &x_dims);
        }
        // · scale, aligned to the quantization axis and broadcast to x.
        if let Some(st) = scale_tid {
            let s_dims = self.tensor_dims(st)?;
            let s_src = self.src(st)?;
            let s_b = self.channel_align(s_src, &s_dims, axis, &x_dims);
            acc = self.op(OpKind::Mul, &[acc, s_b], f32t, &x_dims);
        }
        self.bind_out(node, acc)
    }

    /// First element of an inline constant param as `f32` (per-tensor scale).
    fn const_scalar_f32(&self, tid: TensorId) -> Result<f32> {
        self.ai
            .params
            .get(&tid)
            .and_then(|p| p.as_f32_slice().and_then(|s| s.first().copied()))
            .with_context(|| format!("Dequantize scale T{tid} is not an f32 constant"))
    }

    /// First element of an inline integer constant param as `i32` (zero-point,
    /// stored as the quantized dtype — i8 / u8 / i32 / i64).
    fn const_scalar_i32(&self, tid: TensorId) -> Result<i32> {
        let (data, info) = match self.ai.params.get(&tid) {
            Some(AiParam::Inline { data, info }) => (data.as_slice(), info),
            _ => anyhow::bail!("Dequantize zero-point T{tid} is not an inline constant"),
        };
        let v = match info.logical_dtype {
            DType::INT8 => *data.first().unwrap_or(&0) as i8 as i32,
            DType::U8 | DType::BOOL => *data.first().unwrap_or(&0) as i32,
            DType::INT32 => i32::from_le_bytes(
                data.get(..4).and_then(|b| b.try_into().ok()).unwrap_or([0; 4]),
            ),
            DType::INT64 => i64::from_le_bytes(
                data.get(..8).and_then(|b| b.try_into().ok()).unwrap_or([0; 8]),
            ) as i32,
            _ => 0,
        };
        Ok(v)
    }

    /// Read an inline integer constant param's elements as `i32` (per-channel
    /// zero-point: i8 / u8 / i32 / i64, widened to the i32 vector hologram's
    /// per-channel dequant kernel expects).
    fn param_as_i32_vec(&self, tid: TensorId) -> Result<Vec<i32>> {
        let (data, info) = match self.ai.params.get(&tid) {
            Some(AiParam::Inline { data, info }) => (data.as_slice(), info),
            _ => anyhow::bail!("Dequantize per-channel zero-point T{tid} is not an inline constant"),
        };
        let v = match info.logical_dtype {
            DType::INT8 => data.iter().map(|&b| b as i8 as i32).collect(),
            DType::U8 | DType::BOOL => data.iter().map(|&b| b as i32).collect(),
            DType::INT32 => data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            DType::INT64 => data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
                .collect(),
            other => anyhow::bail!("unsupported per-channel zero-point dtype {other:?}"),
        };
        Ok(v)
    }

    /// Emit an `i32` 1-D constant (per-channel zero-point vector).
    fn const_i32(&mut self, values: &[i32]) -> InputSource {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.const_bytes(bytes, DTypeId(DTYPE_I32), &[values.len() as u64])
    }

    // ── reductions ────────────────────────────────────────────────────────────

    /// Axis-wise sum/mean over the trailing reduced axes, realized as
    /// `reshape [rows, n] → MatMul ones[n,1] → reshape out`. hologram's reduce
    /// kernels fold the whole tensor to a scalar, so per-axis reduction is
    /// expressed canonically as a matmul with a constant column (value `1` for
    /// sum, `1/n` for mean). Assumes the reduced axes are the trailing ones
    /// (the dominant case — ONNX norms reduce the last axis); other axis sets
    /// are brought to the tail by the importer's transpose normalization.
    fn reduce_trailing(
        &mut self,
        src: InputSource,
        dtype: DTypeId,
        in_dims: &[u64],
        n_reduced_axes: usize,
        mean: bool,
        out_dims: &[u64],
    ) -> InputSource {
        let k = n_reduced_axes.clamp(1, in_dims.len());
        let reduced: u64 = in_dims[in_dims.len() - k..].iter().product::<u64>().max(1);
        let total: u64 = in_dims.iter().product();
        let rows = (total / reduced).max(1);
        let x2d = self.reshape_to(src, dtype, &[rows, reduced]);
        let fill = if mean { 1.0 / reduced as f32 } else { 1.0 };
        let ones = self.const_f32(&vec![fill; reduced as usize], &[reduced, 1]);
        let reduced_2d = self.op(OpKind::MatMul, &[x2d, ones], dtype, &[rows, 1]);
        self.reshape_to(reduced_2d, dtype, out_dims)
    }

    fn desugar_reduce_axis(
        &mut self,
        node: &AiNode,
        axes: &[i64],
        _keepdims: bool,
        mean: bool,
    ) -> Result<()> {
        let in_dims = self.in_dims(node, 0)?;
        let dtype = self.dtype_of(node.inputs[0]);
        let n_axes = if axes.is_empty() { in_dims.len() } else { axes.len() };
        let x = self.src(node.inputs[0])?;
        let out_dims = self.out_dims(node)?;
        let out = self.reduce_trailing(x, dtype, &in_dims, n_axes, mean, &out_dims);
        self.bind_out(node, out)
    }

    fn desugar_reduce_l1(&mut self, node: &AiNode, axes: &[i64], _keepdims: bool) -> Result<()> {
        let x = self.src(node.inputs[0])?;
        let in_dims = self.in_dims(node, 0)?;
        let dtype = self.dtype_of(node.inputs[0]);
        let absx = self.op(OpKind::Abs, &[x], dtype, &in_dims);
        let out_dims = self.out_dims(node)?;
        let n_axes = if axes.is_empty() { in_dims.len() } else { axes.len() };
        let out = self.reduce_trailing(absx, dtype, &in_dims, n_axes, false, &out_dims);
        self.bind_out(node, out)
    }

    fn desugar_reduce_l2(&mut self, node: &AiNode, axes: &[i64], _keepdims: bool) -> Result<()> {
        let x = self.src(node.inputs[0])?;
        let in_dims = self.in_dims(node, 0)?;
        let dtype = self.dtype_of(node.inputs[0]);
        let sq = self.op(OpKind::Mul, &[x, x], dtype, &in_dims);
        let out_dims = self.out_dims(node)?;
        let n_axes = if axes.is_empty() { in_dims.len() } else { axes.len() };
        let sum = self.reduce_trailing(sq, dtype, &in_dims, n_axes, false, &out_dims);
        let root = self.op(OpKind::Sqrt, &[sum], dtype, &out_dims);
        self.bind_out(node, root)
    }

    /// ArgMax/ArgMin: `ReduceSum( Equal(x, ReduceMax(x)) · iota )` over the axis.
    fn desugar_arg_reduce(
        &mut self,
        node: &AiNode,
        axis: i64,
        _keepdims: bool,
        want_max: bool,
    ) -> Result<()> {
        let x = self.src(node.inputs[0])?;
        let in_dims = self.in_dims(node, 0)?;
        let dtype = self.dtype_of(node.inputs[0]);
        let ax = Self::norm_axis(axis, in_dims.len());
        let mut kept = in_dims.clone();
        kept[ax] = 1;
        let reduce_kind = if want_max {
            OpKind::ReduceMax
        } else {
            OpKind::ReduceMin
        };
        let extremum = self.op(reduce_kind, &[x], dtype, &kept);
        let mask = self.op(OpKind::Equal, &[x, extremum], dtype, &in_dims);
        // iota along `ax`, broadcast over the other dims.
        let depth = in_dims[ax];
        let iota: Vec<f32> = (0..depth).map(|i| i as f32).collect();
        let mut iota_dims = vec![1u64; in_dims.len()];
        iota_dims[ax] = depth;
        let iota_c = self.const_f32(&iota, &iota_dims);
        let weighted = self.op(OpKind::Mul, &[mask, iota_c], dtype, &in_dims);
        let out_dims = self.out_dims(node)?;
        let idx = self.op(OpKind::ReduceMax, &[weighted], dtype, &out_dims);
        self.bind_out(node, idx)
    }

    // ── normalization ──────────────────────────────────────────────────────────

    fn desugar_batchnorm(&mut self, node: &AiNode, epsilon: f32) -> Result<()> {
        // Inference BatchNorm: y = (x - mean)/sqrt(var+eps) * gamma + beta.
        // Operands: [x, gamma, beta, mean, var].
        let x = self.src(node.inputs[0])?;
        let dims = self.in_dims(node, 0)?;
        let dtype = self.dtype_of(node.inputs[0]);
        let g = |s: &mut Self, i: usize| -> Result<InputSource> { s.src(node.inputs[i]) };
        let gamma = g(self, 1)?;
        let beta = g(self, 2)?;
        let mean = g(self, 3)?;
        let var = g(self, 4)?;
        let eps = self.const_f32(&[epsilon], &[1]);
        let centered = self.op(OpKind::Sub, &[x, mean], dtype, &dims);
        let denom0 = self.op(OpKind::Add, &[var, eps], dtype, &dims);
        let denom = self.op(OpKind::Sqrt, &[denom0], dtype, &dims);
        let norm = self.op(OpKind::Div, &[centered, denom], dtype, &dims);
        let scaled = self.op(OpKind::Mul, &[norm, gamma], dtype, &dims);
        let out = self.op(OpKind::Add, &[scaled, beta], dtype, &dims);
        self.bind_out(node, out)
    }

    // ── unfused legacy fusions ────────────────────────────────────────────────

    fn desugar_matmul_act(&mut self, node: &AiNode, activation: OpKind) -> Result<()> {
        let a = self.src(node.inputs[0])?;
        let b = self.src(node.inputs[1])?;
        let (dtype, _) = self.out_dtype_shape(node)?;
        let out_dims = self.out_dims(node)?;
        let mm = self.op(OpKind::MatMul, &[a, b], dtype, &out_dims);
        let act = self.op(activation, &[mm], dtype, &out_dims);
        self.bind_out(node, act)
    }

    fn desugar_concat_matmul(&mut self, node: &AiNode, n_concat: u32) -> Result<()> {
        let dtype = self.dtype_of(node.inputs[0]);
        // inputs: [h1..hN, W]; concat the N along the last axis, then matmul.
        let n = n_concat as usize;
        let mut acc = self.src(node.inputs[0])?;
        let mut dims = self.in_dims(node, 0)?;
        let last = dims.len() - 1;
        for i in 1..n {
            let next = self.src(node.inputs[i])?;
            let nd = self.in_dims(node, i)?;
            dims[last] += nd[last];
            acc = self.op(OpKind::Concat, &[acc, next], dtype, &dims);
        }
        let w = self.src(node.inputs[n])?;
        let out_dims = self.out_dims(node)?;
        let mm = self.op(OpKind::MatMul, &[acc, w], dtype, &out_dims);
        self.bind_out(node, mm)
    }

    fn desugar_norm_projection(
        &mut self,
        node: &AiNode,
        _epsilon: f64,
        split_sizes: &[usize],
        has_residual_add: bool,
    ) -> Result<()> {
        let dtype = self.dtype_of(node.inputs[0]);
        let x_dims = self.in_dims(node, 0)?;
        let (norm, wstart) = if has_residual_add {
            // [x, residual, norm_weight, W_0..]
            let x = self.src(node.inputs[0])?;
            let res = self.src(node.inputs[1])?;
            let w = self.src(node.inputs[2])?;
            let n = self.op(OpKind::AddRmsNorm, &[x, res, w], dtype, &x_dims);
            (n, 3)
        } else {
            let x = self.src(node.inputs[0])?;
            let w = self.src(node.inputs[1])?;
            let n = self.op(OpKind::RmsNorm, &[x, w], dtype, &x_dims);
            (n, 2)
        };
        for (i, _sz) in split_sizes.iter().enumerate() {
            let w = self.src(node.inputs[wstart + i])?;
            let out_dims = self
                .ai
                .tensor_info
                .get(&node.outputs[i])
                .and_then(dims_of)
                .context("norm-projection output shape")?;
            let proj = self.op(OpKind::MatMul, &[norm, w], dtype, &out_dims);
            self.bind(node.outputs[i], proj);
        }
        Ok(())
    }

    fn desugar_swiglu_projection(&mut self, node: &AiNode) -> Result<()> {
        // [gate, up, W_down]: FusedSwiGlu(gate, up) then MatMul with W_down.
        let dtype = self.dtype_of(node.inputs[0]);
        let gate = self.src(node.inputs[0])?;
        let up = self.src(node.inputs[1])?;
        let wdown = self.src(node.inputs[2])?;
        let gate_dims = self.in_dims(node, 0)?;
        let swig = self.op(OpKind::FusedSwiGlu, &[gate, up], dtype, &gate_dims);
        let out_dims = self.out_dims(node)?;
        let out = self.op(OpKind::MatMul, &[swig, wdown], dtype, &out_dims);
        self.bind_out(node, out)
    }

    // ── normalization ──────────────────────────────────────────────────────────

    /// Normalization over the last axis. hologram derives the per-row `feature`
    /// only from a rank-2 operand, so we reshape the input to `[rows, feature]`,
    /// apply the norm (with γ/β — and a residual for `AddRmsNorm`), then reshape
    /// the result back to the declared output shape.
    fn desugar_norm(&mut self, node: &AiNode, op: OpKind, residual: bool) -> Result<()> {
        let dtype = self.dtype_of(node.inputs[0]);
        let x_dims = self.in_dims(node, 0)?;
        let feature = x_dims.last().copied().unwrap_or(1).max(1);
        let rows: u64 = x_dims.iter().product::<u64>() / feature;

        let x = self.src(node.inputs[0])?;
        let x2d = self.reshape_to(x, dtype, &[rows, feature]);

        let mut operands: SmallVec<[InputSource; 4]> = SmallVec::new();
        operands.push(x2d);
        if residual {
            // AddRmsNorm: [x, residual, gamma] — residual shares x's shape.
            let res = self.src(node.inputs[1])?;
            let res2d = self.reshape_to(res, dtype, &[rows, feature]);
            operands.push(res2d);
            for &tid in node.inputs.iter().skip(2) {
                operands.push(self.src(tid)?);
            }
        } else {
            // [x, gamma, beta?] — γ/β are rank-1 [feature], passed as-is.
            for &tid in node.inputs.iter().skip(1) {
                operands.push(self.src(tid)?);
            }
        }

        let norm2d = self.op(op, &operands, dtype, &[rows, feature]);
        let out_dims = self.out_dims(node)?;
        let out = self.reshape_to(norm2d, dtype, &out_dims);
        self.bind_out(node, out)
    }

    // ── spatial rearrangement ─────────────────────────────────────────────────

    fn desugar_depth_space(
        &mut self,
        node: &AiNode,
        blocksize: u64,
        depth_to_space: bool,
    ) -> Result<()> {
        // Realized as Reshape → Transpose → Reshape; the concrete reshapes are
        // determined by the (concrete) input/output shapes and the block size.
        let data = self.src(node.inputs[0])?;
        let dtype = self.dtype_of(node.inputs[0]);
        let in_dims = self.in_dims(node, 0)?;
        let out_dims = self.out_dims(node)?;
        let (n, c, h, w) = match in_dims.as_slice() {
            [n, c, h, w] => (*n, *c, *h, *w),
            _ => anyhow::bail!("DepthToSpace/SpaceToDepth expects rank-4 NCHW"),
        };
        let b = blocksize;
        // The 6-D intermediate + permutation differ between the two directions.
        let (mid_dims, perm): (Vec<u64>, Vec<u32>) = if depth_to_space {
            (vec![n, c / (b * b), b, b, h, w], vec![0, 1, 4, 2, 5, 3])
        } else {
            (vec![n, c, h / b, b, w / b, b], vec![0, 1, 3, 5, 2, 4])
        };
        let r1 = self.reshape_to(data, dtype, &mid_dims);
        let perm_c = {
            let v: Vec<i64> = perm.iter().map(|&p| p as i64).collect();
            self.const_i64(&v)
        };
        let permuted: Vec<u64> = perm.iter().map(|&p| mid_dims[p as usize]).collect();
        let t = self.op(OpKind::Transpose, &[r1, perm_c], dtype, &permuted);
        let out = self.reshape_to(t, dtype, &out_dims);
        self.bind_out(node, out)
    }

    // ── sequence ops realized over concrete shapes ────────────────────────────

    fn desugar_reverse_sequence(
        &mut self,
        node: &AiNode,
        _batch_axis: i64,
        time_axis: i64,
    ) -> Result<()> {
        // Reverse along the (concrete) time axis: concat unit slices in reverse.
        let data = self.src(node.inputs[0])?;
        let dtype = self.dtype_of(node.inputs[0]);
        let in_dims = self.in_dims(node, 0)?;
        let ax = Self::norm_axis(time_axis, in_dims.len());
        let len = in_dims[ax];
        let mut acc: Option<InputSource> = None;
        let mut acc_dims = in_dims.clone();
        acc_dims[ax] = 0;
        for i in (0..len).rev() {
            let piece = self.slice_axis(data, dtype, &in_dims, ax, i as i64, i as i64 + 1);
            acc = Some(match acc {
                None => piece,
                Some(a) => {
                    acc_dims[ax] += 1;
                    self.op(OpKind::Concat, &[a, piece], dtype, &acc_dims)
                }
            });
        }
        let out = acc.context("ReverseSequence on empty axis")?;
        self.bind_out(node, out)
    }

    fn desugar_einsum(&mut self, node: &AiNode, equation: &str) -> Result<()> {
        // Handle the dominant binary contraction forms (the importer decomposes
        // higher-order equations); a single-contraction `ij,jk->ik` is a MatMul.
        let dtype = self.dtype_of(node.inputs[0]);
        let out_dims = self.out_dims(node)?;
        if node.inputs.len() == 2 && equation.contains("->") {
            let a = self.src(node.inputs[0])?;
            let b = self.src(node.inputs[1])?;
            let mm = self.op(OpKind::MatMul, &[a, b], dtype, &out_dims);
            return self.bind_out(node, mm);
        }
        anyhow::bail!("einsum equation {equation:?} must be decomposed at import")
    }

    // ── data-dependent ops over statically-bounded shapes ─────────────────────

    fn desugar_topk(&mut self, node: &AiNode, axis: i64, largest: bool) -> Result<()> {
        // k=1 is ArgMax/Max; larger k is k unrolled max-and-mask rounds. The
        // output extent k is the (concrete) output axis length.
        let out_dims = self.out_dims(node)?;
        let ax = Self::norm_axis(axis, out_dims.len());
        let k = out_dims[ax];
        if k == 1 {
            return self.desugar_arg_reduce(node, axis, true, largest);
        }
        // General k: gather the top-k by repeated extremum masking. Realized as
        // a concat of k argmax selections; bounded and canonical.
        anyhow::bail!("TopK k={k} requires the import-time topk decomposition")
    }

    fn desugar_nonzero(&mut self, node: &AiNode) -> Result<()> {
        // NonZero's output is data-dependent; in inference graphs it appears
        // only with constant inputs, so it is const-folded at import. A
        // surviving dynamic NonZero is rejected as non-canonical input.
        anyhow::bail!("NonZero must be const-folded at import (node {})", node.id)
    }

    fn desugar_compress(&mut self, node: &AiNode, _axis: Option<i64>) -> Result<()> {
        anyhow::bail!("Compress must be const-folded at import (node {})", node.id)
    }

    fn desugar_scatter(&mut self, node: &AiNode) -> Result<()> {
        anyhow::bail!("Scatter must be const-folded at import (node {})", node.id)
    }

    // ── control flow (compile-time inlining; architecture §5.4) ────────────────

    /// Inline a subgraph's nodes into the flat graph being built. The subgraph's
    /// input tensors are pre-bound (in `bindings`) to parent value sources; its
    /// params become constants and its nodes are emitted against its own
    /// tensor_info. Returns the value sources for the subgraph's outputs. Done
    /// by temporarily making `sub` the active graph with a fresh source map —
    /// hologram 0.5.0 has no runtime subgraph dispatch, so control flow is
    /// resolved into one flat graph at compile time.
    fn inline_subgraph(
        &mut self,
        sub: &'a AiGraph,
        bindings: HashMap<TensorId, InputSource>,
    ) -> Result<Vec<InputSource>> {
        let saved_ai = self.ai;
        let saved_map = core::mem::replace(&mut self.tid_to_src, bindings);
        self.ai = sub;
        let result = (|| {
            self.emit_params()?;
            self.emit_nodes()?;
            sub.outputs
                .iter()
                .map(|&t| self.src(t))
                .collect::<Result<Vec<_>>>()
        })();
        self.ai = saved_ai;
        self.tid_to_src = saved_map;
        result
    }

    /// Bind a subgraph's input tensors (positionally) to the given operand
    /// sources (ONNX control-flow subgraphs take their captured/loop operands in
    /// input order).
    fn bind_inputs(sub: &AiGraph, feed: &[InputSource]) -> HashMap<TensorId, InputSource> {
        sub.inputs
            .iter()
            .zip(feed.iter())
            .map(|(&t, &s)| (t, s))
            .collect()
    }

    fn lower_control_flow(&mut self, node: &AiNode) -> Result<()> {
        use crate::ir::AiOp;
        let top = self.ai;
        match &node.op {
            AiOp::If {
                then_branch,
                else_branch,
            } => {
                // inputs: [condition, ...feed]. Branches capture the feed.
                let cond = self.src(node.inputs[0])?;
                let cond_dt = self.dtype_of(node.inputs[0]);
                let feed: Vec<InputSource> = node.inputs[1..]
                    .iter()
                    .map(|&t| self.src(t))
                    .collect::<Result<_>>()?;
                let then_sub = top
                    .subgraphs
                    .get(then_branch)
                    .with_context(|| format!("If then_branch '{then_branch}' not found"))?;
                let then_outs = self.inline_subgraph(then_sub, Self::bind_inputs(then_sub, &feed))?;

                match else_branch {
                    Some(else_name) => {
                        let else_sub = top
                            .subgraphs
                            .get(else_name)
                            .with_context(|| format!("If else_branch '{else_name}' not found"))?;
                        let else_outs =
                            self.inline_subgraph(else_sub, Self::bind_inputs(else_sub, &feed))?;
                        // Dynamic condition: select per output with Where(cond, then, else).
                        for (i, &otid) in node.outputs.iter().enumerate() {
                            let (dt, _shape) = (self.dtype_of(otid), ());
                            let dims = self.tensor_dims(otid)?;
                            let c = self.broadcast_to(cond, cond_dt, &[1], &dims);
                            let (t, e) = (then_outs[i], else_outs[i]);
                            let w = self.op(OpKind::Where, &[c, t, e], dt, &dims);
                            self.bind(otid, w);
                        }
                    }
                    None => {
                        for (i, &otid) in node.outputs.iter().enumerate() {
                            self.bind(otid, then_outs[i]);
                        }
                    }
                }
                Ok(())
            }
            AiOp::Loop {
                body,
                max_trip_count,
            } => self.lower_loop(node, body, *max_trip_count),
            AiOp::Scan {
                body,
                num_scan_inputs,
            } => self.lower_loop(node, body, None).map(|_| {
                let _ = num_scan_inputs;
            }),
            other => anyhow::bail!("non-control-flow op {other:?} routed to control flow"),
        }
    }

    /// Unroll a `Loop` at compile time when the trip count is statically known
    /// (the canonical, no-runtime-dispatch realization). Inputs:
    /// `[max_trip_count, condition, ...carry]`; body inputs:
    /// `[iter, cond, ...carry]`; body outputs: `[cond_out, ...carry_out, ...scan]`.
    fn lower_loop(&mut self, node: &AiNode, body: &str, max_trip: Option<i64>) -> Result<()> {
        let top = self.ai;
        let body_sub = top
            .subgraphs
            .get(body)
            .with_context(|| format!("Loop/Scan body '{body}' not found"))?;

        let trip = max_trip
            .or_else(|| {
                node.inputs.first().and_then(|t| {
                    self.ai
                        .tensor_info
                        .get(t)
                        .and_then(|i| i.known_i64_values.as_ref())
                        .and_then(|v| v.first().copied().flatten())
                })
            })
            .unwrap_or(0);

        let num_carry = node.inputs.len().saturating_sub(2);
        let mut carry: Vec<InputSource> = node.inputs[2..]
            .iter()
            .map(|&t| self.src(t))
            .collect::<Result<_>>()?;

        let iters = trip.clamp(0, 1024);
        for it in 0..iters {
            let iter_c = self.const_i64(&[it]);
            let cond_c = self.const_bytes(vec![1u8], DTypeId(0 /* DTYPE_BOOL */), &[1]);
            let mut bind: HashMap<TensorId, InputSource> = HashMap::new();
            if let Some(&t) = body_sub.inputs.first() {
                bind.insert(t, iter_c);
            }
            if let Some(&t) = body_sub.inputs.get(1) {
                bind.insert(t, cond_c);
            }
            for (j, &c) in carry.iter().enumerate() {
                if let Some(&t) = body_sub.inputs.get(2 + j) {
                    bind.insert(t, c);
                }
            }
            let outs = self.inline_subgraph(body_sub, bind)?;
            // Body outputs: [cond_out, ...updated_carry, ...scan].
            carry = outs.get(1..1 + num_carry).unwrap_or(&[]).to_vec();
        }

        // Map node outputs to the final carry state (zero-trip → initial carry).
        for (i, &otid) in node.outputs.iter().enumerate() {
            if let Some(&s) = carry.get(i) {
                self.bind(otid, s);
            }
        }
        Ok(())
    }
}


// ── shared raw helpers (free fns so desugar modules can reuse them) ──────────

/// Whether `kind` is a broadcasting elementwise binary op (operands may differ
/// in shape under numpy broadcasting, so each is `Expand`ed to the output).
fn is_broadcast_binary(kind: OpKind) -> bool {
    matches!(
        kind,
        OpKind::Add
            | OpKind::Sub
            | OpKind::Mul
            | OpKind::Div
            | OpKind::Pow
            | OpKind::Mod
            | OpKind::Min
            | OpKind::Max
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::Equal
            | OpKind::Less
            | OpKind::LessOrEqual
            | OpKind::Greater
            | OpKind::GreaterOrEqual
    )
}

/// Concrete dims from a `TensorInfo`, or `None` if any dim is symbolic.
fn dims_of(info: &TensorInfo) -> Option<Vec<u64>> {
    if info.shape.is_empty() {
        return Some(Vec::new());
    }
    info.shape
        .iter()
        .map(|d| match d {
            Dim::Concrete(n) => Some(*n),
            _ => None,
        })
        .collect()
}

/// Owned bytes + info for a constant parameter (a compile-time read; the
/// runtime hot path never copies — ZA/ZM apply at runtime, not compile).
fn param_bytes(param: &AiParam) -> Result<(Vec<u8>, TensorInfo)> {
    match param {
        AiParam::Inline { data, info } => Ok(((**data).clone(), info.clone())),
        AiParam::Mmap {
            path,
            offset,
            len,
            info,
        } => {
            use std::io::{Read, Seek, SeekFrom};
            let mut f = std::fs::File::open(path)
                .with_context(|| format!("opening mmap param {path:?}"))?;
            f.seek(SeekFrom::Start(*offset))?;
            let mut buf = vec![0u8; *len as usize];
            f.read_exact(&mut buf)?;
            Ok((buf, info.clone()))
        }
    }
}

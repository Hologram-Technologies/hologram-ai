use crate::ir::AiGraph;
use rayon::prelude::*;

/// A single optimization pass over `AiGraph`.
pub trait Pass: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, graph: AiGraph) -> anyhow::Result<AiGraph>;
}

/// Sequentially-composed optimization pipeline.
pub struct OptPipeline {
    passes: Vec<Box<dyn Pass>>,
}

impl OptPipeline {
    pub fn new(passes: Vec<Box<dyn Pass>>) -> Self {
        Self { passes }
    }

    /// Standard optimization pipeline.
    pub fn mvp() -> Self {
        use super::{
            add_rmsnorm_fusion::AddRmsNormFusion, attention_fusion::AttentionFusion,
            const_dedup::ConstantDeduplication, const_eval::ConstantEvaluation,
            constant_fold::ConstantFolding, data_prop::DataPropagation,
            dead_node::DeadNodeElimination, decompose::OpDecomposition,
            kv_slot_injection::KvSlotInjection,
            layernorm_fusion::LayerNormFusion,
            matmul_activation_fusion::MatMulActivationFusion,
            norm_projection_fusion::NormProjectionFusion,
            position_ids_injection::PositionIdsInjection, resolve_slice_params::ResolveSliceParams,
            rmsnorm_fusion::RmsNormFusion, scalar_absorption::ScalarAbsorption,
            semantic_prop::SemanticPropagation, shape_prop::ShapePropagation,
            shared_input_projection_fusion::SharedInputProjectionFusion,
            swiglu_fusion::SwiGluFusion, swiglu_projection_fusion::SwiGluProjectionFusion,
            transpose_matmul_fusion::TransposeMatMulFusion,
        };
        Self::new(vec![
            // Resolve ONNX opset 10+ Slice params from constant inputs
            // before shape propagation needs the concrete slice bounds.
            Box::new(ResolveSliceParams),
            Box::new(ShapePropagation),
            Box::new(DataPropagation),
            // Second shape pass: DataPropagation fills known_i64_values for
            // Reshape/Expand shape tensors. This pass uses them to infer
            // output shapes that the first pass couldn't resolve.
            Box::new(ShapePropagation),
            // Evaluate nodes with all-constant inputs at compile time.
            // Handles N-D broadcast that the runtime can't do.
            Box::new(ConstantEvaluation),
            Box::new(ConstantFolding),
            // Fuse explicit Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul chains
            // into AiOp::RmsNorm. Must run after ConstantFolding so that the
            // scalar epsilon and exponent params are already materialized as
            // AiParam::Inline (otherwise scalar_f32_param returns None).
            // Fuse explicit ReduceMean→Sub→Pow→ReduceMean→Add→Sqrt→Div→Mul→Add
            // chains into AiOp::LayerNorm. Runs BEFORE RmsNormFusion because
            // LayerNorm is a superset pattern — RmsNormFusion would otherwise
            // consume the inner Pow→ReduceMean→Sqrt→Div subchain.
            Box::new(LayerNormFusion),
            Box::new(RmsNormFusion),
            // Fuse SiLU(gate) * up → FusedSwiGLU. Runs after RmsNormFusion
            // so norm chains are already collapsed. Must run before
            // AttentionFusion to avoid interfering with SDPA pattern matching.
            Box::new(SwiGluFusion),
            // Absorb Transpose(swap-last-2) → MatMul into Gemm { trans_a/trans_b }.
            // Eliminates intermediate transposed buffer. Must run before
            // MatMulActivationFusion (which matches on MatMul nodes).
            Box::new(TransposeMatMulFusion),
            // Absorb MatMul → Mul(scalar) into Gemm { alpha }.
            // Eliminates full-tensor scalar multiply. Must run before other
            // matmul fusions to simplify the graph.
            Box::new(ScalarAbsorption),
            // Fuse MatMul → SiLU/GeLU/ReLU into MatMulSilu/Gelu/Relu.
            // Eliminates intermediate activation buffer; the tape kernel
            // applies activation in-register during matmul writeback.
            Box::new(MatMulActivationFusion),
            // Fuse Add(x, residual) → RmsNorm(sum, weight, eps) into
            // FusedLayerNormResidual. Runs after RmsNormFusion (needs fused
            // RmsNorm nodes) and before AttentionFusion.
            Box::new(AddRmsNormFusion),
            // Deep decode fusions (Plan 054):
            // Fuse [Add+]RmsNorm → multi-way MatMul projection.
            // Lowered as MultiOutput: 1 norm node + N MatMul nodes sharing
            // the norm output. No weight concatenation — original params reused.
            Box::new(NormProjectionFusion),
            // Fuse FusedSwiGLU → MatMul (down projection).
            Box::new(SwiGluProjectionFusion),
            // Fuse shared-input MatMul projections:
            // QKV: 3 MatMuls → 1 MatMul + 3 Slices (saves 44 BLAS calls)
            // Gate+Up: 2 MatMuls → 1 MatMul + 2 Slices (saves 22 BLAS calls)
            Box::new(SharedInputProjectionFusion),
            // Replace Range(0, seq, 1) position generators with a position_ids
            // input. Enables KV cache decode at seq=1 by passing the correct
            // absolute position from the generation loop.
            Box::new(PositionIdsInjection),
            Box::new(AttentionFusion),
            Box::new(KvSlotInjection),
            // Infer semantic hints (Embedding, AttentionWeight, Residual, etc.)
            // from op types. Runs after all fusion passes so fused ops are present.
            Box::new(SemanticPropagation),
            // Decompose compound ops (ReduceL1/L2, DepthToSpace, SpaceToDepth)
            // into primitive ops before lowering.
            Box::new(OpDecomposition),
            // Deduplicate identical constants by content hash.
            // Cross-layer duplicates (e.g., RoPE constants computed per
            // transformer layer) share the same bytes but have different
            // TensorIds, so op-based CSE can't catch them.
            Box::new(ConstantDeduplication),
            Box::new(DeadNodeElimination),
        ])
    }

    /// Generic optimization pipeline for non-transformer components.
    ///
    /// Runs shape/data propagation, constant evaluation/folding, op
    /// decomposition, and dead node elimination. Skips attention fusion,
    /// KV-cache injection, and other LLM-specific passes.
    pub fn generic() -> Self {
        use super::{
            const_dedup::ConstantDeduplication, const_eval::ConstantEvaluation,
            constant_fold::ConstantFolding, data_prop::DataPropagation,
            dead_node::DeadNodeElimination, decompose::OpDecomposition,
            resolve_slice_params::ResolveSliceParams, semantic_prop::SemanticPropagation,
            shape_prop::ShapePropagation,
        };
        Self::new(vec![
            Box::new(ResolveSliceParams),
            Box::new(ShapePropagation),
            Box::new(DataPropagation),
            Box::new(ShapePropagation),
            Box::new(ConstantEvaluation),
            Box::new(ConstantFolding),
            Box::new(OpDecomposition),
            Box::new(SemanticPropagation),
            Box::new(ConstantDeduplication),
            Box::new(DeadNodeElimination),
        ])
    }

    /// Run all passes in order, short-circuiting on error.
    ///
    /// After running all passes on the main graph, recursively runs the same
    /// pipeline on each subgraph (If branches, Loop/Scan bodies). This ensures
    /// all optimization passes (shape prop, data prop, constant folding, dead
    /// node elimination, etc.) apply to subgraphs too.
    pub fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        for pass in &self.passes {
            tracing::debug!(pass = pass.name(), "running opt pass");
            graph = pass.run(graph)?;
        }

        // Recurse into subgraphs (If/Loop/Scan bodies) in parallel.
        // Each subgraph is independent — no cross-subgraph data flow.
        if !graph.subgraphs.is_empty() {
            let subgraphs: Vec<(String, AiGraph)> =
                std::mem::take(&mut graph.subgraphs).into_iter().collect();
            let optimized: Vec<(String, anyhow::Result<AiGraph>)> = subgraphs
                .into_par_iter()
                .map(|(key, sub)| {
                    tracing::debug!(subgraph = %key, "optimizing subgraph");
                    let result = self.run(sub);
                    (key, result)
                })
                .collect();
            for (key, result) in optimized {
                graph.subgraphs.insert(key, result?);
            }
        }

        Ok(graph)
    }
}

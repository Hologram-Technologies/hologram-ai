use crate::ir::AiGraph;

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
            const_dedup::ConstantDeduplication, const_eval::ConstantEvaluation,
            constant_fold::ConstantFolding, data_prop::DataPropagation,
            dead_node::DeadNodeElimination, rmsnorm_fusion::RmsNormFusion,
            shape_prop::ShapePropagation,
        };
        Self::new(vec![
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
            Box::new(RmsNormFusion),
            // Deduplicate identical constants by content hash.
            // Cross-layer duplicates (e.g., RoPE constants computed per
            // transformer layer) share the same bytes but have different
            // TensorIds, so op-based CSE can't catch them.
            Box::new(ConstantDeduplication),
            Box::new(DeadNodeElimination),
        ])
    }

    /// Run all passes in order, short-circuiting on error.
    pub fn run(&self, mut graph: AiGraph) -> anyhow::Result<AiGraph> {
        for pass in &self.passes {
            tracing::debug!(pass = pass.name(), "running opt pass");
            graph = pass.run(graph)?;
        }
        Ok(graph)
    }
}

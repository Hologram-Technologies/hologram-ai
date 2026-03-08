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
            constant_fold::ConstantFolding,
            dead_node::DeadNodeElimination,
            shape_prop::ShapePropagation,
        };
        Self::new(vec![
            Box::new(ShapePropagation),
            Box::new(ConstantFolding),
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

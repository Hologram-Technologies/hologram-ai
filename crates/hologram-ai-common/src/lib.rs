//! hologram-ai-common: canonical AI IR, optimization passes, memory planner, and lowering.
//!
//! This crate is the compiler core shared by all importers and the `hologram-ai` facade.
//! It does NOT import hologram subcrates directly — only the root `hologram` crate.

pub mod error;
pub mod exec_context;
pub mod ir;
pub mod lower;
pub mod mem;
pub mod opt;
pub mod rules;
pub mod sections;

// Flat re-exports for convenience.
pub use error::CommonError;
pub use exec_context::{ContextBundle, PatchPruneContext, Section, SECTION_PATCH_PRUNE};
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme, ScaleDtype};
pub use ir::{
    canonical_vars, shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, ConstraintStore, DType,
    Dim, DimExpr, DimVarEntry, DimVarId, DimVarSource, DimVarTable, ImportWarning, KvLayout,
    MetaValue, NodeId, ScatterReduce, SemanticHint, Shape, ShapeConstraint, ShapeError, TensorId,
    TensorInfo, ValidationError,
};
pub use lower::{lower, LowerPhase, LoweringOptions, LoweringOutput, OptProfile, QuantStrategy};
pub use mem::{KvCacheLayout, MemoryPlan, MemoryPlanner};
pub use opt::{
    AggressiveShapePropagation, ConstantDeduplication, ExplicitBroadcastBinary, OptPipeline, Pass,
    ShapeHealing, ShapeOraclePass,
};
pub use rules::{pattern_rules::slice_to_gather_rules, RulePass};

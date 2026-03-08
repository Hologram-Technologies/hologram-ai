//! hologram-ai-common: canonical AI IR, optimization passes, memory planner, and lowering.
//!
//! This crate is the compiler core shared by all importers and the `hologram-ai` facade.
//! It does NOT import hologram subcrates directly — only the root `hologram` crate.

pub mod ir;
pub mod opt;
pub mod mem;
pub mod lower;
pub mod error;

// Flat re-exports for convenience.
pub use ir::{
    AiGraph, AiNode, AiOp, AiParam,
    DType, Dim, DimExpr, DimVarId, Shape, TensorInfo, TensorId, NodeId,
    DimVarTable, DimVarEntry, DimVarSource, canonical_vars,
    ShapeConstraint, ConstraintStore, ShapeError,
    ImportWarning, MetaValue, ValidationError, ScatterReduce,
    shape_from_concrete,
};
pub use opt::{OptPipeline, Pass};
pub use mem::{KvCacheLayout, MemoryPlan, MemoryPlanner};
pub use lower::{lower, LoweringOptions, LoweringOutput, QuantStrategy};
pub use error::CommonError;
pub use hologram_ai_quant::{QuantDescriptor, QuantScheme, ScaleDtype};

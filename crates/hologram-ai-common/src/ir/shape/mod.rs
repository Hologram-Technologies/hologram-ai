pub mod constraint;
pub mod dim_expr;
pub mod dim_var;

pub use constraint::{ConstraintStore, ShapeConstraint, ShapeError};
pub use dim_expr::{DimExpr, DimVarId};
pub use dim_var::{DimVarEntry, DimVarSource, DimVarTable};

use smallvec::SmallVec;

/// Compact tensor shape — up to 4 dims inline before heap allocation.
/// Reduced from `SmallVec<[Dim; 6]>` because `DimExpr` is larger per element.
pub type Shape = SmallVec<[DimExpr; 4]>;

/// Construct a concrete `Shape` from a slice of `u64`.
pub fn shape_from_concrete(dims: &[u64]) -> Shape {
    dims.iter().copied().map(DimExpr::Concrete).collect()
}

/// Legacy `Dim` compatibility — maps directly to `DimExpr` variants.
pub type Dim = DimExpr;

/// Canonical dimension variable names used across importers.
pub mod canonical_vars {
    pub const BATCH: &str = "batch";
    pub const SEQ_LEN: &str = "seq_len";
    pub const VOCAB_SIZE: &str = "vocab_size";
    pub const HIDDEN_DIM: &str = "hidden_dim";
    pub const NUM_HEADS: &str = "num_heads";
    pub const NUM_KV_HEADS: &str = "num_kv_heads";
    pub const HEAD_DIM: &str = "head_dim";
    pub const FFN_DIM: &str = "ffn_dim";
}

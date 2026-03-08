use super::dim_expr::DimExpr;

/// Error type for shape-related failures.
#[derive(Debug, thiserror::Error)]
pub enum ShapeError {
    #[error("variable '{var}' value {value} violates bounds [{lower:?}, {upper:?}]")]
    BoundsViolation {
        var: String,
        value: u64,
        lower: Option<u64>,
        upper: Option<u64>,
    },
    #[error("variable '{var}' has no upper bound — cannot concretize")]
    NoBound { var: String },
    #[error("constraint violated: {message}")]
    ConstraintViolation { message: String },
}

/// A shape constraint collected during shape propagation.
#[derive(Debug, Clone)]
pub enum ShapeConstraint {
    /// Two dimensions must be equal (e.g., MatMul inner dimension).
    DimEqual {
        a: DimExpr,
        b: DimExpr,
        reason: String,
    },
    /// Two shapes must be broadcast-compatible.
    BroadcastCompatible {
        a: DimExpr,
        b: DimExpr,
        reason: String,
    },
    /// `a` must be divisible by `b` (e.g., Reshape inferred dimension).
    Divisible {
        a: DimExpr,
        b: DimExpr,
        reason: String,
    },
    /// Product of one shape must equal product of another (Reshape).
    ProductEqual {
        a_dims: Vec<DimExpr>,
        b_dims: Vec<DimExpr>,
        reason: String,
    },
}

/// Collection of shape constraints for validation.
#[derive(Debug, Clone, Default)]
pub struct ConstraintStore {
    constraints: Vec<ShapeConstraint>,
}

impl ConstraintStore {
    pub fn push(&mut self, c: ShapeConstraint) {
        self.constraints.push(c);
    }

    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ShapeConstraint> {
        self.constraints.iter()
    }
}

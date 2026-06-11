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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BroadcastConflictPolicy {
    Dynamic,
    Left,
    Right,
}

impl BroadcastConflictPolicy {
    fn resolve(self, left: &DimExpr, right: &DimExpr) -> DimExpr {
        match self {
            Self::Dynamic => DimExpr::Dynamic,
            Self::Left => left.clone(),
            Self::Right => right.clone(),
        }
    }
}

pub trait SymbolicShapeExt {
    fn is_concrete(&self) -> bool;
    fn concrete_dims(&self) -> Option<Vec<u64>>;
    fn broadcast_to(&self, target: &[DimExpr]) -> Option<Shape>;
    fn broadcast_shape(&self, other: &[DimExpr]) -> Shape;
    fn broadcast_shape_with(&self, other: &[DimExpr], conflict: BroadcastConflictPolicy) -> Shape;
}

impl SymbolicShapeExt for [DimExpr] {
    fn is_concrete(&self) -> bool {
        self.iter().all(DimExpr::is_concrete)
    }

    fn concrete_dims(&self) -> Option<Vec<u64>> {
        self.iter().map(DimExpr::as_concrete).collect()
    }

    fn broadcast_to(&self, target: &[DimExpr]) -> Option<Shape> {
        if self.len() > target.len() {
            return None;
        }

        let one = DimExpr::Concrete(1);
        for offset in 0..target.len() {
            let src = self
                .len()
                .checked_sub(offset + 1)
                .and_then(|idx| self.get(idx))
                .unwrap_or(&one);
            let dst = &target[target.len() - 1 - offset];
            if src == dst || src.as_concrete() == Some(1) {
                continue;
            }
            return None;
        }

        Some(target.iter().cloned().collect())
    }

    fn broadcast_shape(&self, other: &[DimExpr]) -> Shape {
        self.broadcast_shape_with(other, BroadcastConflictPolicy::Dynamic)
    }

    fn broadcast_shape_with(&self, other: &[DimExpr], conflict: BroadcastConflictPolicy) -> Shape {
        let len = self.len().max(other.len());
        let one = DimExpr::Concrete(1);
        let mut result = Shape::with_capacity(len);

        for offset in 0..len {
            let left = self
                .len()
                .checked_sub(offset + 1)
                .and_then(|idx| self.get(idx))
                .unwrap_or(&one);
            let right = other
                .len()
                .checked_sub(offset + 1)
                .and_then(|idx| other.get(idx))
                .unwrap_or(&one);
            let dim = if left == right {
                left.clone()
            } else {
                match (left.as_concrete(), right.as_concrete()) {
                    (Some(1), _) => right.clone(),
                    (_, Some(1)) => left.clone(),
                    (Some(lv), Some(rv)) if lv == rv => left.clone(),
                    _ => conflict.resolve(left, right),
                }
            };
            result.push(dim);
        }

        result.reverse();
        result
    }
}

pub trait ConcreteShapeExt {
    fn broadcast_shape(&self, other: &[usize]) -> Option<Vec<usize>>;
    fn broadcast_shape3(&self, second: &[usize], third: &[usize]) -> Option<Vec<usize>>;
    fn pad_to_rank(&self, target_rank: usize) -> Vec<usize>;
}

impl ConcreteShapeExt for [usize] {
    fn broadcast_shape(&self, other: &[usize]) -> Option<Vec<usize>> {
        let ndims = self.len().max(other.len());
        let left = self.pad_to_rank(ndims);
        let right = other.pad_to_rank(ndims);
        let mut result = Vec::with_capacity(ndims);

        for (left_dim, right_dim) in left.into_iter().zip(right) {
            if left_dim == right_dim {
                result.push(left_dim);
                continue;
            }
            if left_dim == 1 {
                result.push(right_dim);
                continue;
            }
            if right_dim == 1 {
                result.push(left_dim);
                continue;
            }
            return None;
        }

        Some(result)
    }

    fn broadcast_shape3(&self, second: &[usize], third: &[usize]) -> Option<Vec<usize>> {
        let first = self.broadcast_shape(second)?;
        first.as_slice().broadcast_shape(third)
    }

    fn pad_to_rank(&self, target_rank: usize) -> Vec<usize> {
        let mut padded = vec![1usize; target_rank.saturating_sub(self.len())];
        padded.extend_from_slice(self);
        padded
    }
}

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

#[cfg(test)]
mod tests {
    use super::{
        shape_from_concrete, BroadcastConflictPolicy, ConcreteShapeExt, DimExpr, Shape,
        SymbolicShapeExt,
    };

    #[test]
    fn symbolic_shape_broadcast_to_requires_expand_compatible_dims() {
        let input = shape_from_concrete(&[768]);
        let target = shape_from_concrete(&[1, 8, 768]);
        assert_eq!(input.as_slice().broadcast_to(&target), Some(target));
        assert!(shape_from_concrete(&[3])
            .as_slice()
            .broadcast_to(&shape_from_concrete(&[4]))
            .is_none());
    }

    #[test]
    fn symbolic_shape_concrete_dims_and_conflict_policy_work() {
        let left = [DimExpr::Concrete(2), DimExpr::Dynamic];
        let right = [DimExpr::Concrete(1), DimExpr::Concrete(4)];
        assert!(!left.is_concrete());
        assert_eq!(right.concrete_dims(), Some(vec![1, 4]));
        assert_eq!(
            left.broadcast_shape(&right),
            Shape::from(vec![DimExpr::Concrete(2), DimExpr::Dynamic])
        );
        assert_eq!(
            left.broadcast_shape_with(&right, BroadcastConflictPolicy::Right),
            shape_from_concrete(&[2, 4])
        );
    }

    #[test]
    fn concrete_shapes_broadcast_and_pad() {
        assert_eq!([3, 1].broadcast_shape(&[1, 4]), Some(vec![3, 4]));
        assert_eq!([3].pad_to_rank(3), vec![1, 1, 3]);
        assert_eq!(
            [1, 2].broadcast_shape3(&[3, 1, 2], &[1, 3, 2]),
            Some(vec![3, 3, 2])
        );
    }
}

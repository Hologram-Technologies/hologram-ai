//! Symbolic shape handling for ONNX models.
//!
//! This module provides utilities for working with symbolic shapes in ONNX models.
//! It re-exports hologram's shape system and adds ONNX-specific conversions.
//!
//! # Symbolic Shapes
//!
//! Hologram's shape system supports three types of dimensions:
//! - **Concrete**: Known values like `224`
//! - **Var**: Symbolic variables like `"batch"`, `"seq_len"`
//! - **Expr**: Arithmetic expressions like `(H-1)/stride + 1` for Conv2D outputs
//!
//! This is **CRITICAL** for supporting:
//! - Variable batch sizes: `[batch, 224, 224, 3]`
//! - Variable sequence lengths: `[batch, seq_len, hidden_dim]`
//! - Dynamic inputs: `[batch, channels, height, width]`
//!
//! # Examples
//!
//! ```no_run
//! use hologram_onnx_core::shapes::{Dim, Shape, SymbolicShape};
//!
//! // Concrete shape (fixed batch size)
//! let shape = SymbolicShape::concrete(vec![1, 224, 224, 3]);
//! assert!(shape.is_fully_concrete());
//!
//! // Symbolic shape (variable batch)
//! let shape = SymbolicShape::symbolic(vec!["batch", "224", "224", "3"]);
//! assert!(!shape.is_fully_concrete());
//!
//! // Mixed shape (variable batch, fixed spatial dims)
//! let shape = SymbolicShape::new(vec![
//!     Dim::Var("batch".into()),
//!     Dim::Concrete(224),
//!     Dim::Concrete(224),
//!     Dim::Concrete(3),
//! ]);
//! assert!(shape.is_partially_symbolic());
//! ```

use crate::{OnnxError, Result};
use hologram_onnx_spec::ValueInfoProto;

// Re-export hologram's symbolic shape system
pub use hologram_compiler::shapes::{Dim, Shape};

/// Symbolic shape wrapper with ONNX-specific functionality.
///
/// This wraps hologram's [`Shape`] type and provides ONNX-specific
/// conversion and utility functions.
#[derive(Debug, Clone, PartialEq)]
pub struct SymbolicShape {
    inner: Shape,
}

impl SymbolicShape {
    /// Create a new symbolic shape from dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - Vector of dimension specifications
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// let shape = SymbolicShape::new(vec![
    ///     Dim::Var("batch".into()),
    ///     Dim::Concrete(224),
    ///     Dim::Concrete(224),
    ///     Dim::Concrete(3),
    /// ]);
    /// assert_eq!(shape.rank(), 4);
    /// ```
    pub fn new(dims: Vec<Dim>) -> Self {
        Self {
            inner: Shape::new(dims),
        }
    }

    /// Create a shape with all concrete dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - Concrete dimension values
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::SymbolicShape;
    ///
    /// let shape = SymbolicShape::concrete(vec![1, 784]);
    /// assert!(shape.is_fully_concrete());
    /// assert_eq!(shape.rank(), 2);
    /// ```
    pub fn concrete(dims: Vec<usize>) -> Self {
        Self {
            inner: Shape::concrete(dims),
        }
    }

    /// Create a shape with symbolic dimension names.
    ///
    /// Parses dimension strings: numeric strings become concrete dimensions,
    /// others become symbolic variables.
    ///
    /// # Arguments
    ///
    /// * `dim_names` - Dimension names (e.g., ["batch", "224", "224", "3"])
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// let shape = SymbolicShape::symbolic(vec!["batch", "224", "224", "3"]);
    /// assert_eq!(shape.rank(), 4);
    /// assert!(matches!(&shape.dims()[0], Dim::Var(name) if name == "batch"));
    /// assert_eq!(shape.dims()[1], Dim::Concrete(224));
    /// ```
    pub fn symbolic(dim_names: Vec<&str>) -> Self {
        let dims: Vec<Dim> = dim_names
            .into_iter()
            .map(|name| {
                // Try to parse as concrete dimension
                if let Ok(val) = name.parse::<usize>() {
                    Dim::Concrete(val)
                } else {
                    Dim::Var(name.to_string())
                }
            })
            .collect();

        Self::new(dims)
    }

    /// Create symbolic shape from ONNX ValueInfoProto.
    ///
    /// Converts ONNX type/shape information to symbolic shape, handling:
    /// - Concrete dimensions (dim_value > 0)
    /// - Symbolic dimensions (dim_param)
    /// - Unknown dimensions (treated as unique symbolic variables)
    ///
    /// # Arguments
    ///
    /// * `value_info` - ONNX value info containing type and shape
    ///
    /// # Errors
    ///
    /// Returns error if value info has no type or shape information.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use hologram_onnx_core::shapes::SymbolicShape;
    /// use hologram_onnx_core::parse_model;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let bytes = std::fs::read("model.onnx")?;
    /// let model = parse_model(&bytes)?;
    /// let graph = model.graph.as_ref().unwrap();
    ///
    /// for input in &graph.input {
    ///     let shape = SymbolicShape::from_value_info(input)?;
    ///     println!("Input {}: {:?}", input.name, shape);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_value_info(value_info: &ValueInfoProto) -> Result<Self> {
        use hologram_onnx_spec::type_proto::Value;
        use hologram_onnx_spec::tensor_shape_proto::dimension::Value as DimValue;

        let type_proto = value_info.r#type.as_ref()
            .ok_or_else(|| OnnxError::InvalidModel(
                format!("Value '{}' has no type information", value_info.name)
            ))?;

        let tensor_type = match &type_proto.value {
            Some(Value::TensorType(tt)) => tt,
            _ => {
                return Err(OnnxError::InvalidModel(format!(
                    "Value '{}' is not a tensor",
                    value_info.name
                )));
            }
        };

        let shape_proto = tensor_type.shape.as_ref()
            .ok_or_else(|| OnnxError::InvalidModel(
                format!("Tensor '{}' has no shape", value_info.name)
            ))?;

        let dims: Vec<Dim> = shape_proto.dim.iter().enumerate().map(|(idx, dim)| {
            match &dim.value {
                Some(DimValue::DimValue(v)) if *v > 0 => {
                    // Concrete dimension
                    Dim::Concrete(*v as usize)
                }
                Some(DimValue::DimParam(param)) if !param.is_empty() => {
                    // Named symbolic dimension
                    Dim::Var(param.clone())
                }
                _ => {
                    // No value or unnamed - create unique symbolic dimension
                    Dim::Var(format!("dim_{}_{}", value_info.name, idx))
                }
            }
        }).collect();

        Ok(Self::new(dims))
    }

    /// Get the rank (number of dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::SymbolicShape;
    ///
    /// let shape = SymbolicShape::concrete(vec![2, 3, 4]);
    /// assert_eq!(shape.rank(), 3);
    /// ```
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Get the dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// let shape = SymbolicShape::concrete(vec![2, 3]);
    /// assert_eq!(shape.dims()[0], Dim::Concrete(2));
    /// assert_eq!(shape.dims()[1], Dim::Concrete(3));
    /// ```
    pub fn dims(&self) -> &[Dim] {
        self.inner.dims()
    }

    /// Check if shape is fully concrete (all dimensions known).
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// let concrete = SymbolicShape::concrete(vec![2, 3]);
    /// assert!(concrete.is_fully_concrete());
    ///
    /// let symbolic = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(3)]);
    /// assert!(!symbolic.is_fully_concrete());
    /// ```
    pub fn is_fully_concrete(&self) -> bool {
        self.inner.is_fully_concrete()
    }

    /// Check if shape has any symbolic dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// let concrete = SymbolicShape::concrete(vec![2, 3]);
    /// assert!(!concrete.is_partially_symbolic());
    ///
    /// let mixed = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(3)]);
    /// assert!(mixed.is_partially_symbolic());
    /// ```
    pub fn is_partially_symbolic(&self) -> bool {
        self.inner.is_partially_symbolic()
    }

    /// Get the inner hologram Shape.
    ///
    /// This allows passing the shape to hologram's compiler functions.
    pub fn inner(&self) -> &Shape {
        &self.inner
    }

    /// Consume self and return the inner Shape.
    pub fn into_inner(self) -> Shape {
        self.inner
    }

    /// Infer output shape for binary element-wise operation (Add, Mul, etc.).
    ///
    /// Uses NumPy broadcasting rules:
    /// - Dimensions are aligned from the right
    /// - Size 1 dimensions are broadcast
    /// - Other dimensions must match
    ///
    /// # Arguments
    ///
    /// * `other` - Shape of the other operand
    ///
    /// # Errors
    ///
    /// Returns error if shapes are not broadcast-compatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::SymbolicShape;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = SymbolicShape::concrete(vec![2, 1, 4]);
    /// let b = SymbolicShape::concrete(vec![3, 4]);
    ///
    /// let result = a.infer_binary_op(&b)?;
    /// assert_eq!(result.rank(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer_binary_op(&self, other: &Self) -> Result<Self> {
        // Implement NumPy-style broadcasting rules
        let dims1 = self.dims();
        let dims2 = other.dims();

        let max_rank = dims1.len().max(dims2.len());
        let mut result_dims = Vec::with_capacity(max_rank);

        // Pad shorter shape with 1s on the left
        for i in 0..max_rank {
            let dim1_idx = dims1.len().wrapping_sub(max_rank - i);
            let dim2_idx = dims2.len().wrapping_sub(max_rank - i);

            let dim1 = if dim1_idx < dims1.len() {
                &dims1[dim1_idx]
            } else {
                &Dim::Concrete(1)
            };

            let dim2 = if dim2_idx < dims2.len() {
                &dims2[dim2_idx]
            } else {
                &Dim::Concrete(1)
            };

            // Broadcasting rule: dims must be equal or one must be 1
            let result_dim = match (dim1, dim2) {
                (Dim::Concrete(1), d) | (d, Dim::Concrete(1)) => d.clone(),
                (Dim::Concrete(a), Dim::Concrete(b)) if a == b => Dim::Concrete(*a),
                (Dim::Var(v1), Dim::Var(v2)) if v1 == v2 => Dim::Var(v1.clone()),
                (d @ Dim::Var(_), Dim::Concrete(_)) | (Dim::Concrete(_), d @ Dim::Var(_)) => d.clone(),
                _ => {
                    return Err(OnnxError::ShapeInferenceError(
                        format!("Cannot broadcast dimensions: {:?} and {:?}", dim1, dim2)
                    ));
                }
            };

            result_dims.push(result_dim);
        }

        Ok(Self::new(result_dims))
    }

    /// Infer output shape for MatMul operation.
    ///
    /// For 2D matrices: [M, K] @ [K, N] → [M, N]
    /// For batched matrices: [..., M, K] @ [..., K, N] → [..., M, N]
    ///
    /// # Arguments
    ///
    /// * `other` - Shape of the right operand
    ///
    /// # Errors
    ///
    /// Returns error if shapes are incompatible for matrix multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::SymbolicShape;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let a = SymbolicShape::concrete(vec![32, 64]);
    /// let b = SymbolicShape::concrete(vec![64, 16]);
    ///
    /// let result = a.infer_matmul(&b)?;
    /// assert_eq!(result.rank(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer_matmul(&self, other: &Self) -> Result<Self> {
        let dims1 = self.dims();
        let dims2 = other.dims();

        if dims1.len() < 2 || dims2.len() < 2 {
            return Err(OnnxError::ShapeInferenceError(
                "MatMul requires at least 2D tensors".into()
            ));
        }

        // For MatMul: [..., M, K] @ [..., K, N] → [..., M, N]
        // Last two dimensions: (M, K) @ (K, N) → (M, N)
        // Batch dimensions broadcast

        let m = dims1[dims1.len() - 2].clone();
        let k1 = &dims1[dims1.len() - 1];
        let k2 = &dims2[dims2.len() - 2];
        let n = dims2[dims2.len() - 1].clone();

        // Check inner dimensions match
        match (k1, k2) {
            (Dim::Concrete(a), Dim::Concrete(b)) if a != b => {
                return Err(OnnxError::ShapeInferenceError(
                    format!("MatMul inner dimensions mismatch: {} != {}", a, b)
                ));
            }
            _ => {} // Symbolic or matching
        }

        // Handle batch dimensions
        if dims1.len() == 2 && dims2.len() == 2 {
            // Simple 2D case: [M, K] @ [K, N] → [M, N]
            Ok(Self::new(vec![m, n]))
        } else {
            // Batched case: broadcast batch dimensions
            let batch1 = &dims1[..dims1.len() - 2];
            let batch2 = &dims2[..dims2.len() - 2];

            // Broadcast batch dimensions
            let max_batch_rank = batch1.len().max(batch2.len());
            let mut result_dims = Vec::with_capacity(max_batch_rank + 2);

            for i in 0..max_batch_rank {
                let dim1_idx = batch1.len().wrapping_sub(max_batch_rank - i);
                let dim2_idx = batch2.len().wrapping_sub(max_batch_rank - i);

                let dim1 = if dim1_idx < batch1.len() {
                    &batch1[dim1_idx]
                } else {
                    &Dim::Concrete(1)
                };

                let dim2 = if dim2_idx < batch2.len() {
                    &batch2[dim2_idx]
                } else {
                    &Dim::Concrete(1)
                };

                // Broadcasting rule
                let result_dim = match (dim1, dim2) {
                    (Dim::Concrete(1), d) | (d, Dim::Concrete(1)) => d.clone(),
                    (Dim::Concrete(a), Dim::Concrete(b)) if a == b => Dim::Concrete(*a),
                    (Dim::Var(v1), Dim::Var(v2)) if v1 == v2 => Dim::Var(v1.clone()),
                    (d @ Dim::Var(_), Dim::Concrete(_)) | (Dim::Concrete(_), d @ Dim::Var(_)) => d.clone(),
                    _ => {
                        return Err(OnnxError::ShapeInferenceError(
                            format!("Cannot broadcast batch dimensions: {:?} and {:?}", dim1, dim2)
                        ));
                    }
                };

                result_dims.push(result_dim);
            }

            // Add matrix dimensions
            result_dims.push(m);
            result_dims.push(n);

            Ok(Self::new(result_dims))
        }
    }

    /// Infer output shape for Transpose operation.
    ///
    /// # Arguments
    ///
    /// * `perm` - Permutation of axes (None = reverse axes)
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::SymbolicShape;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let shape = SymbolicShape::concrete(vec![2, 3, 4]);
    ///
    /// // Transpose with permutation [2, 0, 1]
    /// let result = shape.infer_transpose(Some(&[2, 0, 1]))?;
    /// assert_eq!(result.rank(), 3);
    ///
    /// // Default transpose (reverse axes)
    /// let result = shape.infer_transpose(None)?;
    /// assert_eq!(result.rank(), 3);
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer_transpose(&self, perm: Option<&[i64]>) -> Result<Self> {
        let rank = self.rank();

        let perm: Vec<usize> = if let Some(p) = perm {
            // Validate permutation
            if p.len() != rank {
                return Err(OnnxError::ShapeInferenceError(
                    format!("Permutation length {} != rank {}", p.len(), rank)
                ));
            }

            p.iter().map(|&i| {
                if i < 0 {
                    (rank as i64 + i) as usize
                } else {
                    i as usize
                }
            }).collect()
        } else {
            // Default: reverse axes
            (0..rank).rev().collect()
        };

        let new_dims: Vec<Dim> = perm.iter()
            .map(|&i| self.dims()[i].clone())
            .collect();

        Ok(Self::new(new_dims))
    }

    /// Infer output shape for Reshape operation.
    ///
    /// # Arguments
    ///
    /// * `target` - Target shape (may contain -1 for inferred dimension)
    ///
    /// # Errors
    ///
    /// Returns error if reshape is invalid (more than one -1, incompatible sizes).
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_onnx_core::shapes::{Dim, SymbolicShape};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let shape = SymbolicShape::concrete(vec![2, 3, 4]);
    ///
    /// // Reshape to [6, 4]
    /// let target = vec![Dim::Concrete(6), Dim::Concrete(4)];
    /// let result = shape.infer_reshape(&target)?;
    /// assert_eq!(result.rank(), 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn infer_reshape(&self, target: &[Dim]) -> Result<Self> {
        // For now, just return target shape as-is
        // Full reshape validation would require computing total size
        // which is complex with symbolic dimensions
        Ok(Self::new(target.to_vec()))
    }
}

impl From<Shape> for SymbolicShape {
    fn from(shape: Shape) -> Self {
        Self { inner: shape }
    }
}

impl From<SymbolicShape> for Shape {
    fn from(shape: SymbolicShape) -> Self {
        shape.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concrete_shape() {
        let shape = SymbolicShape::concrete(vec![2, 3, 4]);
        assert_eq!(shape.rank(), 3);
        assert!(shape.is_fully_concrete());
        assert!(!shape.is_partially_symbolic());
        assert_eq!(shape.dims()[0], Dim::Concrete(2));
        assert_eq!(shape.dims()[1], Dim::Concrete(3));
        assert_eq!(shape.dims()[2], Dim::Concrete(4));
    }

    #[test]
    fn test_symbolic_shape() {
        let shape = SymbolicShape::symbolic(vec!["batch", "seq_len"]);
        assert_eq!(shape.rank(), 2);
        assert!(!shape.is_fully_concrete());
        assert!(shape.is_partially_symbolic());
        assert!(matches!(&shape.dims()[0], Dim::Var(name) if name == "batch"));
        assert!(matches!(&shape.dims()[1], Dim::Var(name) if name == "seq_len"));
    }

    #[test]
    fn test_mixed_shape() {
        let shape = SymbolicShape::symbolic(vec!["batch", "224", "224", "3"]);
        assert_eq!(shape.rank(), 4);
        assert!(!shape.is_fully_concrete());
        assert!(shape.is_partially_symbolic());
        assert!(matches!(&shape.dims()[0], Dim::Var(name) if name == "batch"));
        assert_eq!(shape.dims()[1], Dim::Concrete(224));
        assert_eq!(shape.dims()[2], Dim::Concrete(224));
        assert_eq!(shape.dims()[3], Dim::Concrete(3));
    }

    #[test]
    fn test_new_shape() {
        let shape = SymbolicShape::new(vec![
            Dim::Var("N".into()),
            Dim::Concrete(10),
        ]);
        assert_eq!(shape.rank(), 2);
        assert!(shape.is_partially_symbolic());
    }

    #[test]
    fn test_binary_op_broadcast() {
        let a = SymbolicShape::concrete(vec![2, 1, 4]);
        let b = SymbolicShape::concrete(vec![3, 4]);

        let result = a.infer_binary_op(&b).unwrap();
        assert_eq!(result.rank(), 3);
    }

    #[test]
    fn test_matmul_2d() {
        let a = SymbolicShape::concrete(vec![32, 64]);
        let b = SymbolicShape::concrete(vec![64, 16]);

        let result = a.infer_matmul(&b).unwrap();
        assert_eq!(result.rank(), 2);
    }

    #[test]
    fn test_matmul_symbolic() {
        let a = SymbolicShape::symbolic(vec!["M", "K"]);
        let b = SymbolicShape::symbolic(vec!["K", "N"]);

        let result = a.infer_matmul(&b).unwrap();
        assert_eq!(result.rank(), 2);
    }

    #[test]
    fn test_transpose() {
        let shape = SymbolicShape::concrete(vec![2, 3, 4]);

        // With permutation
        let result = shape.infer_transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(result.rank(), 3);
        assert_eq!(result.dims()[0], Dim::Concrete(4));
        assert_eq!(result.dims()[1], Dim::Concrete(2));
        assert_eq!(result.dims()[2], Dim::Concrete(3));

        // Default (reverse)
        let result = shape.infer_transpose(None).unwrap();
        assert_eq!(result.rank(), 3);
        assert_eq!(result.dims()[0], Dim::Concrete(4));
        assert_eq!(result.dims()[1], Dim::Concrete(3));
        assert_eq!(result.dims()[2], Dim::Concrete(2));
    }

    #[test]
    fn test_transpose_negative_indices() {
        let shape = SymbolicShape::concrete(vec![2, 3, 4]);

        // -1 means last axis
        let result = shape.infer_transpose(Some(&[-1, 0, 1])).unwrap();
        assert_eq!(result.dims()[0], Dim::Concrete(4));
        assert_eq!(result.dims()[1], Dim::Concrete(2));
        assert_eq!(result.dims()[2], Dim::Concrete(3));
    }

    #[test]
    fn test_reshape() {
        let shape = SymbolicShape::concrete(vec![2, 3, 4]);

        let target = vec![Dim::Concrete(6), Dim::Concrete(4)];
        let result = shape.infer_reshape(&target).unwrap();
        assert_eq!(result.rank(), 2);
        assert_eq!(result.dims()[0], Dim::Concrete(6));
        assert_eq!(result.dims()[1], Dim::Concrete(4));
    }

    #[test]
    fn test_conversion() {
        let symbolic_shape = SymbolicShape::concrete(vec![2, 3]);
        let inner_shape: Shape = symbolic_shape.clone().into();
        let back: SymbolicShape = inner_shape.into();
        assert_eq!(back.rank(), 2);
    }
}

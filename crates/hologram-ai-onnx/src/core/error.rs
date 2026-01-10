//! Error types for ONNX compilation.
//!
//! This module defines all error types that can occur during ONNX parsing,
//! translation, and compilation to hologram's .holo format.

use thiserror::Error;

/// Result type for ONNX operations.
pub type Result<T> = std::result::Result<T, OnnxError>;

/// Errors that can occur during ONNX compilation.
#[derive(Error, Debug)]
pub enum OnnxError {
    /// ONNX protobuf parsing failed.
    #[error("Failed to parse ONNX protobuf: {0}")]
    ParseError(String),

    /// ONNX model is invalid or malformed.
    #[error("Invalid ONNX model: {0}")]
    InvalidModel(String),

    /// Unsupported ONNX operation encountered.
    #[error("Unsupported ONNX operation: {op_type} (opset {opset_version})")]
    UnsupportedOp {
        /// Operation type name
        op_type: String,
        /// ONNX opset version
        opset_version: i64,
    },

    /// Unsupported or invalid attribute value.
    #[error("Invalid attribute '{name}': {reason}")]
    InvalidAttribute {
        /// Attribute name
        name: String,
        /// Reason for invalidity
        reason: String,
    },

    /// Shape inference failed.
    #[error("Shape inference failed: {0}")]
    ShapeInferenceError(String),

    /// Symbolic shape validation failed.
    #[error("Symbolic shape error: {0}")]
    SymbolicShapeError(String),

    /// Shape mismatch between tensors.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape
        expected: Vec<String>,
        /// Actual shape
        actual: Vec<String>,
    },

    /// Weight data extraction or processing failed.
    #[error("Weight processing error: {0}")]
    WeightError(String),

    /// Unsupported data type.
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    /// Missing required input or initializer.
    #[error("Missing input: {0}")]
    MissingInput(String),

    /// Missing required output.
    #[error("Missing output: {0}")]
    MissingOutput(String),

    /// Graph partitioning failed.
    #[error("Graph partitioning error: {0}")]
    PartitioningError(String),

    /// Memory budget exceeded during compilation.
    #[error("Memory budget exceeded: {used} MB used, {budget} MB allowed")]
    MemoryBudgetExceeded {
        /// Memory used (MB)
        used: usize,
        /// Memory budget (MB)
        budget: usize,
    },

    /// IR translation failed.
    #[error("IR translation error: {0}")]
    IrTranslationError(String),

    /// Decomposition pass failed.
    #[error("Decomposition error: {0}")]
    DecompositionError(String),

    /// OperationGraph lowering failed.
    #[error("Lowering error: {0}")]
    LoweringError(String),

    /// Serialization failed.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// I/O error occurred.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Protobuf decode error.
    #[error("Protobuf decode error: {0}")]
    ProtobufError(#[from] prost::DecodeError),

    /// Hologram compiler error.
    #[error("Hologram compiler error: {0}")]
    HologramError(String),

    /// Hologram IR error.
    #[error("Hologram IR error: {0}")]
    IrError(#[from] hologram::IrError),

    /// Internal error (should not happen).
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl OnnxError {
    /// Create an unsupported operation error.
    ///
    /// # Arguments
    ///
    /// * `op_type` - ONNX operation type name
    /// * `opset_version` - ONNX opset version
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::unsupported_op("CustomOp", 13);
    /// assert!(err.to_string().contains("CustomOp"));
    /// ```
    pub fn unsupported_op(op_type: impl Into<String>, opset_version: i64) -> Self {
        Self::UnsupportedOp {
            op_type: op_type.into(),
            opset_version,
        }
    }

    /// Create an invalid attribute error.
    ///
    /// # Arguments
    ///
    /// * `name` - Attribute name
    /// * `reason` - Reason for invalidity
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::invalid_attribute("axis", "must be non-negative");
    /// assert!(err.to_string().contains("axis"));
    /// ```
    pub fn invalid_attribute(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidAttribute {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Create a shape mismatch error.
    ///
    /// # Arguments
    ///
    /// * `expected` - Expected shape dimensions
    /// * `actual` - Actual shape dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::shape_mismatch(
    ///     vec!["N", "784"],
    ///     vec!["N", "128"]
    /// );
    /// assert!(err.to_string().contains("expected"));
    /// ```
    pub fn shape_mismatch<S: Into<String>>(expected: Vec<S>, actual: Vec<S>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into_iter().map(|s| s.into()).collect(),
            actual: actual.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Create a memory budget exceeded error.
    ///
    /// # Arguments
    ///
    /// * `used` - Memory used in MB
    /// * `budget` - Memory budget in MB
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::memory_budget_exceeded(12000, 8000);
    /// assert!(err.to_string().contains("12000 MB"));
    /// ```
    pub fn memory_budget_exceeded(used: usize, budget: usize) -> Self {
        Self::MemoryBudgetExceeded { used, budget }
    }

    /// Check if this error is related to unsupported operations.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::unsupported_op("CustomOp", 13);
    /// assert!(err.is_unsupported_op());
    ///
    /// let err = OnnxError::ParseError("invalid".into());
    /// assert!(!err.is_unsupported_op());
    /// ```
    pub fn is_unsupported_op(&self) -> bool {
        matches!(self, OnnxError::UnsupportedOp { .. })
    }

    /// Check if this error is related to shape inference.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::ShapeInferenceError("failed".into());
    /// assert!(err.is_shape_error());
    ///
    /// let err = OnnxError::SymbolicShapeError("invalid".into());
    /// assert!(err.is_shape_error());
    /// ```
    pub fn is_shape_error(&self) -> bool {
        matches!(
            self,
            OnnxError::ShapeInferenceError(_)
                | OnnxError::SymbolicShapeError(_)
                | OnnxError::ShapeMismatch { .. }
        )
    }

    /// Check if this error is related to memory constraints.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxError;
    ///
    /// let err = OnnxError::memory_budget_exceeded(12000, 8000);
    /// assert!(err.is_memory_error());
    /// ```
    pub fn is_memory_error(&self) -> bool {
        matches!(self, OnnxError::MemoryBudgetExceeded { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_op_error() {
        let err = OnnxError::unsupported_op("CustomOp", 13);
        assert!(err.is_unsupported_op());
        assert!(err.to_string().contains("CustomOp"));
        assert!(err.to_string().contains("13"));
    }

    #[test]
    fn test_invalid_attribute_error() {
        let err = OnnxError::invalid_attribute("axis", "must be non-negative");
        let msg = err.to_string();
        assert!(msg.contains("axis"));
        assert!(msg.contains("must be non-negative"));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let err = OnnxError::shape_mismatch(vec!["N", "784"], vec!["N", "128"]);
        assert!(err.is_shape_error());
        let msg = err.to_string();
        assert!(msg.contains("784"));
        assert!(msg.contains("128"));
    }

    #[test]
    fn test_memory_budget_exceeded_error() {
        let err = OnnxError::memory_budget_exceeded(12000, 8000);
        assert!(err.is_memory_error());
        let msg = err.to_string();
        assert!(msg.contains("12000"));
        assert!(msg.contains("8000"));
    }

    #[test]
    fn test_error_classification() {
        let shape_err = OnnxError::ShapeInferenceError("test".into());
        assert!(shape_err.is_shape_error());
        assert!(!shape_err.is_unsupported_op());
        assert!(!shape_err.is_memory_error());

        let symbolic_err = OnnxError::SymbolicShapeError("test".into());
        assert!(symbolic_err.is_shape_error());

        let mismatch_err = OnnxError::shape_mismatch(vec!["1"], vec!["2"]);
        assert!(mismatch_err.is_shape_error());
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let onnx_err: OnnxError = io_err.into();
        assert!(matches!(onnx_err, OnnxError::IoError(_)));
    }
}

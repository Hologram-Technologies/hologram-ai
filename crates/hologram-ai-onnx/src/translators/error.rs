//! Error types for ONNX translation.
//!
//! This module defines structured error types for translation failures,
//! providing clear error messages and enabling pattern matching on errors.

use thiserror::Error;

/// Error type for ONNX to IR translation failures.
#[derive(Debug, Error)]
pub enum TranslationError {
    /// The ONNX operation is not supported.
    #[error("Unsupported ONNX operation: {op} (opset {opset})")]
    UnsupportedOp {
        /// Operation type name
        op: String,
        /// ONNX opset version
        opset: u32,
    },

    /// Wrong number of inputs provided.
    #[error("Wrong input count for {op}: expected {expected}, got {got}")]
    WrongInputCount {
        /// Operation type name
        op: String,
        /// Expected input count
        expected: usize,
        /// Actual input count
        got: usize,
    },

    /// Input count outside valid range.
    #[error("Input count out of range for {op}: expected {min}-{max}, got {got}")]
    InputCountOutOfRange {
        /// Operation type name
        op: String,
        /// Minimum valid inputs
        min: usize,
        /// Maximum valid inputs
        max: usize,
        /// Actual input count
        got: usize,
    },

    /// Not enough inputs provided.
    #[error("Not enough inputs for {op}: expected at least {min}, got {got}")]
    NotEnoughInputs {
        /// Operation type name
        op: String,
        /// Minimum required inputs
        min: usize,
        /// Actual input count
        got: usize,
    },

    /// Required attribute is missing.
    #[error("Missing required attribute '{name}' for {op}")]
    MissingAttribute {
        /// Operation type name
        op: String,
        /// Attribute name
        name: String,
    },

    /// Attribute has invalid value.
    #[error("Invalid attribute '{name}': {reason}")]
    InvalidAttribute {
        /// Attribute name
        name: String,
        /// Reason for invalidity
        reason: String,
    },

    /// Shape inference failed.
    #[error("Shape inference failed: {0}")]
    ShapeInference(String),

    /// IR builder error.
    #[error("IR builder error: {0}")]
    IrBuilder(String),
}

impl TranslationError {
    /// Create an unsupported operation error.
    pub fn unsupported_op(op: impl Into<String>, opset: u32) -> Self {
        Self::UnsupportedOp {
            op: op.into(),
            opset,
        }
    }

    /// Create a wrong input count error.
    pub fn wrong_input_count(op: impl Into<String>, expected: usize, got: usize) -> Self {
        Self::WrongInputCount {
            op: op.into(),
            expected,
            got,
        }
    }

    /// Create an input count out of range error.
    pub fn input_count_out_of_range(
        op: impl Into<String>,
        min: usize,
        max: usize,
        got: usize,
    ) -> Self {
        Self::InputCountOutOfRange {
            op: op.into(),
            min,
            max,
            got,
        }
    }

    /// Create a not enough inputs error.
    pub fn not_enough_inputs(op: impl Into<String>, min: usize, got: usize) -> Self {
        Self::NotEnoughInputs {
            op: op.into(),
            min,
            got,
        }
    }

    /// Create a missing attribute error.
    pub fn missing_attribute(op: impl Into<String>, name: impl Into<String>) -> Self {
        Self::MissingAttribute {
            op: op.into(),
            name: name.into(),
        }
    }

    /// Create an invalid attribute error.
    pub fn invalid_attribute(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidAttribute {
            name: name.into(),
            reason: reason.into(),
        }
    }

    /// Check if this is an unsupported operation error.
    pub fn is_unsupported_op(&self) -> bool {
        matches!(self, Self::UnsupportedOp { .. })
    }
}

impl From<TranslationError> for crate::core::OnnxError {
    fn from(err: TranslationError) -> Self {
        match err {
            TranslationError::UnsupportedOp { op, opset } => {
                crate::core::OnnxError::UnsupportedOp {
                    op_type: op,
                    opset_version: opset as i64,
                }
            }
            TranslationError::WrongInputCount { op, expected, got } => {
                crate::core::OnnxError::InvalidModel(format!(
                    "Wrong input count for {}: expected {}, got {}",
                    op, expected, got
                ))
            }
            TranslationError::InputCountOutOfRange { op, min, max, got } => {
                crate::core::OnnxError::InvalidModel(format!(
                    "Input count out of range for {}: expected {}-{}, got {}",
                    op, min, max, got
                ))
            }
            TranslationError::NotEnoughInputs { op, min, got } => {
                crate::core::OnnxError::InvalidModel(format!(
                    "Not enough inputs for {}: expected at least {}, got {}",
                    op, min, got
                ))
            }
            TranslationError::MissingAttribute { op, name } => {
                crate::core::OnnxError::InvalidAttribute {
                    name,
                    reason: format!("Missing required attribute for {}", op),
                }
            }
            TranslationError::InvalidAttribute { name, reason } => {
                crate::core::OnnxError::InvalidAttribute { name, reason }
            }
            TranslationError::ShapeInference(msg) => {
                crate::core::OnnxError::ShapeInferenceError(msg)
            }
            TranslationError::IrBuilder(msg) => crate::core::OnnxError::IrTranslationError(msg),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unsupported_op_error() {
        let err = TranslationError::unsupported_op("CustomOp", 13);
        assert!(err.is_unsupported_op());
        assert!(err.to_string().contains("CustomOp"));
        assert!(err.to_string().contains("13"));
    }

    #[test]
    fn test_wrong_input_count_error() {
        let err = TranslationError::wrong_input_count("Add", 2, 1);
        assert!(!err.is_unsupported_op());
        assert!(err.to_string().contains("Add"));
        assert!(err.to_string().contains("expected 2"));
        assert!(err.to_string().contains("got 1"));
    }

    #[test]
    fn test_input_count_out_of_range_error() {
        let err = TranslationError::input_count_out_of_range("Conv", 2, 3, 5);
        assert!(err.to_string().contains("Conv"));
        assert!(err.to_string().contains("2-3"));
        assert!(err.to_string().contains("got 5"));
    }

    #[test]
    fn test_not_enough_inputs_error() {
        let err = TranslationError::not_enough_inputs("Concat", 1, 0);
        assert!(err.to_string().contains("Concat"));
        assert!(err.to_string().contains("at least 1"));
    }

    #[test]
    fn test_missing_attribute_error() {
        let err = TranslationError::missing_attribute("Conv", "kernel_shape");
        assert!(err.to_string().contains("Conv"));
        assert!(err.to_string().contains("kernel_shape"));
    }

    #[test]
    fn test_invalid_attribute_error() {
        let err = TranslationError::invalid_attribute("axis", "must be non-negative");
        assert!(err.to_string().contains("axis"));
        assert!(err.to_string().contains("must be non-negative"));
    }

    #[test]
    fn test_from_translation_error_to_onnx_error() {
        use crate::core::OnnxError;

        // Test UnsupportedOp conversion
        let trans_err = TranslationError::unsupported_op("CustomOp", 13);
        let onnx_err: OnnxError = trans_err.into();
        assert!(
            matches!(onnx_err, OnnxError::UnsupportedOp { op_type, opset_version }
            if op_type == "CustomOp" && opset_version == 13)
        );

        // Test WrongInputCount conversion
        let trans_err = TranslationError::wrong_input_count("Add", 2, 1);
        let onnx_err: OnnxError = trans_err.into();
        assert!(matches!(onnx_err, OnnxError::InvalidModel(msg) if msg.contains("Add")));

        // Test ShapeInference conversion
        let trans_err = TranslationError::ShapeInference("shape failed".into());
        let onnx_err: OnnxError = trans_err.into();
        assert!(matches!(onnx_err, OnnxError::ShapeInferenceError(msg) if msg == "shape failed"));

        // Test IrBuilder conversion
        let trans_err = TranslationError::IrBuilder("builder failed".into());
        let onnx_err: OnnxError = trans_err.into();
        assert!(matches!(onnx_err, OnnxError::IrTranslationError(msg) if msg == "builder failed"));
    }
}

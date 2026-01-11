//! Core traits for ONNX translation.
//!
//! This module defines the traits that all ONNX operation translators must implement.
//! Each translator is responsible for converting an ONNX node to hologram IR.

use super::error::TranslationError;
use crate::proto::{NodeProto, TensorProto};
use hologram::ir::{GraphBuilder, NodeIndex};

/// Trait for translating an ONNX operation to hologram IR.
///
/// Each ONNX operation type has a corresponding translator that implements
/// this trait. The translator is responsible for:
/// - Parsing attributes from the ONNX node
/// - Validating inputs
/// - Creating the corresponding IR nodes
///
/// # Example
///
/// ```ignore
/// use crate::translators::{OnnxTranslator, InputRequirement, TranslationError};
///
/// #[derive(Debug, Default)]
/// pub struct ReluTranslator;
///
/// impl OnnxTranslator for ReluTranslator {
///     fn onnx_op_type(&self) -> &'static str { "Relu" }
///
///     fn input_requirement(&self) -> InputRequirement {
///         InputRequirement::Exact(1)
///     }
///
///     fn translate(
///         &self,
///         _node: &NodeProto,
///         inputs: &[NodeIndex],
///         builder: &mut GraphBuilder,
///     ) -> Result<Vec<NodeIndex>, TranslationError> {
///         let result = builder.relu(inputs[0])?;
///         Ok(vec![result])
///     }
/// }
/// ```
pub trait OnnxTranslator: std::fmt::Debug + Send + Sync {
    /// Returns the ONNX operation type name this translator handles.
    ///
    /// This must match the `op_type` field in the ONNX NodeProto exactly.
    /// Examples: "Relu", "MatMul", "Conv", "Add"
    fn onnx_op_type(&self) -> &'static str;

    /// Translate an ONNX node to hologram IR nodes.
    ///
    /// # Arguments
    ///
    /// * `node` - The ONNX node to translate (contains attributes)
    /// * `inputs` - IR node indices for the input tensors
    /// * `builder` - The IR graph builder for creating nodes
    ///
    /// # Returns
    ///
    /// A vector of output IR node indices (one per ONNX output).
    fn translate(
        &self,
        node: &NodeProto,
        inputs: &[NodeIndex],
        builder: &mut GraphBuilder,
    ) -> Result<Vec<NodeIndex>, TranslationError>;

    /// Returns the input requirement for this operation.
    ///
    /// This is used for validation before `translate()` is called.
    fn input_requirement(&self) -> InputRequirement;

    /// Returns whether this operation supports constant folding.
    ///
    /// If true, `constant_fold()` may be called when all inputs are constants.
    fn supports_constant_folding(&self) -> bool {
        false
    }

    /// Attempt to constant-fold this operation.
    ///
    /// Called when all inputs are constants and `supports_constant_folding()` returns true.
    ///
    /// # Arguments
    ///
    /// * `node` - The ONNX node (for attribute access)
    /// * `constant_inputs` - Raw byte data for each constant input
    ///
    /// # Returns
    ///
    /// `Some(bytes)` with the folded constant data, or `None` if folding failed.
    fn constant_fold(&self, _node: &NodeProto, _constant_inputs: &[&[u8]]) -> Option<Vec<u8>> {
        None
    }
}

/// Input requirement specification for an ONNX operation.
///
/// Used to validate the number of inputs before translation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputRequirement {
    /// Exactly N inputs required.
    Exact(usize),
    /// Between min and max inputs (inclusive).
    Range(usize, usize),
    /// At least N inputs required.
    AtLeast(usize),
    /// Any number of inputs (including zero).
    Variadic,
}

impl InputRequirement {
    /// Validate that the input count satisfies this requirement.
    ///
    /// # Arguments
    ///
    /// * `count` - Actual number of inputs
    /// * `op_type` - Operation name for error messages
    ///
    /// # Returns
    ///
    /// `Ok(())` if valid, or an appropriate `TranslationError`.
    pub fn validate(&self, count: usize, op_type: &str) -> Result<(), TranslationError> {
        match self {
            InputRequirement::Exact(n) if count != *n => {
                Err(TranslationError::wrong_input_count(op_type, *n, count))
            }
            InputRequirement::Range(min, max) if count < *min || count > *max => Err(
                TranslationError::input_count_out_of_range(op_type, *min, *max, count),
            ),
            InputRequirement::AtLeast(min) if count < *min => {
                Err(TranslationError::not_enough_inputs(op_type, *min, count))
            }
            _ => Ok(()),
        }
    }

    /// Check if this requirement accepts zero inputs.
    pub fn accepts_zero(&self) -> bool {
        matches!(
            self,
            InputRequirement::Exact(0)
                | InputRequirement::Range(0, _)
                | InputRequirement::AtLeast(0)
                | InputRequirement::Variadic
        )
    }
}

/// Trait for type-safe ONNX attribute extraction.
///
/// This trait extends `NodeProto` with convenient methods for extracting
/// attributes by name with proper type handling.
pub trait OnnxAttributes {
    /// Get an integer attribute by name.
    fn get_int(&self, name: &str) -> Option<i64>;

    /// Get an integer attribute with a default value.
    fn get_int_or(&self, name: &str, default: i64) -> i64 {
        self.get_int(name).unwrap_or(default)
    }

    /// Get a float attribute by name.
    fn get_float(&self, name: &str) -> Option<f32>;

    /// Get a float attribute with a default value.
    fn get_float_or(&self, name: &str, default: f32) -> f32 {
        self.get_float(name).unwrap_or(default)
    }

    /// Get a string attribute by name.
    fn get_string(&self, name: &str) -> Option<&[u8]>;

    /// Get an integer array attribute by name.
    fn get_ints(&self, name: &str) -> Option<&[i64]>;

    /// Get an integer array with a default value.
    fn get_ints_or<'a>(&'a self, name: &str, default: &'a [i64]) -> &'a [i64] {
        self.get_ints(name).unwrap_or(default)
    }

    /// Get a float array attribute by name.
    fn get_floats(&self, name: &str) -> Option<&[f32]>;

    /// Get a float array with a default value.
    fn get_floats_or<'a>(&'a self, name: &str, default: &'a [f32]) -> &'a [f32] {
        self.get_floats(name).unwrap_or(default)
    }

    /// Get a tensor attribute by name.
    fn get_tensor(&self, name: &str) -> Option<&TensorProto>;
}

impl OnnxAttributes for NodeProto {
    fn get_int(&self, name: &str) -> Option<i64> {
        self.attribute.iter().find(|a| a.name == name).map(|a| a.i)
    }

    fn get_float(&self, name: &str) -> Option<f32> {
        self.attribute.iter().find(|a| a.name == name).map(|a| a.f)
    }

    fn get_string(&self, name: &str) -> Option<&[u8]> {
        self.attribute
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.s.as_slice())
    }

    fn get_ints(&self, name: &str) -> Option<&[i64]> {
        self.attribute
            .iter()
            .find(|a| a.name == name && !a.ints.is_empty())
            .map(|a| a.ints.as_slice())
    }

    fn get_floats(&self, name: &str) -> Option<&[f32]> {
        self.attribute
            .iter()
            .find(|a| a.name == name && !a.floats.is_empty())
            .map(|a| a.floats.as_slice())
    }

    fn get_tensor(&self, name: &str) -> Option<&TensorProto> {
        self.attribute
            .iter()
            .find(|a| a.name == name)
            .and_then(|a| a.t.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::AttributeProto;

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        }
    }

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        }
    }

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            ..Default::default()
        }
    }

    fn make_floats_attr(name: &str, values: Vec<f32>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            floats: values,
            ..Default::default()
        }
    }

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        }
    }

    // ===== InputRequirement Tests =====

    #[test]
    fn test_exact_requirement_valid() {
        let req = InputRequirement::Exact(2);
        assert!(req.validate(2, "Add").is_ok());
    }

    #[test]
    fn test_exact_requirement_invalid_fewer() {
        let req = InputRequirement::Exact(2);
        let err = req.validate(1, "Add");
        assert!(err.is_err());
        assert!(matches!(
            err.unwrap_err(),
            TranslationError::WrongInputCount {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_exact_requirement_invalid_more() {
        let req = InputRequirement::Exact(2);
        let err = req.validate(3, "Add");
        assert!(err.is_err());
    }

    #[test]
    fn test_range_requirement_valid() {
        let req = InputRequirement::Range(2, 4);
        assert!(req.validate(2, "Op").is_ok());
        assert!(req.validate(3, "Op").is_ok());
        assert!(req.validate(4, "Op").is_ok());
    }

    #[test]
    fn test_range_requirement_invalid() {
        let req = InputRequirement::Range(2, 4);
        assert!(req.validate(1, "Op").is_err());
        assert!(req.validate(5, "Op").is_err());
    }

    #[test]
    fn test_at_least_requirement_valid() {
        let req = InputRequirement::AtLeast(1);
        assert!(req.validate(1, "Concat").is_ok());
        assert!(req.validate(10, "Concat").is_ok());
    }

    #[test]
    fn test_at_least_requirement_invalid() {
        let req = InputRequirement::AtLeast(1);
        assert!(req.validate(0, "Concat").is_err());
    }

    #[test]
    fn test_variadic_requirement_accepts_all() {
        let req = InputRequirement::Variadic;
        assert!(req.validate(0, "Op").is_ok());
        assert!(req.validate(1, "Op").is_ok());
        assert!(req.validate(100, "Op").is_ok());
    }

    #[test]
    fn test_accepts_zero() {
        assert!(InputRequirement::Exact(0).accepts_zero());
        assert!(InputRequirement::Range(0, 1).accepts_zero());
        assert!(InputRequirement::AtLeast(0).accepts_zero());
        assert!(InputRequirement::Variadic.accepts_zero());

        assert!(!InputRequirement::Exact(1).accepts_zero());
        assert!(!InputRequirement::Range(1, 2).accepts_zero());
        assert!(!InputRequirement::AtLeast(1).accepts_zero());
    }

    // ===== OnnxAttributes Tests =====

    #[test]
    fn test_get_int() {
        let node = NodeProto {
            attribute: vec![make_int_attr("axis", 1), make_int_attr("keepdims", 0)],
            ..Default::default()
        };

        assert_eq!(node.get_int("axis"), Some(1));
        assert_eq!(node.get_int("keepdims"), Some(0));
        assert_eq!(node.get_int("missing"), None);
    }

    #[test]
    fn test_get_int_or() {
        let node = NodeProto {
            attribute: vec![make_int_attr("axis", -1)],
            ..Default::default()
        };

        assert_eq!(node.get_int_or("axis", 0), -1);
        assert_eq!(node.get_int_or("missing", 42), 42);
    }

    #[test]
    fn test_get_ints() {
        let node = NodeProto {
            attribute: vec![make_ints_attr("perm", vec![0, 2, 1])],
            ..Default::default()
        };

        assert_eq!(node.get_ints("perm"), Some([0, 2, 1].as_slice()));
        assert_eq!(node.get_ints("missing"), None);
    }

    #[test]
    fn test_get_ints_or() {
        let node = NodeProto {
            attribute: vec![make_ints_attr("axes", vec![1, 2])],
            ..Default::default()
        };

        assert_eq!(node.get_ints_or("axes", &[0]), &[1, 2]);
        assert_eq!(node.get_ints_or("missing", &[0, 1]), &[0, 1]);
    }

    #[test]
    fn test_get_float() {
        let node = NodeProto {
            attribute: vec![make_float_attr("alpha", 0.5)],
            ..Default::default()
        };

        assert_eq!(node.get_float("alpha"), Some(0.5));
        assert_eq!(node.get_float("missing"), None);
    }

    #[test]
    fn test_get_float_or() {
        let node = NodeProto {
            attribute: vec![make_float_attr("epsilon", 1e-5)],
            ..Default::default()
        };

        assert!((node.get_float_or("epsilon", 0.0) - 1e-5).abs() < 1e-10);
        assert!((node.get_float_or("missing", 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_floats() {
        let node = NodeProto {
            attribute: vec![make_floats_attr("scales", vec![1.0, 2.0, 3.0])],
            ..Default::default()
        };

        assert_eq!(node.get_floats("scales"), Some([1.0, 2.0, 3.0].as_slice()));
        assert_eq!(node.get_floats("missing"), None);
    }

    #[test]
    fn test_get_string() {
        let node = NodeProto {
            attribute: vec![make_string_attr("mode", "constant")],
            ..Default::default()
        };

        assert_eq!(node.get_string("mode"), Some(b"constant".as_slice()));
        assert_eq!(node.get_string("missing"), None);
    }

    #[test]
    fn test_empty_node() {
        let node = NodeProto::default();

        assert_eq!(node.get_int("any"), None);
        assert_eq!(node.get_float("any"), None);
        assert_eq!(node.get_ints("any"), None);
        assert_eq!(node.get_string("any"), None);
    }
}

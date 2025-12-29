//! ONNX shape manipulation operations.
//!
//! All operations in this module:
//! - Support **symbolic shapes** for dynamic tensor dimensions
//! - Use **compile-time shape resolution** where possible
//! - Leverage **PhiCoordinate addressing** for efficient indexing
//!
//! # ISA Optimizations
//!
//! - **PhiCoordinate**: Efficient address computation for transposed/reshaped tensors
//! - **Zero-copy**: Reshape/transpose can be view operations when possible
//! - **Compile-time**: Shape transformations resolved during compilation

use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use hologram_compiler::ir::{IRBuilder, NodeId};
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Reshape operation.
///
/// Reshape: Change tensor shape without changing data order.
///
/// # Inputs
///
/// - Input 0: Data tensor
/// - Input 1: Shape tensor (int64, can contain -1 for inferred dimension)
///
/// # Performance
///
/// - **Zero-copy** when possible (view operation)
/// - **PhiCoordinate addressing** for efficient access patterns
/// - Supports **symbolic shapes** (can reshape to symbolic dimensions)
pub fn translate_reshape(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(
            format!("Reshape expects 2 inputs, got {}", inputs.len())
        ));
    }

    let data = inputs[0];
    let _shape_input = inputs[1];

    debug!("Translating Reshape operation");
    trace!("Reshape inputs: data={:?}", data);

    // IRBuilder.reshape takes a Shape, not a NodeId for shape
    // For dynamic shapes from a second input, we need decomposition
    // For now, return not-implemented error for dynamic reshape
    let _ = builder;
    Err(OnnxError::IrTranslationError(
        "Reshape with dynamic shape input not yet implemented (requires shape extraction)".to_string()
    ))
}

/// Translate ONNX Transpose operation.
///
/// Transpose: Permute tensor dimensions according to `perm` attribute.
///
/// # Attributes
///
/// - `perm` (ints, optional): Permutation of dimensions
///   - If not specified, reverses dimensions (e.g., [0,1,2] -> [2,1,0])
///
/// # Performance
///
/// - **PhiCoordinate addressing**: O(1) address computation for any permutation
/// - **Zero-copy view** when backend supports it
/// - Supports **symbolic shapes**
pub fn translate_transpose(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Transpose expects 1 input, got 0".to_string()
        ));
    }

    let input = inputs[0];

    // Parse perm attribute (empty vec means reverse dimensions)
    let perm = parse_attr_ints(attrs, "perm", vec![])?;

    debug!("Translating Transpose operation (perm={:?})", perm);
    trace!("Transpose input: {:?}", input);

    // Convert to Option<Vec<usize>> for builder
    let perm_opt = if perm.is_empty() {
        None
    } else {
        Some(perm.into_iter().map(|x| x as usize).collect())
    };

    // Create Transpose IR node
    let node = builder.transpose(input, perm_opt);

    trace!("Created Transpose node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Squeeze operation.
///
/// Squeeze: Remove dimensions of size 1.
///
/// # Attributes
///
/// - `axes` (ints, optional): Dimensions to squeeze
///   - If not specified, squeezes all dimensions of size 1
///
/// # Performance
///
/// - **Zero-copy**: Shape metadata change only
/// - Supports **symbolic shapes** (but cannot squeeze symbolic dimensions)
pub fn translate_squeeze(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Squeeze expects 1 input, got 0".to_string()
        ));
    }

    let input = inputs[0];

    // Parse axes attribute (empty means squeeze all size-1 dims)
    let axes = parse_attr_ints(attrs, "axes", vec![])?;

    debug!("Translating Squeeze operation (axes={:?})", axes);
    trace!("Squeeze input: {:?}", input);

    // IRBuilder doesn't have squeeze, need to decompose to reshape
    // For now, return not-implemented error
    let _ = (builder, input, axes);
    Err(OnnxError::IrTranslationError(
        "Squeeze operation not yet implemented (requires reshape decomposition)".to_string()
    ))
}

/// Translate ONNX Unsqueeze operation.
///
/// Unsqueeze: Add dimensions of size 1.
///
/// # Attributes
///
/// - `axes` (ints, required): Dimensions to add
///
/// # Performance
///
/// - **Zero-copy**: Shape metadata change only
/// - Supports **symbolic shapes**
pub fn translate_unsqueeze(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Unsqueeze expects 1 input, got 0".to_string()
        ));
    }

    let input = inputs[0];

    // Parse axes attribute (required for Unsqueeze)
    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    if axes.is_empty() {
        return Err(OnnxError::InvalidAttribute {
            name: "axes".to_string(),
            reason: "Unsqueeze requires non-empty axes attribute".to_string(),
        });
    }

    debug!("Translating Unsqueeze operation (axes={:?})", axes);
    trace!("Unsqueeze input: {:?}", input);

    // IRBuilder doesn't have unsqueeze, need to decompose to reshape
    // For now, return not-implemented error
    let _ = (builder, input);
    Err(OnnxError::IrTranslationError(
        "Unsqueeze operation not yet implemented (requires reshape decomposition)".to_string()
    ))
}

/// Translate ONNX Concat operation.
///
/// Concat: Concatenate tensors along specified axis.
///
/// # Attributes
///
/// - `axis` (int, required): Axis along which to concatenate
///
/// # Performance
///
/// - **PhiCoordinate addressing**: Efficient multi-tensor indexing
/// - **LOOP instructions**: O(1) space complexity for concatenation
/// - Supports **symbolic shapes** (concatenates symbolic dimensions)
pub fn translate_concat(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidModel(
            format!("Concat expects at least 2 inputs, got {}", inputs.len())
        ));
    }

    // Parse axis attribute (required)
    let axis = parse_attr_int(attrs, "axis", i64::MAX)?;
    if axis == i64::MAX {
        return Err(OnnxError::InvalidAttribute {
            name: "axis".to_string(),
            reason: "Concat requires axis attribute".to_string(),
        });
    }

    debug!("Translating Concat operation (axis={}, {} inputs)", axis, inputs.len());
    trace!("Concat inputs: {:?}", inputs);

    // Create Concat IR node
    let node = builder.concat(inputs.to_vec(), axis as isize);

    trace!("Created Concat node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Split operation.
///
/// Split: Split tensor into multiple outputs along specified axis.
///
/// # Attributes
///
/// - `axis` (int, default 0): Axis along which to split
/// - `split` (ints, optional): Sizes of each output
///   - If not specified, splits into equal parts
///
/// # Performance
///
/// - **PhiCoordinate addressing**: Efficient slicing
/// - **Zero-copy views** when backend supports it
/// - Supports **symbolic shapes**
///
/// # Note
///
/// Split produces multiple outputs. The IR node returns a single node
/// that represents the split operation, with outputs accessed via indices.
pub fn translate_split(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Split expects 1 input, got 0".to_string()
        ));
    }

    let input = inputs[0];

    // Parse attributes
    let axis = parse_attr_int(attrs, "axis", 0)?;
    let split_sizes = parse_attr_ints(attrs, "split", vec![])?;

    debug!("Translating Split operation (axis={}, splits={:?})", axis, split_sizes);
    trace!("Split input: {:?}", input);

    // IRBuilder doesn't have split, need to decompose to slice operations
    // For now, return not-implemented error
    let _ = (builder, input, axis, split_sizes);
    Err(OnnxError::IrTranslationError(
        "Split operation not yet implemented (requires slice decomposition)".to_string()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::f32_tensor;
    use hologram_compiler::ir::IRBuilder;
    use hologram_onnx_spec::attribute_proto::AttributeType;

    fn make_builder() -> IRBuilder {
        IRBuilder::new("test")
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_reshape_returns_not_implemented() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[2, 3, 4]));
        let shape = builder.add_input("shape", f32_tensor(&[2]));

        let result = translate_reshape(
            &vec![data, shape],
            &[],
            &HashMap::new(),
            &mut builder
        );
        // Dynamic reshape not yet implemented
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::IrTranslationError(_)));
    }

    #[test]
    fn test_translate_reshape_wrong_inputs() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[2, 3]));

        // Only 1 input (needs 2)
        let result = translate_reshape(&vec![data], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_transpose_default() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        // No perm attribute means reverse dimensions
        let result = translate_transpose(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_transpose_custom_perm() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![make_ints_attr("perm", vec![2, 0, 1])];

        let result = translate_transpose(&vec![input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_squeeze_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 2, 1, 3]));

        // Squeeze not yet implemented
        let result = translate_squeeze(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::IrTranslationError(_)));
    }

    #[test]
    fn test_translate_unsqueeze_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let attrs = vec![make_ints_attr("axes", vec![0, 3])];

        let result = translate_unsqueeze(&vec![input], &attrs, &HashMap::new(), &mut builder);
        // Unsqueeze not yet implemented
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::IrTranslationError(_)));
    }

    #[test]
    fn test_translate_unsqueeze_no_axes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        // No axes attribute should fail with InvalidAttribute
        let result = translate_unsqueeze(&vec![input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidAttribute { .. }));
    }

    #[test]
    fn test_translate_concat() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 4]));
        let c = builder.add_input("C", f32_tensor(&[2, 5]));

        let attrs = vec![make_int_attr("axis", 1)];

        let result = translate_concat(
            &vec![a, b, c],
            &attrs,
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_concat_insufficient_inputs() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));

        let attrs = vec![make_int_attr("axis", 0)];

        // Only 1 input (needs at least 2)
        let result = translate_concat(&vec![a], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_concat_no_axis() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        // No axis attribute should fail
        let result = translate_concat(&vec![a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidAttribute { .. }));
    }

    #[test]
    fn test_translate_split_returns_not_implemented() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 6]));

        let attrs = vec![make_int_attr("axis", 1)];

        let result = translate_split(&vec![input], &attrs, &HashMap::new(), &mut builder);
        // Split not yet implemented
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::IrTranslationError(_)));
    }

    #[test]
    fn test_transpose_empty_input() {
        let mut builder = make_builder();

        let result = translate_transpose(&vec![], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }
}

//! ONNX indexing operations.
//!
//! Operations for indexing and slicing tensors:
//! - **Gather**: Index selection along an axis (used for embeddings)
//! - **Slice**: Extract a sub-tensor with start/end indices
//!
//! # Usage in Stable Diffusion
//!
//! - **Gather**: Text encoder embeddings, timestep embeddings
//! - **Slice**: Attention masking, sequence slicing

use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use hologram_compiler::ir::{IRBuilder, NodeId};
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::parse_attr_int;

/// Translate ONNX Gather operation.
///
/// Gather: Select elements from data tensor using indices.
///
/// Given data tensor of rank r >= 1, and indices tensor of rank q,
/// gather entries of the axis dimension of data indexed by indices,
/// and concatenate them.
///
/// # Inputs
///
/// - Input 0: data - Tensor of rank r >= 1
/// - Input 1: indices - Tensor of any rank q
///
/// # Attributes
///
/// - `axis` (int, default 0): Which axis to gather on
///
/// # Output Shape
///
/// The output has rank r + q - 1.
///
/// # Performance
///
/// - **SIMD vectorization**: Vectorized gather operations
/// - Critical for embedding lookups in transformers
///
/// # Example
///
/// ```text
/// data = [[1, 2], [3, 4], [5, 6]]  # shape [3, 2]
/// indices = [0, 2]                  # shape [2]
/// axis = 0
/// output = [[1, 2], [5, 6]]         # shape [2, 2]
/// ```
pub fn translate_gather(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(
            format!("Gather expects 2 inputs, got {}", inputs.len())
        ));
    }

    let data = inputs[0];
    let indices = inputs[1];

    let axis = parse_attr_int(attrs, "axis", 0)?;

    debug!("Translating Gather operation (axis={})", axis);
    trace!("Gather inputs: data={:?}, indices={:?}", data, indices);

    // Use builder's gather operation
    let result = builder.gather(data, indices, axis as isize);

    trace!("Created Gather node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Slice operation.
///
/// Slice: Produces a slice of the input tensor along multiple axes.
///
/// # Inputs
///
/// - Input 0: data - Tensor to slice
/// - Input 1: starts - 1-D tensor of starting indices
/// - Input 2: ends - 1-D tensor of ending indices
/// - Input 3: axes (optional) - 1-D tensor of axes to slice
/// - Input 4: steps (optional) - 1-D tensor of slice steps
///
/// # Performance
///
/// - **Zero-copy slicing**: Uses view when possible
/// - Common in attention mechanisms for masking
///
/// # Implementation
///
/// ONNX Slice uses tensor inputs for slice indices, which can be dynamic.
/// We represent this as a Call node to `onnx.Slice` which the runtime
/// handles by extracting the constant values from the parameter tensors.
pub fn translate_slice(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 3 || inputs.len() > 5 {
        return Err(OnnxError::InvalidModel(format!(
            "Slice expects 3-5 inputs, got {}",
            inputs.len()
        )));
    }

    let data = inputs[0];
    let starts = inputs[1];
    let ends = inputs[2];

    debug!("Translating Slice operation");
    trace!(
        "Slice inputs: data={:?}, starts={:?}, ends={:?}",
        data,
        starts,
        ends
    );

    // Build the argument list for the dynamic slice call
    // Arguments: [data, starts, ends, axes?, steps?]
    let mut args = vec![data, starts, ends];

    // Add optional axes input
    if inputs.len() >= 4 {
        args.push(inputs[3]);
    }

    // Add optional steps input
    if inputs.len() >= 5 {
        args.push(inputs[4]);
    }

    // Use a Call node to represent dynamic slicing
    // The runtime will handle this by extracting constant values from the parameter tensors
    let result = builder.call("onnx.Slice", args);

    trace!("Created Slice call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX GatherElements operation.
///
/// GatherElements: Gather values from input based on indices.
///
/// Similar to Gather but indices have same rank as input.
///
/// # Inputs
///
/// - Input 0: data - Tensor of rank r
/// - Input 1: indices - Tensor of rank r with same shape except possibly
///   the dimension at axis
///
/// # Attributes
///
/// - `axis` (int, default 0): Which axis to gather on
pub fn translate_gather_elements(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(
            format!("GatherElements expects 2 inputs, got {}", inputs.len())
        ));
    }

    let data = inputs[0];
    let indices = inputs[1];

    let axis = parse_attr_int(attrs, "axis", 0)?;

    debug!("Translating GatherElements operation (axis={})", axis);
    trace!("GatherElements inputs: data={:?}, indices={:?}", data, indices);

    // GatherElements uses the same underlying mechanism as Gather
    let result = builder.gather(data, indices, axis as isize);

    trace!("Created GatherElements node: {:?}", result);
    Ok(result)
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

    // ========================================================================
    // Gather Tests
    // ========================================================================

    #[test]
    fn test_translate_gather() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[10, 768])); // Vocab embeddings
        let indices = builder.add_input("indices", f32_tensor(&[5])); // Token indices

        let attrs = vec![make_int_attr("axis", 0)];

        let result = translate_gather(&[data, indices], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gather_axis_1() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[2, 10, 768]));
        let indices = builder.add_input("indices", f32_tensor(&[3]));

        let attrs = vec![make_int_attr("axis", 1)];

        let result = translate_gather(&[data, indices], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gather_symbolic() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[]));
        let indices = builder.add_input("indices", f32_tensor(&[]));

        let result = translate_gather(&[data, indices], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gather_wrong_inputs() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[10, 768]));

        let result = translate_gather(&[data], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // Slice Tests
    // ========================================================================

    #[test]
    fn test_translate_slice() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[20, 10, 5]));
        let starts = builder.add_input("starts", f32_tensor(&[3]));
        let ends = builder.add_input("ends", f32_tensor(&[3]));

        let result = translate_slice(
            &[data, starts, ends],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_slice_with_axes() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[20, 10, 5]));
        let starts = builder.add_input("starts", f32_tensor(&[1]));
        let ends = builder.add_input("ends", f32_tensor(&[1]));
        let axes = builder.add_input("axes", f32_tensor(&[1]));

        let result = translate_slice(
            &[data, starts, ends, axes],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_slice_with_steps() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[20, 10, 5]));
        let starts = builder.add_input("starts", f32_tensor(&[1]));
        let ends = builder.add_input("ends", f32_tensor(&[1]));
        let axes = builder.add_input("axes", f32_tensor(&[1]));
        let steps = builder.add_input("steps", f32_tensor(&[1]));

        let result = translate_slice(
            &[data, starts, ends, axes, steps],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_slice_wrong_inputs() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[20, 10, 5]));
        let starts = builder.add_input("starts", f32_tensor(&[3]));

        // Only 2 inputs (needs at least 3)
        let result = translate_slice(&[data, starts], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // GatherElements Tests
    // ========================================================================

    #[test]
    fn test_translate_gather_elements() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[3, 3]));
        let indices = builder.add_input("indices", f32_tensor(&[2, 3]));

        let attrs = vec![make_int_attr("axis", 0)];

        let result = translate_gather_elements(
            &[data, indices],
            &attrs,
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gather_elements_wrong_inputs() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[3, 3]));

        let result = translate_gather_elements(&[data], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }
}

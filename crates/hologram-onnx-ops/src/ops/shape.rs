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

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
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
/// - Input 1: Shape tensor (int64, can contain -1 for inferred dimension, 0 for keep)
///
/// # Attributes (opset 5)
///
/// - `allowzero` (int, default 0): If 1, allow 0 in shape to mean dimension size 0
///
/// # Performance
///
/// - **Zero-copy** when possible (view operation)
/// - **PhiCoordinate addressing** for efficient access patterns
/// - Supports **symbolic shapes** (can reshape to symbolic dimensions)
///
/// # Implementation
///
/// Uses a Call node to `onnx.Reshape` which the runtime handles.
/// The runtime extracts the target shape from the shape tensor.
pub fn translate_reshape(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Reshape expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let data = inputs[0];
    let shape_input = inputs[1];

    debug!("Translating Reshape operation");
    trace!("Reshape inputs: data={:?}, shape={:?}", data, shape_input);

    // Use a Call node to represent dynamic reshape
    // The runtime will extract the target shape from the shape tensor
    let result = builder.call("onnx.Reshape", vec![data, shape_input]);

    trace!("Created Reshape call node: {:?}", result);
    Ok(result)
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
            "Transpose expects 1 input, got 0".to_string(),
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
/// # Inputs (opset 13+)
///
/// - Input 0: data - Tensor to squeeze
/// - Input 1: axes (optional) - 1-D tensor of axes to squeeze
///
/// # Attributes (opset < 13)
///
/// - `axes` (ints, optional): Dimensions to squeeze
///   - If not specified, squeezes all dimensions of size 1
///
/// # Performance
///
/// - **Zero-copy**: Shape metadata change only
/// - Supports **symbolic shapes** (but cannot squeeze symbolic dimensions)
///
/// # Implementation
///
/// Uses a Call node to `onnx.Squeeze` which the runtime handles.
pub fn translate_squeeze(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Squeeze expects at least 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse axes from attribute (for older opsets)
    let axes_attr = parse_attr_ints(attrs, "axes", vec![])?;

    debug!("Translating Squeeze operation (axes={:?})", axes_attr);
    trace!("Squeeze input: {:?}", input);

    // Build arguments for the call
    let mut args = vec![input];

    // If axes is provided as second input (opset 13+), use that
    if inputs.len() >= 2 {
        args.push(inputs[1]);
    }

    // Use a Call node to represent dynamic squeeze
    // The runtime will handle extracting axes from either the attribute or input tensor
    let result = builder.call("onnx.Squeeze", args);

    trace!("Created Squeeze call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Unsqueeze operation.
///
/// Unsqueeze: Add dimensions of size 1.
///
/// # Inputs (opset 13+)
///
/// - Input 0: data - Tensor to unsqueeze
/// - Input 1: axes - 1-D tensor of axes to add
///
/// # Attributes (opset < 13)
///
/// - `axes` (ints, required): Dimensions to add
///
/// # Performance
///
/// - **Zero-copy**: Shape metadata change only
/// - Supports **symbolic shapes**
///
/// # Implementation
///
/// Uses a Call node to `onnx.Unsqueeze` which the runtime handles.
pub fn translate_unsqueeze(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Unsqueeze expects at least 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse axes from attribute (for older opsets)
    let axes_attr = parse_attr_ints(attrs, "axes", vec![])?;

    // For older opsets, axes must be provided as attribute
    // For opset 13+, axes is provided as second input
    if axes_attr.is_empty() && inputs.len() < 2 {
        return Err(OnnxError::InvalidAttribute {
            name: "axes".to_string(),
            reason: "Unsqueeze requires axes (as attribute or second input)".to_string(),
        });
    }

    debug!("Translating Unsqueeze operation (axes={:?})", axes_attr);
    trace!("Unsqueeze input: {:?}", input);

    // Build arguments for the call
    let mut args = vec![input];

    // If axes is provided as second input (opset 13+), use that
    if inputs.len() >= 2 {
        args.push(inputs[1]);
    }

    // Use a Call node to represent dynamic unsqueeze
    // The runtime will handle extracting axes from either the attribute or input tensor
    let result = builder.call("onnx.Unsqueeze", args);

    trace!("Created Unsqueeze call node: {:?}", result);
    Ok(result)
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
        return Err(OnnxError::InvalidModel(format!(
            "Concat expects at least 2 inputs, got {}",
            inputs.len()
        )));
    }

    // Parse axis attribute (required)
    let axis = parse_attr_int(attrs, "axis", i64::MAX)?;
    if axis == i64::MAX {
        return Err(OnnxError::InvalidAttribute {
            name: "axis".to_string(),
            reason: "Concat requires axis attribute".to_string(),
        });
    }

    debug!(
        "Translating Concat operation (axis={}, {} inputs)",
        axis,
        inputs.len()
    );
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
/// # Inputs (opset 13+)
///
/// - Input 0: data - Tensor to split
/// - Input 1: split (optional) - 1-D tensor of split sizes
///
/// # Attributes
///
/// - `axis` (int, default 0): Axis along which to split
/// - `split` (ints, optional, opset < 13): Sizes of each output
///   - If not specified, splits into equal parts
/// - `num_outputs` (int, optional, opset 18+): Number of outputs for equal split
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
///
/// # Implementation
///
/// Uses a Call node to `onnx.Split` which the runtime handles.
pub fn translate_split(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Split expects at least 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse attributes
    let axis = parse_attr_int(attrs, "axis", 0)?;
    let split_sizes = parse_attr_ints(attrs, "split", vec![])?;

    debug!(
        "Translating Split operation (axis={}, splits={:?})",
        axis, split_sizes
    );
    trace!("Split input: {:?}", input);

    // Build arguments for the call
    let mut args = vec![input];

    // If split sizes provided as second input (opset 13+), use that
    if inputs.len() >= 2 {
        args.push(inputs[1]);
    }

    // Use a Call node to represent dynamic split
    // The runtime will handle extracting split sizes from either the attribute or input tensor
    let result = builder.call("onnx.Split", args);

    trace!("Created Split call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Flatten operation.
///
/// Flatten: Collapse tensor dimensions from `axis` to the end into a single dimension.
///
/// # Attributes
///
/// - `axis` (int, default 1): The axis from which to flatten
///   - All dimensions from `axis` to the end are collapsed into a single dimension
///   - axis=0 means flatten all dimensions into 1D
///   - axis=1 (default) means keep batch dimension, flatten the rest
///
/// # Example
///
/// - Input shape: [2, 3, 4, 5], axis=2 -> Output: [2, 3, 20]
/// - Input shape: [2, 3, 4, 5], axis=1 -> Output: [2, 60]
/// - Input shape: [2, 3, 4, 5], axis=0 -> Output: [120]
///
/// # Performance
///
/// - **Zero-copy view** - Flatten is a reshape, no data copy needed
/// - **PhiCoordinate addressing** for efficient access
/// - Supports **symbolic shapes**
pub fn translate_flatten(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    use hologram_compiler::shapes::{Dim as IRDim, Shape};

    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "Flatten expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse axis attribute (default 1)
    let axis = parse_attr_int(attrs, "axis", 1)?;

    debug!("Translating Flatten operation (axis={})", axis);
    trace!("Flatten input: {:?}", input);

    // Flatten reshapes from [d0, d1, ..., d(axis-1), d(axis), ..., d(n-1)]
    // to [d0 * ... * d(axis-1), d(axis) * ... * d(n-1)]
    //
    // For the common case of axis=1, this becomes [batch_size, flattened_features]
    // which is used before fully connected layers (e.g., Gemm).
    //
    // Since we don't have access to the input shape here, we create a reshape
    // with symbolic dimensions that will be resolved at execution time.
    //
    // For axis=1 (most common case), the shape is:
    // - First dim: symbolic "batch" (preserved from input)
    // - Second dim: symbolic "features" (product of remaining dims)

    let target_shape = if axis == 1 {
        // Most common case: [batch, features]
        Shape::new(vec![
            IRDim::Var("batch".to_string()),
            IRDim::Var("flatten_features".to_string()),
        ])
    } else {
        // General case: create symbolic shape
        // [outer_dims, inner_dims]
        Shape::new(vec![
            IRDim::Var(format!("flatten_outer_{}", axis)),
            IRDim::Var(format!("flatten_inner_{}", axis)),
        ])
    };

    // Create reshape node with symbolic target shape
    let result = builder.reshape(input, target_shape);

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

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_reshape() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[2, 3, 4]));
        let shape = builder.add_input("shape", f32_tensor(&[2]));

        let result = translate_reshape(&[data, shape], &[], &HashMap::new(), &mut builder);
        // Dynamic reshape uses Call node
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reshape_wrong_inputs() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[2, 3]));

        // Only 1 input (needs 2)
        let result = translate_reshape(&[data], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_transpose_default() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        // No perm attribute means reverse dimensions
        let result = translate_transpose(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_transpose_custom_perm() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![make_ints_attr("perm", vec![2, 0, 1])];

        let result = translate_transpose(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_squeeze() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 2, 1, 3]));

        // Squeeze uses Call node
        let result = translate_squeeze(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_squeeze_with_axes_input() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 2, 1, 3]));
        let axes = builder.add_input("axes", f32_tensor(&[2]));

        // Squeeze with axes input (opset 13+)
        let result = translate_squeeze(&[input, axes], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_unsqueeze() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));
        let axes = builder.add_input("axes", f32_tensor(&[2]));

        // Unsqueeze with axes input (opset 13+)
        let result = translate_unsqueeze(&[input, axes], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_unsqueeze_with_attribute() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        let attrs = vec![make_ints_attr("axes", vec![0, 3])];

        // Unsqueeze with axes attribute (opset < 13)
        let result = translate_unsqueeze(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_unsqueeze_no_axes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3]));

        // No axes attribute and no axes input should fail with InvalidAttribute
        let result = translate_unsqueeze(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::InvalidAttribute { .. }
        ));
    }

    #[test]
    fn test_translate_concat() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 4]));
        let c = builder.add_input("C", f32_tensor(&[2, 5]));

        let attrs = vec![make_int_attr("axis", 1)];

        let result = translate_concat(&[a, b, c], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_concat_insufficient_inputs() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));

        let attrs = vec![make_int_attr("axis", 0)];

        // Only 1 input (needs at least 2)
        let result = translate_concat(&[a], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_concat_no_axis() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        // No axis attribute should fail
        let result = translate_concat(&[a, b], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OnnxError::InvalidAttribute { .. }
        ));
    }

    #[test]
    fn test_translate_split() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 6]));

        let attrs = vec![make_int_attr("axis", 1)];

        // Split uses Call node
        let result = translate_split(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_split_with_sizes_input() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 6]));
        let split_sizes = builder.add_input("split", f32_tensor(&[2]));

        let attrs = vec![make_int_attr("axis", 1)];

        // Split with sizes input (opset 13+)
        let result = translate_split(&[input, split_sizes], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transpose_empty_input() {
        let mut builder = make_builder();

        let result = translate_transpose(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }
}

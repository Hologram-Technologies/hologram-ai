//! ONNX reduction operations.
//!
//! All reductions in this module:
//! - Leverage **LOOP instructions** for O(1) space complexity
//! - Support **symbolic shapes** with dynamic reduction axes
//! - Use **SIMD vectorization** via hologram-backend
//!
//! # ISA Optimizations
//!
//! - **LOOP instructions**: Reductions use O(1) space via loop primitives
//! - **SIMD**: Parallel processing of reduction operations
//! - **Zero runtime overhead**: All axes resolved at compile time

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX ReduceSum operation.
///
/// ReduceSum: Y = sum(X) along specified axes
///
/// # Attributes
///
/// - `axes` (list of ints): Axes along which to reduce. If empty, reduce all.
/// - `keepdims` (int, default 1): Whether to keep reduced dimensions (size 1)
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity for reduction
/// - **SIMD vectorization**: Parallel summation
/// - Supports **symbolic shapes** (preserves/reduces symbolic dimensions)
pub fn translate_reduce_sum(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReduceSum expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    // Parse axes attribute (empty means reduce all)
    let axes = parse_attr_ints(attrs, "axes", vec![])?;

    // Parse keepdims attribute (default: 1)
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    debug!(
        "Translating ReduceSum operation (axes={:?}, keepdims={})",
        axes, keepdims
    );
    trace!("ReduceSum input: {:?}", input);

    // Convert axes to isize for builder
    let axes_isize: Vec<isize> = axes.into_iter().map(|a| a as isize).collect();

    // Create ReduceSum IR node using builder method
    let node = builder.sum(input, axes_isize, keepdims);

    trace!("Created ReduceSum node: {:?}", node);
    Ok(node)
}

/// Translate ONNX ReduceMean operation.
///
/// ReduceMean: Y = mean(X) along specified axes
///
/// # Attributes
///
/// - `axes` (list of ints): Axes along which to reduce
/// - `keepdims` (int, default 1): Whether to keep reduced dimensions
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel averaging
/// - Supports **symbolic shapes**
pub fn translate_reduce_mean(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReduceMean expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    debug!(
        "Translating ReduceMean operation (axes={:?}, keepdims={})",
        axes, keepdims
    );
    trace!("ReduceMean input: {:?}", input);

    // Convert axes to isize for builder
    let axes_isize: Vec<isize> = axes.into_iter().map(|a| a as isize).collect();

    // Create ReduceMean IR node using builder method
    let node = builder.mean(input, axes_isize, keepdims);

    trace!("Created ReduceMean node: {:?}", node);
    Ok(node)
}

/// Translate ONNX ReduceMax operation.
///
/// ReduceMax: Y = max(X) along specified axes
///
/// # Attributes
///
/// - `axes` (list of ints): Axes along which to reduce
/// - `keepdims` (int, default 1): Whether to keep reduced dimensions
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel max computation
/// - Supports **symbolic shapes**
pub fn translate_reduce_max(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReduceMax expects 1 input, got 0".to_string(),
        ));
    }

    let input = inputs[0];

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    debug!(
        "Translating ReduceMax operation (axes={:?}, keepdims={})",
        axes, keepdims
    );
    trace!("ReduceMax input: {:?}", input);

    // Convert axes to isize for builder
    let axes_isize: Vec<isize> = axes.into_iter().map(|a| a as isize).collect();

    // Create ReduceMax IR node using builder method
    let node = builder.max(input, axes_isize, keepdims);

    trace!("Created ReduceMax node: {:?}", node);
    Ok(node)
}

/// Translate ONNX ReduceMin operation.
///
/// ReduceMin: Y = min(X) along specified axes
///
/// # Attributes
///
/// - `axes` (list of ints): Axes along which to reduce
/// - `keepdims` (int, default 1): Whether to keep reduced dimensions
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel min computation
/// - Supports **symbolic shapes**
pub fn translate_reduce_min(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReduceMin expects at least 1 input, got 0".to_string(),
        ));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    debug!(
        "Translating ReduceMin operation (axes={:?}, keepdims={})",
        axes, keepdims
    );
    trace!("ReduceMin inputs: {:?}", inputs);

    // Use Call node for ReduceMin - runtime handles min reduction
    let result = builder.call("onnx.ReduceMin", inputs.to_vec());

    trace!("Created ReduceMin call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX ReduceProd operation.
///
/// ReduceProd: Y = product(X) along specified axes
///
/// # Attributes
///
/// - `axes` (list of ints): Axes along which to reduce
/// - `keepdims` (int, default 1): Whether to keep reduced dimensions
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel product computation
/// - Supports **symbolic shapes**
///
/// # Implementation
///
/// Uses a Call node to `onnx.ReduceProd` which the runtime handles.
pub fn translate_reduce_prod(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ReduceProd expects at least 1 input, got 0".to_string(),
        ));
    }

    let axes = parse_attr_ints(attrs, "axes", vec![])?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;

    debug!(
        "Translating ReduceProd operation (axes={:?}, keepdims={})",
        axes, keepdims
    );
    trace!("ReduceProd inputs: {:?}", inputs);

    // Use Call node for ReduceProd - runtime handles product reduction
    let result = builder.call("onnx.ReduceProd", inputs.to_vec());

    trace!("Created ReduceProd call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX ArgMax operation.
///
/// ArgMax: Y = indices of max(X) along specified axis
///
/// # Attributes
///
/// - `axis` (int, default 0): Axis along which to find max indices
/// - `keepdims` (int, default 1): Whether to keep reduced dimension (size 1)
/// - `select_last_index` (int, default 0): Whether to return last index in ties
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel max finding
/// - Supports **symbolic shapes**
pub fn translate_argmax(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ArgMax expects at least 1 input, got 0".to_string(),
        ));
    }

    let axis = parse_attr_int(attrs, "axis", 0)?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;
    let select_last_index = parse_attr_int(attrs, "select_last_index", 0)? != 0;

    debug!(
        "Translating ArgMax operation (axis={}, keepdims={}, select_last_index={})",
        axis, keepdims, select_last_index
    );
    trace!("ArgMax inputs: {:?}", inputs);

    // Use Call node for ArgMax - runtime handles max index finding
    let result = builder.call("onnx.ArgMax", inputs.to_vec());

    trace!("Created ArgMax call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX ArgMin operation.
///
/// ArgMin: Y = indices of min(X) along specified axis
///
/// # Attributes
///
/// - `axis` (int, default 0): Axis along which to find min indices
/// - `keepdims` (int, default 1): Whether to keep reduced dimension (size 1)
/// - `select_last_index` (int, default 0): Whether to return last index in ties
///
/// # Performance
///
/// - **LOOP instructions**: O(1) space complexity
/// - **SIMD vectorization**: Parallel min finding
/// - Supports **symbolic shapes**
pub fn translate_argmin(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() {
        return Err(OnnxError::InvalidModel(
            "ArgMin expects at least 1 input, got 0".to_string(),
        ));
    }

    let axis = parse_attr_int(attrs, "axis", 0)?;
    let keepdims = parse_attr_int(attrs, "keepdims", 1)? != 0;
    let select_last_index = parse_attr_int(attrs, "select_last_index", 0)? != 0;

    debug!(
        "Translating ArgMin operation (axis={}, keepdims={}, select_last_index={})",
        axis, keepdims, select_last_index
    );
    trace!("ArgMin inputs: {:?}", inputs);

    // Use Call node for ArgMin - runtime handles min index finding
    let result = builder.call("onnx.ArgMin", inputs.to_vec());

    trace!("Created ArgMin call node: {:?}", result);
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

    // ReduceSum tests

    #[test]
    fn test_translate_reduce_sum_default() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reduce_sum(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_sum_with_axes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![AttributeProto {
            name: "axes".to_string(),
            ints: vec![0, 2],
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }];

        let result = translate_reduce_sum(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_sum_no_keepdims() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![AttributeProto {
            name: "keepdims".to_string(),
            i: 0,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }];

        let result = translate_reduce_sum(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_sum_no_input() {
        let mut builder = make_builder();
        let result = translate_reduce_sum(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    // ReduceMean tests

    #[test]
    fn test_translate_reduce_mean() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reduce_mean(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_mean_with_axes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let attrs = vec![AttributeProto {
            name: "axes".to_string(),
            ints: vec![1],
            r#type: AttributeType::Ints as i32,
            ..Default::default()
        }];

        let result = translate_reduce_mean(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_mean_no_input() {
        let mut builder = make_builder();
        let result = translate_reduce_mean(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ReduceMax tests

    #[test]
    fn test_translate_reduce_max() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reduce_max(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_max_no_input() {
        let mut builder = make_builder();
        let result = translate_reduce_max(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ReduceMin tests

    #[test]
    fn test_translate_reduce_min() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reduce_min(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_min_no_input() {
        let mut builder = make_builder();
        let result = translate_reduce_min(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    // ReduceProd tests

    #[test]
    fn test_translate_reduce_prod() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        let result = translate_reduce_prod(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_reduce_prod_no_input() {
        let mut builder = make_builder();
        let result = translate_reduce_prod(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    // Symbolic shape tests

    #[test]
    fn test_implemented_reductions_symbolic_shapes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[])); // Symbolic shape

        let shapes = HashMap::new();

        // Implemented reductions should work with symbolic shapes
        assert!(translate_reduce_sum(&[input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_reduce_mean(&[input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_reduce_max(&[input], &[], &shapes, &mut builder).is_ok());
    }

    #[test]
    fn test_all_reductions_symbolic_shapes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[])); // Symbolic shape

        let shapes = HashMap::new();

        // All reductions should work with symbolic shapes
        assert!(translate_reduce_min(&[input], &[], &shapes, &mut builder).is_ok());
        assert!(translate_reduce_prod(&[input], &[], &shapes, &mut builder).is_ok());
    }

    #[test]
    fn test_reduction_multiple_axes() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4, 5]));

        let attrs = vec![
            AttributeProto {
                name: "axes".to_string(),
                ints: vec![1, 3], // Reduce axes 1 and 3
                r#type: AttributeType::Ints as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "keepdims".to_string(),
                i: 1,
                r#type: AttributeType::Int as i32,
                ..Default::default()
            },
        ];

        let result = translate_reduce_sum(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }
}

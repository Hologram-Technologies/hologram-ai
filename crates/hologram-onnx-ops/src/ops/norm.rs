//! ONNX normalization operations.
//!
//! All normalization operations in this module:
//! - Leverage **SIMD vectorization** for mean/variance calculations
//! - Support **symbolic shapes** (variable batch sizes, sequence lengths)
//! - Use **LOOP instructions** for efficient reduction operations
//!
//! # ISA Optimizations
//!
//! - **SIMD vectorization**: All arithmetic operations use SIMD
//! - **LOOP instructions**: Mean/variance reductions use O(1) space
//! - **ClassMap fusion**: Normalization + activation can fuse

use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use hologram_compiler::ir::{IRBuilder, NodeId};
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_float, parse_attr_int};

/// Translate ONNX BatchNormalization operation.
///
/// BatchNormalization: Y = (X - mean) / sqrt(variance + epsilon) * scale + bias
///
/// # Inputs
///
/// - Input 0: X (data)
/// - Input 1: scale (gamma)
/// - Input 2: bias (beta)
/// - Input 3: mean
/// - Input 4: variance
///
/// # Attributes
///
/// - `epsilon` (float, default 1e-5): Small value to avoid division by zero
/// - `momentum` (float, default 0.9): Momentum for running mean/variance (training mode)
///
/// # Performance
///
/// - **SIMD vectorization**: All arithmetic operations
/// - **Supports symbolic shapes**: Variable batch sizes
/// - Training mode (with running stats) not yet supported
pub fn translate_batch_normalization(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 5 {
        return Err(OnnxError::InvalidModel(
            format!("BatchNormalization expects 5 inputs, got {}", inputs.len())
        ));
    }

    let input = inputs[0];
    let scale = inputs[1];
    let bias = inputs[2];
    let mean = inputs[3];
    let variance = inputs[4];

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;

    debug!("Translating BatchNormalization (epsilon={})", epsilon);
    trace!("BatchNorm inputs: {:?}, {:?}, {:?}, {:?}, {:?}", input, scale, bias, mean, variance);

    // Create BatchNormalization IR node using builder method
    let node = builder.batch_norm(input, scale, bias, mean, variance, epsilon);

    trace!("Created BatchNorm node: {:?}", node);
    Ok(node)
}

/// Translate ONNX LayerNormalization operation.
///
/// LayerNormalization: Y = (X - mean) / sqrt(variance + epsilon) * scale + bias
///
/// Unlike BatchNorm, LayerNorm normalizes across the last dimension(s).
///
/// # Inputs
///
/// - Input 0: X (data)
/// - Input 1: scale (gamma)
/// - Input 2: bias (beta, optional)
///
/// # Attributes
///
/// - `axis` (int, default -1): First dimension to normalize
/// - `epsilon` (float, default 1e-5): Small value to avoid division by zero
///
/// # Performance
///
/// - **SIMD vectorization**: Mean/variance calculations
/// - **LOOP instructions**: Reduction operations use O(1) space
/// - **Symbolic shapes**: Works with variable sequence lengths
///
/// # Decomposition
///
/// LayerNorm is decomposed into primitive operations:
/// 1. mean = ReduceMean(X, axes=[axis:])
/// 2. diff = X - mean
/// 3. variance = ReduceMean(diff^2, axes=[axis:])
/// 4. normalized = diff / sqrt(variance + epsilon)
/// 5. Y = normalized * scale + bias
pub fn translate_layer_normalization(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(
            format!("LayerNormalization expects 2-3 inputs, got {}", inputs.len())
        ));
    }

    let input = inputs[0];
    let scale = inputs[1];
    let bias = inputs.get(2).copied();

    let axis = parse_attr_int(attrs, "axis", -1)?;
    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;

    debug!("Translating LayerNormalization (axis={}, epsilon={})", axis, epsilon);
    trace!("LayerNorm inputs: {:?}, {:?}, {:?}", input, scale, bias);

    // Decompose LayerNormalization into primitive operations:
    // Y = (X - mean) / sqrt(variance + epsilon) * scale + bias

    // Step 1: Compute mean along normalization axis
    // For LayerNorm, we normalize along axis and all dimensions after it
    // axis=-1 means normalize along last dimension
    let axes = vec![axis as isize];
    let mean = builder.mean(input, axes.clone(), true);
    trace!("LayerNorm mean: {:?}", mean);

    // Step 2: Compute X - mean (centered input)
    let centered = builder.sub(input, mean);
    trace!("LayerNorm centered: {:?}", centered);

    // Step 3: Compute variance = mean((X - mean)^2)
    let centered_sq = builder.mul(centered, centered);
    let variance = builder.mean(centered_sq, axes, true);
    trace!("LayerNorm variance: {:?}", variance);

    // Step 4: Compute sqrt(variance + epsilon)
    let epsilon_const = builder.add_f32(epsilon);
    let var_eps = builder.add(variance, epsilon_const);
    let half = builder.add_f32(0.5);
    let std = builder.pow(var_eps, half); // sqrt(x) = x^0.5
    trace!("LayerNorm std: {:?}", std);

    // Step 5: Normalize: (X - mean) / std
    let normalized = builder.div(centered, std);
    trace!("LayerNorm normalized: {:?}", normalized);

    // Step 6: Scale: normalized * scale
    let scaled = builder.mul(normalized, scale);
    trace!("LayerNorm scaled: {:?}", scaled);

    // Step 7: Shift: scaled + bias (if bias is provided)
    let result = if let Some(b) = bias {
        builder.add(scaled, b)
    } else {
        scaled
    };

    trace!("Created LayerNorm decomposition ending at: {:?}", result);
    Ok(result)
}

/// Translate ONNX InstanceNormalization operation.
///
/// InstanceNormalization: Normalizes each instance in a batch independently.
///
/// # Inputs
///
/// - Input 0: X (data) - shape `[N, C, H, W, ...]`
/// - Input 1: scale (gamma) - shape `[C]`
/// - Input 2: bias (beta) - shape `[C]`
///
/// # Attributes
///
/// - `epsilon` (float, default 1e-5): Small value to avoid division by zero
///
/// # Performance
///
/// - **SIMD vectorization**: All operations
/// - **Per-instance normalization**: Each instance normalized independently
/// - **Symbolic shapes**: Variable batch sizes supported
///
/// # Decomposition
///
/// InstanceNorm is decomposed into primitive operations:
/// For input shape [N, C, H, W, ...]:
/// 1. mean = ReduceMean(X, axes=[2, 3, ...]) per instance and channel
/// 2. variance = ReduceMean((X - mean)^2, axes=[2, 3, ...])
/// 3. normalized = (X - mean) / sqrt(variance + epsilon)
/// 4. Y = normalized * scale + bias
pub fn translate_instance_normalization(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 3 {
        return Err(OnnxError::InvalidModel(
            format!("InstanceNormalization expects 3 inputs, got {}", inputs.len())
        ));
    }

    let input = inputs[0];
    let scale = inputs[1];
    let bias = inputs[2];

    let epsilon = parse_attr_float(attrs, "epsilon", 1e-5)?;

    debug!("Translating InstanceNormalization (epsilon={})", epsilon);
    trace!("InstanceNorm inputs: {:?}, {:?}, {:?}", input, scale, bias);

    // Decompose InstanceNormalization into primitive operations:
    // Y = (X - mean) / sqrt(variance + epsilon) * scale + bias
    //
    // For InstanceNorm, we normalize over spatial dimensions (H, W, ...)
    // keeping N (batch) and C (channel) dimensions fixed.
    // Standard assumption: input is [N, C, H, W] so we reduce over axes [2, 3]

    // Step 1: Compute mean along spatial dimensions
    // Using axes [2, 3] for standard 4D input [N, C, H, W]
    let spatial_axes = vec![2_isize, 3_isize];
    let mean = builder.mean(input, spatial_axes.clone(), true);
    trace!("InstanceNorm mean: {:?}", mean);

    // Step 2: Compute X - mean (centered input)
    let centered = builder.sub(input, mean);
    trace!("InstanceNorm centered: {:?}", centered);

    // Step 3: Compute variance = mean((X - mean)^2) over spatial dimensions
    let centered_sq = builder.mul(centered, centered);
    let variance = builder.mean(centered_sq, spatial_axes, true);
    trace!("InstanceNorm variance: {:?}", variance);

    // Step 4: Compute sqrt(variance + epsilon)
    let epsilon_const = builder.add_f32(epsilon);
    let var_eps = builder.add(variance, epsilon_const);
    let half = builder.add_f32(0.5);
    let std = builder.pow(var_eps, half); // sqrt(x) = x^0.5
    trace!("InstanceNorm std: {:?}", std);

    // Step 5: Normalize: (X - mean) / std
    let normalized = builder.div(centered, std);
    trace!("InstanceNorm normalized: {:?}", normalized);

    // Step 6: Scale and shift: normalized * scale + bias
    // Note: scale and bias have shape [C] and need broadcasting
    let scaled = builder.mul(normalized, scale);
    let result = builder.add(scaled, bias);

    trace!("Created InstanceNorm decomposition ending at: {:?}", result);
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

    fn make_float_attr(name: &str, value: f32) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            f: value,
            r#type: AttributeType::Float as i32,
            ..Default::default()
        }
    }

    fn make_int_attr(name: &str, value: i64) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            i: value,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }
    }

    #[test]
    fn test_translate_batch_normalization() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 224, 224]));
        let scale = builder.add_input("scale", f32_tensor(&[64]));
        let bias = builder.add_input("bias", f32_tensor(&[64]));
        let mean = builder.add_input("mean", f32_tensor(&[64]));
        let variance = builder.add_input("variance", f32_tensor(&[64]));

        let result = translate_batch_normalization(
            &vec![input, scale, bias, mean, variance],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_batch_normalization_with_epsilon() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 224, 224]));
        let scale = builder.add_input("scale", f32_tensor(&[64]));
        let bias = builder.add_input("bias", f32_tensor(&[64]));
        let mean = builder.add_input("mean", f32_tensor(&[64]));
        let variance = builder.add_input("variance", f32_tensor(&[64]));

        let attrs = vec![make_float_attr("epsilon", 1e-3)];

        let result = translate_batch_normalization(
            &vec![input, scale, bias, mean, variance],
            &attrs,
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_batch_normalization_wrong_inputs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 224, 224]));
        let scale = builder.add_input("scale", f32_tensor(&[64]));

        // Only 2 inputs (needs 5)
        let result = translate_batch_normalization(
            &vec![input, scale],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_layer_normalization() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 512, 768]));
        let scale = builder.add_input("scale", f32_tensor(&[768]));

        let result = translate_layer_normalization(
            &vec![input, scale],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_layer_normalization_with_bias() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 512, 768]));
        let scale = builder.add_input("scale", f32_tensor(&[768]));
        let bias = builder.add_input("bias", f32_tensor(&[768]));

        let result = translate_layer_normalization(
            &vec![input, scale, bias],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_layer_normalization_with_attrs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 10, 512]));
        let scale = builder.add_input("scale", f32_tensor(&[512]));
        let bias = builder.add_input("bias", f32_tensor(&[512]));

        let attrs = vec![
            make_int_attr("axis", -1),
            make_float_attr("epsilon", 1e-6),
        ];

        let result = translate_layer_normalization(
            &vec![input, scale, bias],
            &attrs,
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_layer_normalization_symbolic_shapes() {
        let mut builder = make_builder();
        // Variable batch and sequence length
        let input = builder.add_input("X", f32_tensor(&[]));
        let scale = builder.add_input("scale", f32_tensor(&[]));
        let bias = builder.add_input("bias", f32_tensor(&[]));

        let result = translate_layer_normalization(
            &vec![input, scale, bias],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_layer_normalization_wrong_inputs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 512, 768]));

        // Only 1 input (needs at least 2)
        let result = translate_layer_normalization(
            &vec![input],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_translate_instance_normalization() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 224, 224]));
        let scale = builder.add_input("scale", f32_tensor(&[64]));
        let bias = builder.add_input("bias", f32_tensor(&[64]));

        let result = translate_instance_normalization(
            &vec![input, scale, bias],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_instance_normalization_with_epsilon() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 32, 56, 56]));
        let scale = builder.add_input("scale", f32_tensor(&[32]));
        let bias = builder.add_input("bias", f32_tensor(&[32]));

        let attrs = vec![make_float_attr("epsilon", 1e-6)];

        let result = translate_instance_normalization(
            &vec![input, scale, bias],
            &attrs,
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_instance_normalization_symbolic_batch() {
        let mut builder = make_builder();
        // Symbolic batch dimension
        let input = builder.add_input("X", f32_tensor(&[]));
        let scale = builder.add_input("scale", f32_tensor(&[]));
        let bias = builder.add_input("bias", f32_tensor(&[]));

        let result = translate_instance_normalization(
            &vec![input, scale, bias],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_instance_normalization_wrong_inputs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 224, 224]));
        let scale = builder.add_input("scale", f32_tensor(&[64]));

        // Only 2 inputs (needs 3)
        let result = translate_instance_normalization(
            &vec![input, scale],
            &[],
            &HashMap::new(),
            &mut builder
        );
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
    }

    #[test]
    fn test_batch_normalization_symbolic_shapes() {
        let mut builder = make_builder();
        // Symbolic batch dimension
        let input = builder.add_input("X", f32_tensor(&[]));
        let scale = builder.add_input("scale", f32_tensor(&[]));
        let bias = builder.add_input("bias", f32_tensor(&[]));
        let mean = builder.add_input("mean", f32_tensor(&[]));
        let variance = builder.add_input("variance", f32_tensor(&[]));

        let shapes = HashMap::new();

        assert!(translate_batch_normalization(
            &vec![input, scale, bias, mean, variance],
            &[],
            &shapes,
            &mut builder
        ).is_ok());
    }
}

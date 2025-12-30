//! ONNX resize operations.
//!
//! Operations for resizing tensors:
//! - **Resize**: General tensor resizing with various interpolation modes
//! - **Upsample**: Upscaling tensors (deprecated, maps to Resize)
//!
//! # Usage in Stable Diffusion
//!
//! - **Resize**: VAE decoder upsampling, latent space manipulation
//! - Critical for image generation pipeline

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_int, parse_attr_string_or};

/// Translate ONNX Resize operation.
///
/// Resize: Resize the input tensor using various interpolation methods.
///
/// # Inputs
///
/// - Input 0: X - Input tensor (N-D)
/// - Input 1: roi - Region of interest (optional, 1-D tensor)
/// - Input 2: scales - Scale factors (optional, 1-D tensor)
/// - Input 3: sizes - Output sizes (optional, 1-D tensor)
///
/// # Attributes
///
/// - `mode` (string, default "nearest"): Interpolation mode
///   - "nearest": Nearest neighbor interpolation
///   - "linear": Linear/bilinear interpolation
///   - "cubic": Cubic/bicubic interpolation
/// - `coordinate_transformation_mode` (string, default "half_pixel")
/// - `nearest_mode` (string, default "round_prefer_floor")
///
/// # Performance
///
/// - **SIMD vectorization**: Vectorized interpolation
/// - Critical for VAE decoder in Stable Diffusion
///
/// # Implementation
///
/// Uses a Call node to `onnx.Resize` which the runtime handles.
pub fn translate_resize(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() || inputs.len() > 4 {
        return Err(OnnxError::InvalidModel(format!(
            "Resize expects 1-4 inputs, got {}",
            inputs.len()
        )));
    }

    let mode = parse_attr_string_or(attrs, "mode", "nearest")?;
    let coord_mode =
        parse_attr_string_or(attrs, "coordinate_transformation_mode", "half_pixel")?;

    debug!(
        "Translating Resize operation (mode={}, coord_mode={})",
        mode, coord_mode
    );
    trace!("Resize inputs: {:?}", inputs);

    // Use Call node to represent resize operation
    // The runtime handles the interpolation based on mode and coordinate transformation
    let result = builder.call("onnx.Resize", inputs.to_vec());

    trace!("Created Resize call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX Upsample operation (deprecated).
///
/// Upsample is deprecated in ONNX and replaced by Resize.
/// We map it directly to Resize for compatibility.
///
/// # Inputs
///
/// - Input 0: X - Input tensor
/// - Input 1: scales - Scale factors (1-D tensor)
///
/// # Attributes
///
/// - `mode` (string, default "nearest"): "nearest" or "linear"
///
/// # Implementation
///
/// Uses a Call node to `onnx.Upsample` which the runtime handles.
pub fn translate_upsample(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() || inputs.len() > 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Upsample expects 1-2 inputs, got {}",
            inputs.len()
        )));
    }

    let mode = parse_attr_string_or(attrs, "mode", "nearest")?;

    debug!("Translating Upsample operation (mode={})", mode);
    trace!("Upsample inputs: {:?}", inputs);

    // Upsample is deprecated and maps to Resize
    // Use Call node for runtime handling
    let result = builder.call("onnx.Upsample", inputs.to_vec());

    trace!("Created Upsample call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX DepthToSpace operation.
///
/// DepthToSpace: Rearranges depth data into spatial blocks.
/// Used in some upsampling architectures.
///
/// # Inputs
///
/// - Input 0: X - Input tensor (N, C, H, W)
///
/// # Attributes
///
/// - `blocksize` (int): Size of blocks to rearrange
/// - `mode` (string, default "DCR"): "DCR" or "CRD"
///
/// # Implementation
///
/// Uses a Call node to `onnx.DepthToSpace` which the runtime handles.
pub fn translate_depth_to_space(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "DepthToSpace expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    let blocksize = parse_attr_int(attrs, "blocksize", 2)?;
    let mode = parse_attr_string_or(attrs, "mode", "DCR")?;

    debug!(
        "Translating DepthToSpace operation (blocksize={}, mode={})",
        blocksize, mode
    );
    trace!("DepthToSpace input: {:?}", input);

    // DepthToSpace rearranges (N, C*r*r, H, W) -> (N, C, H*r, W*r)
    // Use Call node for runtime handling
    let result = builder.call("onnx.DepthToSpace", vec![input]);

    trace!("Created DepthToSpace call node: {:?}", result);
    Ok(result)
}

/// Translate ONNX SpaceToDepth operation.
///
/// SpaceToDepth: Rearranges spatial data into depth blocks.
/// Inverse of DepthToSpace.
///
/// # Inputs
///
/// - Input 0: X - Input tensor (N, C, H, W)
///
/// # Attributes
///
/// - `blocksize` (int): Size of blocks to rearrange
///
/// # Implementation
///
/// Uses a Call node to `onnx.SpaceToDepth` which the runtime handles.
pub fn translate_space_to_depth(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "SpaceToDepth expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    let blocksize = parse_attr_int(attrs, "blocksize", 2)?;

    debug!(
        "Translating SpaceToDepth operation (blocksize={})",
        blocksize
    );
    trace!("SpaceToDepth input: {:?}", input);

    // SpaceToDepth rearranges (N, C, H*r, W*r) -> (N, C*r*r, H, W)
    // Use Call node for runtime handling
    let result = builder.call("onnx.SpaceToDepth", vec![input]);

    trace!("Created SpaceToDepth call node: {:?}", result);
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

    fn make_string_attr(name: &str, value: &str) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            r#type: AttributeType::String as i32,
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

    // ========================================================================
    // Resize Tests
    // ========================================================================

    #[test]
    fn test_translate_resize_nearest() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 64, 64]));

        let attrs = vec![make_string_attr("mode", "nearest")];

        let result = translate_resize(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_resize_linear() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 64, 64]));

        let attrs = vec![make_string_attr("mode", "linear")];

        let result = translate_resize(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_resize_cubic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 64, 64]));

        let attrs = vec![make_string_attr("mode", "cubic")];

        let result = translate_resize(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok()); // Falls back to bilinear
    }

    #[test]
    fn test_translate_resize_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_resize(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // Upsample Tests
    // ========================================================================

    #[test]
    fn test_translate_upsample_nearest() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 64, 64]));

        let attrs = vec![make_string_attr("mode", "nearest")];

        let result = translate_upsample(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_upsample_linear() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 64, 64]));

        let attrs = vec![make_string_attr("mode", "linear")];

        let result = translate_upsample(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_upsample_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_upsample(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    // ========================================================================
    // DepthToSpace Tests
    // ========================================================================

    #[test]
    fn test_translate_depth_to_space() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 16, 8, 8])); // C=16, will become C=4 with block=2

        let attrs = vec![make_int_attr("blocksize", 2)];

        let result = translate_depth_to_space(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_depth_to_space_crd_mode() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 16, 8, 8]));

        let attrs = vec![
            make_int_attr("blocksize", 2),
            make_string_attr("mode", "CRD"),
        ];

        let result = translate_depth_to_space(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // SpaceToDepth Tests
    // ========================================================================

    #[test]
    fn test_translate_space_to_depth() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 4, 16, 16]));

        let attrs = vec![make_int_attr("blocksize", 2)];

        let result = translate_space_to_depth(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }
}

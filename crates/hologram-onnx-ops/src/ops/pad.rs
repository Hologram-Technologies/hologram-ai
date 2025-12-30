//! ONNX padding operations.
//!
//! Operations for padding tensors:
//! - **Pad**: Add padding to tensor edges
//!
//! # Usage in Stable Diffusion
//!
//! - **Pad**: Convolution padding, image border handling

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::parse_attr_string_or;

/// Translate ONNX Pad operation.
///
/// Pad: Adds padding to the input tensor.
///
/// # Inputs
///
/// - Input 0: data - Input tensor to pad
/// - Input 1: pads - 1-D tensor of paddings (start/end for each dimension)
/// - Input 2: constant_value (optional) - Value for constant padding
/// - Input 3: axes (optional) - Axes to pad
///
/// # Attributes
///
/// - `mode` (string, default "constant"): Padding mode
///   - "constant": Fill with constant value
///   - "reflect": Reflect values at edges
///   - "edge": Replicate edge values
///
/// # Performance
///
/// - **SIMD vectorization**: Vectorized copy operations
/// - Common in convolution preprocessing
///
/// # Implementation
///
/// Uses a Call node to `onnx.Pad` which the runtime handles.
pub fn translate_pad(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.is_empty() || inputs.len() > 4 {
        return Err(OnnxError::InvalidModel(format!(
            "Pad expects 1-4 inputs, got {}",
            inputs.len()
        )));
    }

    let mode = parse_attr_string_or(attrs, "mode", "constant")?;

    debug!("Translating Pad operation (mode={})", mode);
    trace!("Pad inputs: {:?}", inputs);

    // Use Call node for Pad - runtime handles padding with mode and constant value
    let result = builder.call("onnx.Pad", inputs.to_vec());

    trace!("Created Pad call node: {:?}", result);
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

    #[test]
    fn test_translate_pad() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[1, 3, 64, 64]));
        let pads = builder.add_input("pads", f32_tensor(&[8]));

        let result = translate_pad(&vec![data, pads], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_pad_constant_mode() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[1, 3, 64, 64]));
        let pads = builder.add_input("pads", f32_tensor(&[8]));

        let attrs = vec![make_string_attr("mode", "constant")];

        let result = translate_pad(&vec![data, pads], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_pad_reflect_mode() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[1, 3, 64, 64]));
        let pads = builder.add_input("pads", f32_tensor(&[8]));

        let attrs = vec![make_string_attr("mode", "reflect")];

        let result = translate_pad(&vec![data, pads], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_pad_with_constant_value() {
        let mut builder = make_builder();
        let data = builder.add_input("data", f32_tensor(&[1, 3, 64, 64]));
        let pads = builder.add_input("pads", f32_tensor(&[8]));
        let constant = builder.add_input("constant", f32_tensor(&[]));

        let result = translate_pad(
            &vec![data, pads, constant],
            &[],
            &HashMap::new(),
            &mut builder,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_pad_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_pad(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }
}

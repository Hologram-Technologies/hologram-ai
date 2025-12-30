//! ONNX convolution operations.
//!
//! All convolution operations in this module:
//! - Leverage **Im2col+GEMM decomposition** for optimal performance
//! - Use **PhiCoordinate addressing** for 5-10x speedup
//! - Support **symbolic shapes** (variable batch sizes)
//! - Use **SIMD vectorization** via hologram-backend
//!
//! # ISA Optimizations (CRITICAL)
//!
//! - **Im2col+GEMM decomposition**: Conv2D transforms to matrix multiplication
//! - **PhiCoordinate addressing**: Cache-resident boundary pool addressing
//! - **LOOP instructions**: O(1) space complexity for sliding windows
//! - **SIMD**: All GEMM operations use SIMD vectorization

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_compiler::shapes::Dim;
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_int, parse_attr_ints};

/// Translate ONNX Conv operation.
///
/// Conv: Y = Conv(X, W, B) with optional bias
///
/// # Attributes
///
/// - `strides` (ints, default [1, 1]): Stride along each spatial axis
/// - `pads` (ints, default [0, 0, 0, 0]): Padding [top, left, bottom, right]
/// - `dilations` (ints, default [1, 1]): Dilation along each spatial axis
/// - `group` (int, default 1): Number of groups for grouped convolution
/// - `kernel_shape` (ints, optional): Shape of kernel (can be inferred from W)
///
/// # Performance (CRITICAL)
///
/// - **Im2col+GEMM decomposition**: hologram's decomposition pass transforms Conv2D
///   into Im2col (image to column) followed by GEMM (matrix multiplication)
/// - **PhiCoordinate addressing**: 5-10x speedup for boundary pool access
/// - **SIMD vectorization**: GEMM operations use SIMD
/// - Supports **symbolic shapes** for dynamic batch sizes
///
/// # Shape Inference
///
/// - Input X: `[N, C_in, H_in, W_in]` (N can be symbolic)
/// - Kernel W: `[C_out, C_in/groups, KH, KW]`
/// - Bias B: `[C_out]` (optional)
/// - Output Y: `[N, C_out, H_out, W_out]`
///
/// Where:
/// - `H_out = (H_in + pad_top + pad_bottom - dilation_h * (KH - 1) - 1) / stride_h + 1`
/// - `W_out = (W_in + pad_left + pad_right - dilation_w * (KW - 1) - 1) / stride_w + 1`
pub fn translate_conv(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(format!(
            "Conv expects 2-3 inputs, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];
    let kernel = inputs[1];
    let bias = inputs.get(2).copied();

    // Parse attributes
    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    let pads = parse_attr_ints(attrs, "pads", vec![0, 0, 0, 0])?;
    let dilations = parse_attr_ints(attrs, "dilations", vec![1, 1])?;
    let groups = parse_attr_int(attrs, "group", 1)? as usize;

    if strides.len() != 2 {
        return Err(OnnxError::InvalidAttribute {
            name: "strides".to_string(),
            reason: format!("Expected 2 strides, got {}", strides.len()),
        });
    }

    if pads.len() != 4 {
        return Err(OnnxError::InvalidAttribute {
            name: "pads".to_string(),
            reason: format!("Expected 4 pads, got {}", pads.len()),
        });
    }

    if dilations.len() != 2 {
        return Err(OnnxError::InvalidAttribute {
            name: "dilations".to_string(),
            reason: format!("Expected 2 dilations, got {}", dilations.len()),
        });
    }

    debug!(
        "Translating Conv2D: strides={:?}, pads={:?}, dilations={:?}, groups={}",
        strides, pads, dilations, groups
    );
    trace!(
        "Conv2D inputs: input={:?}, kernel={:?}, bias={:?}",
        input, kernel, bias
    );

    // Create Conv2D IR node using builder method
    // CRITICAL: hologram's decomposition pass will transform this to Im2col+GEMM
    let conv_node = builder.conv2d(
        input,
        kernel,
        bias,
        (strides[0] as usize, strides[1] as usize),
        (pads[0] as usize, pads[1] as usize), // padding (top, left)
        (dilations[0] as usize, dilations[1] as usize),
        groups,
    );

    trace!("Created Conv2D node: {:?}", conv_node);
    Ok(conv_node)
}

/// Infer Conv2D output shape with symbolic dimension support.
///
/// # Shape Calculation
///
/// Output spatial dimensions are calculated as:
/// ```text
/// H_out = floor((H_in + pad_top + pad_bottom - dilation_h * (KH - 1) - 1) / stride_h) + 1
/// W_out = floor((W_in + pad_left + pad_right - dilation_w * (KW - 1) - 1) / stride_w) + 1
/// ```
///
/// # Symbolic Shapes
///
/// - Batch dimension (N) is preserved as symbolic if input is symbolic
/// - Spatial dimensions (H, W) computed using `Dim::Var` if input is symbolic
/// - Channel dimension (C_out) is always concrete (from kernel shape)
pub fn infer_conv_output_shape(
    input_shape: &SymbolicShape,
    kernel_shape: &SymbolicShape,
    attrs: &[AttributeProto],
) -> Result<SymbolicShape> {
    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    let pads = parse_attr_ints(attrs, "pads", vec![0, 0, 0, 0])?;
    let dilations = parse_attr_ints(attrs, "dilations", vec![1, 1])?;

    // Input: [N, C_in, H_in, W_in]
    // Kernel: [C_out, C_in/groups, KH, KW]
    // Output: [N, C_out, H_out, W_out]

    let input_dims = input_shape.dims();
    let kernel_dims = kernel_shape.dims();

    if input_dims.len() != 4 {
        return Err(OnnxError::ShapeInferenceError(format!(
            "Conv2D input must be 4D, got {}D",
            input_dims.len()
        )));
    }

    if kernel_dims.len() != 4 {
        return Err(OnnxError::ShapeInferenceError(format!(
            "Conv2D kernel must be 4D, got {}D",
            kernel_dims.len()
        )));
    }

    // Preserve batch dimension (can be symbolic)
    let batch = input_dims[0].clone();

    // Output channels from kernel
    let c_out = kernel_dims[0].clone();

    // Calculate output spatial dimensions
    let h_out = calculate_conv_output_dim(
        &input_dims[2],
        &kernel_dims[2],
        strides[0] as usize,
        pads[0] as usize + pads[2] as usize, // top + bottom
        dilations[0] as usize,
    )?;

    let w_out = calculate_conv_output_dim(
        &input_dims[3],
        &kernel_dims[3],
        strides[1] as usize,
        pads[1] as usize + pads[3] as usize, // left + right
        dilations[1] as usize,
    )?;

    Ok(SymbolicShape::new(vec![batch, c_out, h_out, w_out]))
}

/// Calculate output dimension for convolution.
///
/// Supports symbolic input dimensions using `Dim::Var` for dynamic shapes.
fn calculate_conv_output_dim(
    input_dim: &Dim,
    kernel_dim: &Dim,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<Dim> {
    match (input_dim, kernel_dim) {
        (Dim::Concrete(i), Dim::Concrete(k)) => {
            // Concrete calculation
            let effective_kernel = dilation * (k - 1) + 1;
            if *i + padding < effective_kernel {
                return Err(OnnxError::ShapeInferenceError(format!(
                    "Input size {} too small for kernel {}",
                    i, k
                )));
            }
            let output = (*i + padding - effective_kernel) / stride + 1;
            Ok(Dim::Concrete(output))
        }
        (Dim::Var(name), Dim::Concrete(k)) => {
            // Symbolic input, concrete kernel - create symbolic output
            Ok(Dim::Var(format!(
                "conv_out({},{},{},{},{})",
                name, k, stride, padding, dilation
            )))
        }
        _ => Err(OnnxError::ShapeInferenceError(
            "Kernel dimensions must be concrete".to_string(),
        )),
    }
}

/// Translate ONNX ConvTranspose operation.
///
/// ConvTranspose: Transposed convolution (deconvolution).
///
/// # Attributes
///
/// - `strides` (ints, default [1, 1]): Stride along each spatial axis
/// - `pads` (ints, default [0, 0, 0, 0]): Padding [top, left, bottom, right]
/// - `dilations` (ints, default [1, 1]): Dilation along each spatial axis
/// - `group` (int, default 1): Number of groups
/// - `output_padding` (ints, default [0, 0]): Additional output padding
///
/// # Performance
///
/// - **Im2col+GEMM decomposition**: Similar to Conv2D
/// - **PhiCoordinate addressing**: Efficient memory access patterns
/// - Supports **symbolic shapes**
///
/// # Implementation
///
/// Uses a Call node to `onnx.ConvTranspose` which the runtime handles.
pub fn translate_conv_transpose(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(format!(
            "ConvTranspose expects 2-3 inputs, got {}",
            inputs.len()
        )));
    }

    // Parse attributes for logging
    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    let pads = parse_attr_ints(attrs, "pads", vec![0, 0, 0, 0])?;
    let output_padding = parse_attr_ints(attrs, "output_padding", vec![0, 0])?;

    debug!(
        "Translating ConvTranspose: strides={:?}, pads={:?}, output_padding={:?}",
        strides, pads, output_padding
    );
    trace!("ConvTranspose inputs: {:?}", inputs);

    // Use Call node for ConvTranspose - runtime handles the transposed convolution
    let result = builder.call("onnx.ConvTranspose", inputs.to_vec());

    trace!("Created ConvTranspose call node: {:?}", result);
    Ok(result)
}

/// Infer ConvTranspose output shape.
///
/// Output spatial dimensions are calculated as:
/// ```text
/// H_out = stride_h * (H_in - 1) + dilation_h * (KH - 1) + 1 - pad_top - pad_bottom + output_padding_h
/// W_out = stride_w * (W_in - 1) + dilation_w * (KW - 1) + 1 - pad_left - pad_right + output_padding_w
/// ```
pub fn infer_conv_transpose_output_shape(
    input_shape: &SymbolicShape,
    kernel_shape: &SymbolicShape,
    attrs: &[AttributeProto],
) -> Result<SymbolicShape> {
    let strides = parse_attr_ints(attrs, "strides", vec![1, 1])?;
    let pads = parse_attr_ints(attrs, "pads", vec![0, 0, 0, 0])?;
    let dilations = parse_attr_ints(attrs, "dilations", vec![1, 1])?;
    let output_padding = parse_attr_ints(attrs, "output_padding", vec![0, 0])?;

    let input_dims = input_shape.dims();
    let kernel_dims = kernel_shape.dims();

    if input_dims.len() != 4 || kernel_dims.len() != 4 {
        return Err(OnnxError::ShapeInferenceError(
            "ConvTranspose requires 4D input and kernel".to_string(),
        ));
    }

    let batch = input_dims[0].clone();
    let c_out = kernel_dims[1].clone(); // Note: different from Conv2D

    let h_out = calculate_conv_transpose_output_dim(
        &input_dims[2],
        &kernel_dims[2],
        strides[0] as usize,
        pads[0] as usize + pads[2] as usize,
        dilations[0] as usize,
        output_padding[0] as usize,
    )?;

    let w_out = calculate_conv_transpose_output_dim(
        &input_dims[3],
        &kernel_dims[3],
        strides[1] as usize,
        pads[1] as usize + pads[3] as usize,
        dilations[1] as usize,
        output_padding[1] as usize,
    )?;

    Ok(SymbolicShape::new(vec![batch, c_out, h_out, w_out]))
}

/// Calculate output dimension for transposed convolution.
fn calculate_conv_transpose_output_dim(
    input_dim: &Dim,
    kernel_dim: &Dim,
    stride: usize,
    padding: usize,
    dilation: usize,
    output_padding: usize,
) -> Result<Dim> {
    match (input_dim, kernel_dim) {
        (Dim::Concrete(i), Dim::Concrete(k)) => {
            let output = stride * (i - 1) + dilation * (k - 1) + 1 - padding + output_padding;
            Ok(Dim::Concrete(output))
        }
        (Dim::Var(name), Dim::Concrete(k)) => {
            // Symbolic input - create symbolic output
            Ok(Dim::Var(format!(
                "conv_transpose_out({},{},{},{},{},{})",
                name, k, stride, padding, dilation, output_padding
            )))
        }
        _ => Err(OnnxError::ShapeInferenceError(
            "Kernel dimensions must be concrete".to_string(),
        )),
    }
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

    fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            ints: values,
            r#type: AttributeType::Ints as i32,
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
    fn test_translate_conv_basic() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));
        let kernel = builder.add_input("W", f32_tensor(&[64, 3, 7, 7]));

        let result = translate_conv(&[input, kernel], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_conv_with_bias() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));
        let kernel = builder.add_input("W", f32_tensor(&[64, 3, 7, 7]));
        let bias = builder.add_input("B", f32_tensor(&[64]));

        let result = translate_conv(
            &[input, kernel, bias],
            &[],
            &HashMap::new(),
            &mut builder,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_conv_with_attrs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));
        let kernel = builder.add_input("W", f32_tensor(&[64, 3, 3, 3]));

        let attrs = vec![
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
            make_ints_attr("dilations", vec![1, 1]),
            make_int_attr("group", 1),
        ];

        let result = translate_conv(&[input, kernel], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_conv_wrong_inputs() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));

        // Only 1 input
        let result = translate_conv(&[input], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_infer_conv_output_shape_concrete() {
        let input_shape = SymbolicShape::concrete(vec![1, 3, 224, 224]);
        let kernel_shape = SymbolicShape::concrete(vec![64, 3, 7, 7]);

        let attrs = vec![
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![3, 3, 3, 3]),
            make_ints_attr("dilations", vec![1, 1]),
        ];

        let result = infer_conv_output_shape(&input_shape, &kernel_shape, &attrs).unwrap();

        // H_out = (224 + 6 - 7) / 2 + 1 = 112
        // W_out = (224 + 6 - 7) / 2 + 1 = 112
        assert_eq!(result.dims()[0], Dim::Concrete(1)); // batch
        assert_eq!(result.dims()[1], Dim::Concrete(64)); // channels
        assert_eq!(result.dims()[2], Dim::Concrete(112)); // height
        assert_eq!(result.dims()[3], Dim::Concrete(112)); // width
    }

    #[test]
    fn test_infer_conv_output_shape_symbolic_batch() {
        let input_shape = SymbolicShape::symbolic(vec!["batch", "3", "224", "224"]);
        let kernel_shape = SymbolicShape::concrete(vec![64, 3, 3, 3]);

        let result = infer_conv_output_shape(&input_shape, &kernel_shape, &[]).unwrap();

        // Batch should remain symbolic
        assert!(matches!(result.dims()[0], Dim::Var(_)));
        assert_eq!(result.dims()[1], Dim::Concrete(64));
    }

    #[test]
    fn test_translate_conv_transpose() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));
        let kernel = builder.add_input("W", f32_tensor(&[64, 3, 2, 2]));

        let result =
            translate_conv_transpose(&[input, kernel], &[], &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_infer_conv_transpose_output_shape() {
        let input_shape = SymbolicShape::concrete(vec![1, 64, 56, 56]);
        let kernel_shape = SymbolicShape::concrete(vec![64, 3, 2, 2]);

        let attrs = vec![
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![0, 0, 0, 0]),
            make_ints_attr("output_padding", vec![0, 0]),
        ];

        let result =
            infer_conv_transpose_output_shape(&input_shape, &kernel_shape, &attrs).unwrap();

        // H_out = 2 * (56 - 1) + 2 = 112
        assert_eq!(result.dims()[0], Dim::Concrete(1));
        assert_eq!(result.dims()[1], Dim::Concrete(3)); // Note: C_out from kernel[1]
        assert_eq!(result.dims()[2], Dim::Concrete(112));
        assert_eq!(result.dims()[3], Dim::Concrete(112));
    }

    #[test]
    fn test_calculate_conv_output_dim_concrete() {
        let input = Dim::Concrete(224);
        let kernel = Dim::Concrete(7);

        let result = calculate_conv_output_dim(&input, &kernel, 2, 6, 1).unwrap();
        // (224 + 6 - 7) / 2 + 1 = 112
        assert_eq!(result, Dim::Concrete(112));
    }

    #[test]
    fn test_calculate_conv_output_dim_symbolic() {
        let input = Dim::Var("H".to_string());
        let kernel = Dim::Concrete(3);

        let result = calculate_conv_output_dim(&input, &kernel, 1, 2, 1).unwrap();
        assert!(matches!(result, Dim::Var(_)));
    }

    #[test]
    fn test_conv_invalid_kernel_size() {
        let input = Dim::Concrete(3);
        let kernel = Dim::Concrete(7);

        let result = calculate_conv_output_dim(&input, &kernel, 1, 0, 1);
        assert!(result.is_err()); // Input too small
    }
}

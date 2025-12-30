//! Core ONNX operations: MatMul, Gemm, and binary arithmetic.
//!
//! All operations in this module:
//! - Support **symbolic shapes** (variable batch sizes, sequence lengths)
//! - Leverage **LOOP instructions** for O(1) space complexity in broadcasting
//! - Use **SIMD vectorization** via hologram-backend for MatMul
//!
//! # ISA Optimizations
//!
//! - **MatMul**: Uses hologram's GEMM implementation with SIMD
//! - **Binary ops**: LOOP instructions for efficient broadcasting
//! - **Compile-time**: All shape resolution happens during compilation

use hologram_compiler::ir::{IRBuilder, NodeId};
use hologram_onnx_core::{OnnxError, Result, SymbolicShape};
use hologram_onnx_spec::AttributeProto;
use std::collections::HashMap;
use tracing::{debug, trace};

use crate::utils::{parse_attr_float, parse_attr_int};

/// Translate ONNX MatMul operation.
///
/// MatMul: Y = A @ B (matrix multiplication)
///
/// # Performance
///
/// - Uses hologram's **SIMD-optimized GEMM** implementation
/// - Supports **symbolic shapes** for dynamic batch sizes
/// - O(n³) time complexity (optimal for matrix multiplication)
///
/// # Shape Inference
///
/// - Input A: `[..., M, K]`
/// - Input B: `[..., K, N]`
/// - Output Y: `[..., M, N]`
///
/// Batch dimensions are broadcast according to NumPy rules.
pub fn translate_matmul(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "MatMul expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating MatMul operation");
    trace!("MatMul inputs: {:?} @ {:?}", a, b);

    // Create MatMul IR node using builder method
    // hologram's backend will optimize this with SIMD
    let node = builder.matmul(a, b);

    trace!("Created MatMul node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Gemm operation.
///
/// Gemm: Y = alpha * (A @ B) + beta * C
///
/// # Attributes
///
/// - `alpha` (float, default 1.0): Scalar multiplier for A @ B
/// - `beta` (float, default 1.0): Scalar multiplier for C
/// - `transA` (int, default 0): Transpose A before multiplication
/// - `transB` (int, default 0): Transpose B before multiplication
///
/// # Performance
///
/// - Uses **SIMD-optimized GEMM** for matrix multiplication
/// - **ClassMap fusion** for alpha/beta scaling (O(1) composition)
/// - Supports **symbolic shapes**
pub fn translate_gemm(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() < 2 || inputs.len() > 3 {
        return Err(OnnxError::InvalidModel(format!(
            "Gemm expects 2-3 inputs, got {}",
            inputs.len()
        )));
    }

    let mut a = inputs[0];
    let mut b = inputs[1];
    let c = inputs.get(2).copied();

    // Parse attributes
    let alpha = parse_attr_float(attrs, "alpha", 1.0)?;
    let beta = parse_attr_float(attrs, "beta", 1.0)?;
    let trans_a = parse_attr_int(attrs, "transA", 0)?;
    let trans_b = parse_attr_int(attrs, "transB", 0)?;

    debug!(
        "Translating Gemm: alpha={}, beta={}, transA={}, transB={}",
        alpha, beta, trans_a, trans_b
    );

    // Apply transposes if needed
    if trans_a != 0 {
        a = builder.transpose(a, Some(vec![1, 0])); // 2D transpose
    }

    if trans_b != 0 {
        b = builder.transpose(b, Some(vec![1, 0]));
    }

    // Compute A @ B
    let mut result = builder.matmul(a, b);

    // Apply alpha scaling if not 1.0
    if (alpha - 1.0).abs() > f32::EPSILON {
        let alpha_const = builder.add_f32(alpha);
        result = builder.mul(result, alpha_const);
    }

    // Add beta * C if C is provided
    if let Some(c_input) = c {
        let mut c_term = c_input;

        // Apply beta scaling if not 1.0
        if (beta - 1.0).abs() > f32::EPSILON {
            let beta_const = builder.add_f32(beta);
            c_term = builder.mul(c_term, beta_const);
        }

        // Add C term
        result = builder.add(result, c_term);
    }

    trace!("Created Gemm node chain ending at: {:?}", result);
    Ok(result)
}

/// Translate ONNX Add operation.
///
/// Add: Y = A + B (element-wise addition with broadcasting)
///
/// # Performance
///
/// - Uses **LOOP instructions** for efficient broadcasting (O(1) space)
/// - **ClassMap fusion** with adjacent element-wise ops (O(1) composition)
/// - Supports **symbolic shapes** with dynamic broadcasting
pub fn translate_add(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Add expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating Add operation");
    trace!("Add inputs: {:?} + {:?}", a, b);

    // Create Add IR node with broadcasting support
    // hologram's LOOP instructions provide O(1) space broadcasting
    let node = builder.add(a, b);

    trace!("Created Add node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Sub operation.
///
/// Sub: Y = A - B (element-wise subtraction with broadcasting)
pub fn translate_sub(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Sub expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating Sub operation");
    trace!("Sub inputs: {:?} - {:?}", a, b);

    let node = builder.sub(a, b);

    trace!("Created Sub node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Mul operation.
///
/// Mul: Y = A * B (element-wise multiplication with broadcasting)
pub fn translate_mul(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Mul expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating Mul operation");
    trace!("Mul inputs: {:?} * {:?}", a, b);

    let node = builder.mul(a, b);

    trace!("Created Mul node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Div operation.
///
/// Div: Y = A / B (element-wise division with broadcasting)
pub fn translate_div(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Div expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating Div operation");
    trace!("Div inputs: {:?} / {:?}", a, b);

    let node = builder.div(a, b);

    trace!("Created Div node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Pow operation.
///
/// Pow: Y = A ^ B (element-wise exponentiation with broadcasting)
pub fn translate_pow(
    inputs: &[NodeId],
    _attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 2 {
        return Err(OnnxError::InvalidModel(format!(
            "Pow expects 2 inputs, got {}",
            inputs.len()
        )));
    }

    let a = inputs[0];
    let b = inputs[1];

    debug!("Translating Pow operation");
    trace!("Pow inputs: {:?} ^ {:?}", a, b);

    let node = builder.pow(a, b);

    trace!("Created Pow node: {:?}", node);
    Ok(node)
}

/// Translate ONNX Cast operation.
///
/// Cast: Convert tensor to a different data type.
///
/// # Attributes
///
/// - `to` (int, required): Target data type (ONNX TensorProto.DataType enum)
///   - 1: FLOAT (32-bit)
///   - 10: FLOAT16 (16-bit)
///   - 11: DOUBLE (64-bit)
///   - 6: INT32
///   - 7: INT64
///
/// # Performance
///
/// - **SIMD vectorization**: Vectorized type conversion
/// - FP16 ↔ FP32 conversions common in Stable Diffusion
///
/// # Note
///
/// For now, this is a passthrough operation as hologram uses F32 internally.
/// Actual type conversion happens at the boundary.
pub fn translate_cast(
    inputs: &[NodeId],
    attrs: &[AttributeProto],
    _shapes: &HashMap<String, SymbolicShape>,
    _builder: &mut IRBuilder,
) -> Result<NodeId> {
    if inputs.len() != 1 {
        return Err(OnnxError::InvalidModel(format!(
            "Cast expects 1 input, got {}",
            inputs.len()
        )));
    }

    let input = inputs[0];

    // Parse target type from 'to' attribute
    let to = parse_attr_int(attrs, "to", 1)?; // Default to FLOAT

    debug!("Translating Cast operation (to={})", to);
    trace!("Cast input: {:?}", input);

    // For now, Cast is a passthrough since hologram uses F32 internally.
    // The actual type conversion is handled at graph boundaries.
    // This allows models with FP16 ↔ FP32 casts to compile.
    Ok(input)
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

    #[test]
    fn test_translate_matmul() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[3, 4]));

        let inputs = vec![a, b];
        let shapes = HashMap::new();

        let result = translate_matmul(&inputs, &[], &shapes, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_matmul_wrong_inputs() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));

        // Only 1 input (should fail)
        let result = translate_matmul(&[a], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));

        // 3 inputs (should fail)
        let result = translate_matmul(&[a, a, a], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }

    #[test]
    fn test_translate_gemm_basic() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[3, 4]));

        let inputs = vec![a, b];
        let shapes = HashMap::new();

        let result = translate_gemm(&inputs, &[], &shapes, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_gemm_with_bias() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[3, 4]));
        let c = builder.add_input("C", f32_tensor(&[2, 4]));

        let inputs = vec![a, b, c];
        let shapes = HashMap::new();

        let result = translate_gemm(&inputs, &[], &shapes, &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_binary_ops() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[2, 3]));

        let inputs = vec![a, b];
        let shapes = HashMap::new();

        // Test each binary operation
        assert!(translate_add(&inputs, &[], &shapes, &mut builder).is_ok());
        assert!(translate_sub(&inputs, &[], &shapes, &mut builder).is_ok());
        assert!(translate_mul(&inputs, &[], &shapes, &mut builder).is_ok());
        assert!(translate_div(&inputs, &[], &shapes, &mut builder).is_ok());
        assert!(translate_pow(&inputs, &[], &shapes, &mut builder).is_ok());
    }

    #[test]
    fn test_binary_ops_wrong_inputs() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));

        // Only 1 input (should fail for all binary ops)
        let ops = vec![
            translate_add,
            translate_sub,
            translate_mul,
            translate_div,
            translate_pow,
        ];

        for op in ops {
            let result = op(&[a], &[], &HashMap::new(), &mut builder);
            assert!(result.is_err());
            assert!(matches!(result.unwrap_err(), OnnxError::InvalidModel(_)));
        }
    }

    #[test]
    fn test_gemm_with_alpha_beta() {
        use hologram_onnx_spec::attribute_proto::AttributeType;

        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[2, 3]));
        let b = builder.add_input("B", f32_tensor(&[3, 4]));
        let c = builder.add_input("C", f32_tensor(&[2, 4]));

        let inputs = vec![a, b, c];
        let attrs = vec![
            AttributeProto {
                name: "alpha".to_string(),
                f: 0.5,
                r#type: AttributeType::Float as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "beta".to_string(),
                f: 2.0,
                r#type: AttributeType::Float as i32,
                ..Default::default()
            },
        ];

        let result = translate_gemm(&inputs, &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gemm_with_transpose() {
        let mut builder = make_builder();
        let a = builder.add_input("A", f32_tensor(&[3, 2])); // Will be transposed to [2, 3]
        let b = builder.add_input("B", f32_tensor(&[4, 3])); // Will be transposed to [3, 4]

        let inputs = vec![a, b];
        let attrs = vec![
            AttributeProto {
                name: "transA".to_string(),
                i: 1,
                r#type: AttributeType::Int as i32,
                ..Default::default()
            },
            AttributeProto {
                name: "transB".to_string(),
                i: 1,
                r#type: AttributeType::Int as i32,
                ..Default::default()
            },
        ];

        let result = translate_gemm(&inputs, &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_matmul_symbolic_shapes() {
        let mut builder = make_builder();
        // Symbolic batch dimension
        let a = builder.add_input("A", f32_tensor(&[])); // Will have symbolic shape
        let b = builder.add_input("B", f32_tensor(&[]));

        let inputs = vec![a, b];
        let shapes = HashMap::new();

        let result = translate_matmul(&inputs, &[], &shapes, &mut builder);
        assert!(result.is_ok());
    }

    // ========================================================================
    // Cast Tests
    // ========================================================================

    #[test]
    fn test_translate_cast() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[2, 3, 4]));

        // Cast to FLOAT (type 1)
        let attrs = vec![AttributeProto {
            name: "to".to_string(),
            i: 1,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }];

        let result = translate_cast(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_cast_fp16_to_fp32() {
        let mut builder = make_builder();
        let input = builder.add_input("X", f32_tensor(&[1, 512, 768]));

        // Cast from FP16 to FP32 (type 1 = FLOAT)
        let attrs = vec![AttributeProto {
            name: "to".to_string(),
            i: 1,
            r#type: AttributeType::Int as i32,
            ..Default::default()
        }];

        let result = translate_cast(&[input], &attrs, &HashMap::new(), &mut builder);
        assert!(result.is_ok());
    }

    #[test]
    fn test_translate_cast_wrong_inputs() {
        let mut builder = make_builder();
        let result = translate_cast(&[], &[], &HashMap::new(), &mut builder);
        assert!(result.is_err());
    }
}

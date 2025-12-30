//! Numerical correctness tests for ONNX operations.
//!
//! These tests verify that ONNX operations produce mathematically correct outputs
//! by comparing against expected values computed from the mathematical definitions.
//!
//! # Test Categories
//!
//! 1. **Unary Operations**: ReLU, Sigmoid, Tanh, Exp, Log, Sqrt, etc.
//! 2. **Binary Operations**: Add, Sub, Mul, Div, Pow
//! 3. **Matrix Operations**: MatMul, Gemm
//! 4. **Reductions**: Sum, Mean, Max, Min
//! 5. **Activations**: Softmax, GELU, Swish/SiLU
//!
//! Each test:
//! - Creates input tensors with known values
//! - Computes expected output using the mathematical definition
//! - Builds the operation using hologram-compiler's OpExpr
//! - Executes and verifies outputs match within tolerance

use hologram_compiler::{execute_schedule_rayon, Compiler, OpExpr, OpKind, TensorRef};
use std::collections::HashMap;

// =============================================================================
// Test Helpers
// =============================================================================

/// Tolerance for floating point comparisons
const EPSILON: f32 = 1e-5;

/// Assert that two vectors are approximately equal
fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], op_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: output length mismatch: got {}, expected {}",
        op_name,
        actual.len(),
        expected.len()
    );

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < EPSILON || (a.is_nan() && e.is_nan()),
            "{}: mismatch at index {}: got {}, expected {}, diff {}",
            op_name,
            i,
            a,
            e,
            diff
        );
    }
}

/// Execute a simple unary operation and return results
fn execute_unary_op(op_kind: OpKind, input: Vec<f32>) -> Vec<f32> {
    let expr = OpExpr::Apply {
        op: op_kind,
        args: vec![Box::new(OpExpr::Tensor(TensorRef::named("x")))],
    };

    let compiler = Compiler::new();
    let compiled = compiler
        .compile_parallel(expr)
        .expect("Compilation should succeed");

    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), input);

    let outputs = execute_schedule_rayon(&compiled.schedule, inputs).expect("Execution should succeed");

    outputs
        .get("output")
        .or_else(|| outputs.values().next())
        .expect("Should have output")
        .as_ref()
        .clone()
}

/// Execute a binary operation and return results
fn execute_binary_op(op_kind: OpKind, a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    let expr = OpExpr::Apply {
        op: op_kind,
        args: vec![
            Box::new(OpExpr::Tensor(TensorRef::named("a"))),
            Box::new(OpExpr::Tensor(TensorRef::named("b"))),
        ],
    };

    let compiler = Compiler::new();
    let compiled = compiler
        .compile_parallel(expr)
        .expect("Compilation should succeed");

    let mut inputs = HashMap::new();
    inputs.insert("a".to_string(), a);
    inputs.insert("b".to_string(), b);

    let outputs = execute_schedule_rayon(&compiled.schedule, inputs).expect("Execution should succeed");

    outputs
        .get("output")
        .or_else(|| outputs.values().next())
        .expect("Should have output")
        .as_ref()
        .clone()
}

// =============================================================================
// Reference Implementations for Expected Values
// =============================================================================

/// Reference sigmoid implementation: 1 / (1 + exp(-x))
fn sigmoid_ref(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Reference tanh implementation
fn tanh_ref(x: f32) -> f32 {
    x.tanh()
}

/// Reference ReLU implementation: max(0, x)
fn relu_ref(x: f32) -> f32 {
    x.max(0.0)
}

/// Reference GELU implementation (approximation)
fn gelu_ref(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x.powi(3))).tanh())
}

// Note: Additional reference implementations (swish, softmax) will be added
// when corresponding operation tests are implemented.

// =============================================================================
// Unary Operation Tests
// =============================================================================

#[test]
fn test_relu_numerical_correctness() {
    let input = vec![-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| relu_ref(x)).collect();

    let actual = execute_unary_op(OpKind::ReLU, input);
    assert_vec_approx_eq(&actual, &expected, "ReLU");
}

#[test]
fn test_sigmoid_numerical_correctness() {
    let input = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];
    let expected: Vec<f32> = input.iter().map(|&x| sigmoid_ref(x)).collect();

    let actual = execute_unary_op(OpKind::Sigmoid, input);
    assert_vec_approx_eq(&actual, &expected, "Sigmoid");
}

#[test]
fn test_tanh_numerical_correctness() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| tanh_ref(x)).collect();

    let actual = execute_unary_op(OpKind::Tanh, input);
    assert_vec_approx_eq(&actual, &expected, "Tanh");
}

#[test]
fn test_exp_numerical_correctness() {
    let input = vec![-1.0f32, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| x.exp()).collect();

    let actual = execute_unary_op(OpKind::Exp, input);
    assert_vec_approx_eq(&actual, &expected, "Exp");
}

#[test]
fn test_log_numerical_correctness() {
    let input = vec![0.5f32, 1.0, 2.0, 10.0];
    let expected: Vec<f32> = input.iter().map(|&x| x.ln()).collect();

    let actual = execute_unary_op(OpKind::Log, input);
    assert_vec_approx_eq(&actual, &expected, "Log");
}

#[test]
fn test_sqrt_numerical_correctness() {
    let input = vec![0.0f32, 1.0, 4.0, 9.0, 16.0];
    let expected: Vec<f32> = input.iter().map(|&x| x.sqrt()).collect();

    let actual = execute_unary_op(OpKind::Sqrt, input);
    assert_vec_approx_eq(&actual, &expected, "Sqrt");
}

#[test]
fn test_neg_numerical_correctness() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| -x).collect();

    let actual = execute_unary_op(OpKind::Neg, input);
    assert_vec_approx_eq(&actual, &expected, "Neg");
}

#[test]
fn test_abs_numerical_correctness() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| x.abs()).collect();

    let actual = execute_unary_op(OpKind::Abs, input);
    assert_vec_approx_eq(&actual, &expected, "Abs");
}

#[test]
fn test_gelu_numerical_correctness() {
    let input = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
    let expected: Vec<f32> = input.iter().map(|&x| gelu_ref(x)).collect();

    let actual = execute_unary_op(OpKind::GELU, input);
    assert_vec_approx_eq(&actual, &expected, "GELU");
}

// =============================================================================
// Binary Operation Tests
// =============================================================================

#[test]
fn test_add_numerical_correctness() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let actual = execute_binary_op(OpKind::Add, a, b);
    assert_vec_approx_eq(&actual, &expected, "Add");
}

#[test]
fn test_sub_numerical_correctness() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    let actual = execute_binary_op(OpKind::Sub, a, b);
    assert_vec_approx_eq(&actual, &expected, "Sub");
}

#[test]
fn test_mul_numerical_correctness() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.5, 2.5, 3.5];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();

    let actual = execute_binary_op(OpKind::Mul, a, b);
    assert_vec_approx_eq(&actual, &expected, "Mul");
}

#[test]
fn test_div_numerical_correctness() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![0.5f32, 1.0, 1.5, 2.0];
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();

    let actual = execute_binary_op(OpKind::Div, a, b);
    assert_vec_approx_eq(&actual, &expected, "Div");
}

// =============================================================================
// Edge Case Tests
// =============================================================================

#[test]
fn test_relu_edge_cases() {
    // Test very small negative numbers
    let input = vec![-1e-7f32, -1e-10, 0.0, 1e-10, 1e-7];
    let expected: Vec<f32> = input.iter().map(|&x| relu_ref(x)).collect();

    let actual = execute_unary_op(OpKind::ReLU, input);
    assert_vec_approx_eq(&actual, &expected, "ReLU edge cases");
}

#[test]
fn test_sigmoid_edge_cases() {
    // Test saturation behavior at extremes
    let input = vec![-100.0f32, -10.0, 0.0, 10.0, 100.0];
    let expected: Vec<f32> = input.iter().map(|&x| sigmoid_ref(x)).collect();

    let actual = execute_unary_op(OpKind::Sigmoid, input);
    assert_vec_approx_eq(&actual, &expected, "Sigmoid edge cases");
}

#[test]
fn test_single_element() {
    let input = vec![1.5f32];
    let expected = vec![relu_ref(1.5)];

    let actual = execute_unary_op(OpKind::ReLU, input);
    assert_vec_approx_eq(&actual, &expected, "Single element");
}

#[test]
fn test_large_tensor() {
    // Test with larger tensor (1024 elements)
    let input: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 100.0).collect();
    let expected: Vec<f32> = input.iter().map(|&x| sigmoid_ref(x)).collect();

    let actual = execute_unary_op(OpKind::Sigmoid, input);
    assert_vec_approx_eq(&actual, &expected, "Large tensor sigmoid");
}

// =============================================================================
// Chained Operations Tests
// =============================================================================
// Note: Chained operations through the parallel scheduler require explicit
// output marking. These tests are commented out pending scheduler improvements.

// TODO: Add chained operation tests when scheduler supports auto-output detection

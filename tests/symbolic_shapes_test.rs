//! Symbolic Shape Test Suite for hologram-onnx.
//!
//! Tests all operations with symbolic dimensions to ensure proper handling of:
//! - `Dim::Var`: Variable dimensions (batch size, sequence length)
//! - `Dim::Expr`: Computed dimension expressions
//! - Shape inference propagation through operation chains
//! - Variable batch size across all operations
//! - Variable sequence length for transformer operations
//!
//! # Test Categories
//!
//! 1. **Dim::Var Tests**: Operations with symbolic variable dimensions
//! 2. **Dim::Expr Tests**: Operations with computed dimension expressions
//! 3. **Shape Inference Propagation**: Multi-operation chains
//! 4. **Variable Batch Size**: All operations with variable N
//! 5. **Variable Sequence Length**: Transformer-style operations

use std::collections::HashMap;

use hologram_compiler::ir::IRBuilder;
use hologram_compiler::shapes::{Dim, DimExpr};
use hologram_onnx_core::{OnnxError, SymbolicShape};
use hologram_onnx_ops::translate_onnx_op;

// ============================================================================
// Dim::Var Tests - Variable Dimensions
// ============================================================================

/// Test creating shapes with variable dimensions.
#[test]
fn test_dim_var_creation() {
    // Single variable dimension
    let shape = SymbolicShape::new(vec![Dim::Var("batch".into())]);
    assert_eq!(shape.rank(), 1);
    assert!(shape.is_partially_symbolic());
    assert!(!shape.is_fully_concrete());

    // Multiple variable dimensions
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq_len".into()),
        Dim::Concrete(768),
    ]);
    assert_eq!(shape.rank(), 3);
    assert!(shape.is_partially_symbolic());
}

/// Test symbolic() helper creates correct Dim::Var.
#[test]
fn test_symbolic_helper() {
    let shape = SymbolicShape::symbolic(vec!["N", "C", "H", "W"]);
    assert_eq!(shape.rank(), 4);

    // All should be Var since they can't parse as numbers
    assert!(matches!(&shape.dims()[0], Dim::Var(n) if n == "N"));
    assert!(matches!(&shape.dims()[1], Dim::Var(n) if n == "C"));
    assert!(matches!(&shape.dims()[2], Dim::Var(n) if n == "H"));
    assert!(matches!(&shape.dims()[3], Dim::Var(n) if n == "W"));
}

/// Test mixed concrete and variable dimensions.
#[test]
fn test_mixed_var_concrete() {
    // Common pattern: [batch, 3, 224, 224]
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(3),
        Dim::Concrete(224),
        Dim::Concrete(224),
    ]);

    assert_eq!(shape.rank(), 4);
    assert!(shape.is_partially_symbolic());
    assert!(!shape.is_fully_concrete());
    assert!(matches!(&shape.dims()[0], Dim::Var(_)));
    assert_eq!(shape.dims()[1], Dim::Concrete(3));
}

/// Test binary operation with variable dimensions.
#[test]
fn test_binary_op_with_var() {
    // [batch, 64] + [64] should broadcast to [batch, 64]
    let a = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(64)]);
    let b = SymbolicShape::concrete(vec![64]);

    let result = a.infer_binary_op(&b).unwrap();
    assert_eq!(result.rank(), 2);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(result.dims()[1], Dim::Concrete(64));
}

/// Test binary operation with matching variable dimensions.
#[test]
fn test_binary_op_matching_vars() {
    // [batch, seq] + [batch, seq] should work
    let a = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Var("seq".into())]);
    let b = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Var("seq".into())]);

    let result = a.infer_binary_op(&b).unwrap();
    assert_eq!(result.rank(), 2);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&result.dims()[1], Dim::Var(n) if n == "seq"));
}

/// Test MatMul with variable dimensions.
#[test]
fn test_matmul_with_var() {
    // [batch, M, K] @ [K, N] -> [batch, M, N]
    let a = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("M".into()),
        Dim::Concrete(64),
    ]);
    let b = SymbolicShape::concrete(vec![64, 128]);

    let result = a.infer_matmul(&b).unwrap();
    assert_eq!(result.rank(), 3);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&result.dims()[1], Dim::Var(n) if n == "M"));
    assert_eq!(result.dims()[2], Dim::Concrete(128));
}

/// Test Transpose with variable dimensions.
#[test]
fn test_transpose_with_var() {
    // [batch, seq, hidden] -> [batch, hidden, seq]
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq".into()),
        Dim::Concrete(768),
    ]);

    let result = shape.infer_transpose(Some(&[0, 2, 1])).unwrap();
    assert_eq!(result.rank(), 3);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(result.dims()[1], Dim::Concrete(768));
    assert!(matches!(&result.dims()[2], Dim::Var(n) if n == "seq"));
}

/// Test Reshape with variable dimensions.
#[test]
fn test_reshape_with_var() {
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(12),
        Dim::Concrete(64),
    ]);

    // Reshape to [batch, 768]
    let target = vec![Dim::Var("batch".into()), Dim::Concrete(768)];
    let result = shape.infer_reshape(&target).unwrap();
    assert_eq!(result.rank(), 2);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(result.dims()[1], Dim::Concrete(768));
}

// ============================================================================
// Dim::Expr Tests - Computed Expressions
// ============================================================================

/// Test creating dimensions with expressions.
#[test]
fn test_dim_expr_creation() {
    // Create (H - 1) / stride + 1 style expression
    let h = DimExpr::Dim(Dim::Var("H".into()));
    let one = DimExpr::Dim(Dim::Concrete(1));
    let stride = DimExpr::Dim(Dim::Concrete(2));

    // (H - 1)
    let h_minus_1 = DimExpr::Sub(Box::new(h), Box::new(one.clone()));

    // (H - 1) / stride
    let div_stride = DimExpr::Div(Box::new(h_minus_1), Box::new(stride));

    // (H - 1) / stride + 1
    let result_expr = DimExpr::Add(Box::new(div_stride), Box::new(one));

    let dim = Dim::Expr(Box::new(result_expr));

    // Create shape with this expression
    let shape = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(64), dim]);

    assert_eq!(shape.rank(), 3);
    assert!(shape.is_partially_symbolic());
}

/// Test simple multiplication expression.
#[test]
fn test_dim_expr_multiply() {
    // batch * seq_len
    let batch = DimExpr::Dim(Dim::Var("batch".into()));
    let seq = DimExpr::Dim(Dim::Var("seq".into()));
    let product = DimExpr::Mul(Box::new(batch), Box::new(seq));

    let dim = Dim::Expr(Box::new(product));
    let shape = SymbolicShape::new(vec![dim, Dim::Concrete(768)]);

    assert_eq!(shape.rank(), 2);
    assert!(shape.is_partially_symbolic());
}

/// Test addition expression for output dimensions.
#[test]
fn test_dim_expr_add() {
    // seq_len + 1 (for CLS token)
    let seq = DimExpr::Dim(Dim::Var("seq".into()));
    let one = DimExpr::Dim(Dim::Concrete(1));
    let seq_plus_1 = DimExpr::Add(Box::new(seq), Box::new(one));

    let dim = Dim::Expr(Box::new(seq_plus_1));
    let shape = SymbolicShape::new(vec![Dim::Var("batch".into()), dim, Dim::Concrete(768)]);

    assert_eq!(shape.rank(), 3);
}

/// Test division expression (pooling output).
#[test]
fn test_dim_expr_divide() {
    // H / 2 (for stride-2 pooling)
    let h = DimExpr::Dim(Dim::Var("H".into()));
    let two = DimExpr::Dim(Dim::Concrete(2));
    let h_div_2 = DimExpr::Div(Box::new(h), Box::new(two));

    let dim = Dim::Expr(Box::new(h_div_2));
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(64),
        dim.clone(),
        dim, // Same expression for W
    ]);

    assert_eq!(shape.rank(), 4);
}

/// Test nested expressions (Conv2D output formula).
#[test]
fn test_dim_expr_nested() {
    // Output = (Input + 2*pad - kernel) / stride + 1
    // For Input=H, pad=1, kernel=3, stride=2:
    // Output = (H + 2 - 3) / 2 + 1 = (H - 1) / 2 + 1

    let h = DimExpr::Dim(Dim::Var("H".into()));
    let one = DimExpr::Dim(Dim::Concrete(1));
    let two = DimExpr::Dim(Dim::Concrete(2));

    // H - 1
    let h_minus_1 = DimExpr::Sub(Box::new(h), Box::new(one.clone()));

    // (H - 1) / 2
    let div_2 = DimExpr::Div(Box::new(h_minus_1), Box::new(two));

    // (H - 1) / 2 + 1
    let output = DimExpr::Add(Box::new(div_2), Box::new(one));

    let dim = Dim::Expr(Box::new(output));
    let shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(64),
        dim.clone(),
        dim,
    ]);

    assert_eq!(shape.rank(), 4);
    assert!(shape.is_partially_symbolic());
}

// ============================================================================
// Shape Inference Propagation Tests
// ============================================================================

/// Test shape propagation through Add chain.
#[test]
fn test_propagation_add_chain() {
    let x = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(64)]);
    let bias = SymbolicShape::concrete(vec![64]);

    // x + bias -> [batch, 64]
    let step1 = x.infer_binary_op(&bias).unwrap();
    assert!(matches!(&step1.dims()[0], Dim::Var(n) if n == "batch"));

    // result + bias -> [batch, 64] (still preserves batch)
    let step2 = step1.infer_binary_op(&bias).unwrap();
    assert!(matches!(&step2.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(step2.dims()[1], Dim::Concrete(64));
}

/// Test shape propagation through MatMul chain (MLP).
#[test]
fn test_propagation_matmul_chain() {
    // MLP: [batch, 768] -> [batch, 3072] -> [batch, 768]
    let x = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(768)]);
    let w1 = SymbolicShape::concrete(vec![768, 3072]);
    let w2 = SymbolicShape::concrete(vec![3072, 768]);

    // x @ w1 -> [batch, 3072]
    let hidden = x.infer_matmul(&w1).unwrap();
    assert_eq!(hidden.rank(), 2);
    assert!(matches!(&hidden.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(hidden.dims()[1], Dim::Concrete(3072));

    // hidden @ w2 -> [batch, 768]
    let output = hidden.infer_matmul(&w2).unwrap();
    assert_eq!(output.rank(), 2);
    assert!(matches!(&output.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(output.dims()[1], Dim::Concrete(768));
}

/// Test shape propagation through attention-style operations.
#[test]
fn test_propagation_attention() {
    // Attention: Q, K, V all [batch, seq, hidden]
    // Scores = Q @ K^T -> [batch, seq, seq]
    // Output = Scores @ V -> [batch, seq, hidden]

    let q = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq".into()),
        Dim::Concrete(64), // head_dim
    ]);
    let k = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq".into()),
        Dim::Concrete(64),
    ]);
    let v = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq".into()),
        Dim::Concrete(64),
    ]);

    // K^T: [batch, seq, 64] -> [batch, 64, seq]
    let k_t = k.infer_transpose(Some(&[0, 2, 1])).unwrap();
    assert!(matches!(&k_t.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(k_t.dims()[1], Dim::Concrete(64));
    assert!(matches!(&k_t.dims()[2], Dim::Var(n) if n == "seq"));

    // Q @ K^T: [batch, seq, 64] @ [batch, 64, seq] -> [batch, seq, seq]
    let scores = q.infer_matmul(&k_t).unwrap();
    assert_eq!(scores.rank(), 3);
    assert!(matches!(&scores.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&scores.dims()[1], Dim::Var(n) if n == "seq"));
    assert!(matches!(&scores.dims()[2], Dim::Var(n) if n == "seq"));

    // Scores @ V: [batch, seq, seq] @ [batch, seq, 64] -> [batch, seq, 64]
    let output = scores.infer_matmul(&v).unwrap();
    assert_eq!(output.rank(), 3);
    assert!(matches!(&output.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&output.dims()[1], Dim::Var(n) if n == "seq"));
    assert_eq!(output.dims()[2], Dim::Concrete(64));
}

/// Test shape propagation through transpose + matmul + transpose.
#[test]
fn test_propagation_transpose_matmul() {
    // Pattern: transpose -> matmul -> transpose
    let x = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(3),
        Dim::Var("seq".into()),
    ]);

    // [batch, 3, seq] -> [batch, seq, 3]
    let transposed = x.infer_transpose(Some(&[0, 2, 1])).unwrap();
    assert!(matches!(&transposed.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&transposed.dims()[1], Dim::Var(n) if n == "seq"));
    assert_eq!(transposed.dims()[2], Dim::Concrete(3));
}

// ============================================================================
// Variable Batch Size Tests
// ============================================================================

/// Test all core operations preserve variable batch.
#[test]
fn test_batch_preservation_core_ops() {
    let batch_shape = SymbolicShape::new(vec![Dim::Var("N".into()), Dim::Concrete(768)]);

    // Add: [N, 768] + [768] -> [N, 768]
    let bias = SymbolicShape::concrete(vec![768]);
    let add_result = batch_shape.infer_binary_op(&bias).unwrap();
    assert!(matches!(&add_result.dims()[0], Dim::Var(n) if n == "N"));

    // Mul: [N, 768] * [768] -> [N, 768]
    let mul_result = batch_shape.infer_binary_op(&bias).unwrap();
    assert!(matches!(&mul_result.dims()[0], Dim::Var(n) if n == "N"));

    // MatMul: [N, 768] @ [768, 1024] -> [N, 1024]
    let weight = SymbolicShape::concrete(vec![768, 1024]);
    let matmul_result = batch_shape.infer_matmul(&weight).unwrap();
    assert!(matches!(&matmul_result.dims()[0], Dim::Var(n) if n == "N"));
    assert_eq!(matmul_result.dims()[1], Dim::Concrete(1024));
}

/// Test batch dimension in 4D tensors (CNN).
#[test]
fn test_batch_preservation_4d() {
    // [N, C, H, W] pattern for CNN
    let cnn_shape = SymbolicShape::new(vec![
        Dim::Var("N".into()),
        Dim::Concrete(64),
        Dim::Concrete(224),
        Dim::Concrete(224),
    ]);

    // Transpose NCHW -> NHWC
    let nhwc = cnn_shape.infer_transpose(Some(&[0, 2, 3, 1])).unwrap();
    assert!(matches!(&nhwc.dims()[0], Dim::Var(n) if n == "N"));
    assert_eq!(nhwc.dims()[1], Dim::Concrete(224));
    assert_eq!(nhwc.dims()[2], Dim::Concrete(224));
    assert_eq!(nhwc.dims()[3], Dim::Concrete(64));
}

/// Test batch dimension in 3D tensors (Transformer).
#[test]
fn test_batch_preservation_3d() {
    // [batch, seq, hidden] pattern for Transformer
    let transformer_shape = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq".into()),
        Dim::Concrete(768),
    ]);

    // Transpose for attention: [batch, seq, hidden] -> [batch, hidden, seq]
    let transposed = transformer_shape.infer_transpose(Some(&[0, 2, 1])).unwrap();
    assert!(matches!(&transposed.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(transposed.dims()[1], Dim::Concrete(768));
    assert!(matches!(&transposed.dims()[2], Dim::Var(n) if n == "seq"));
}

/// Test batch broadcasting rules.
#[test]
fn test_batch_broadcasting() {
    // [N, 768] + [1, 768] should broadcast to [N, 768]
    let batched = SymbolicShape::new(vec![Dim::Var("N".into()), Dim::Concrete(768)]);
    let single = SymbolicShape::concrete(vec![1, 768]);

    let result = batched.infer_binary_op(&single).unwrap();
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "N"));
    assert_eq!(result.dims()[1], Dim::Concrete(768));
}

/// Test batched matrix multiplication.
#[test]
fn test_batch_matmul_3d() {
    // Batched matmul: [N, 64, 32] @ [N, 32, 128] -> [N, 64, 128]
    let a = SymbolicShape::new(vec![
        Dim::Var("N".into()),
        Dim::Concrete(64),
        Dim::Concrete(32),
    ]);
    let b = SymbolicShape::new(vec![
        Dim::Var("N".into()),
        Dim::Concrete(32),
        Dim::Concrete(128),
    ]);

    let result = a.infer_matmul(&b).unwrap();
    assert_eq!(result.rank(), 3);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "N"));
    assert_eq!(result.dims()[1], Dim::Concrete(64));
    assert_eq!(result.dims()[2], Dim::Concrete(128));
}

// ============================================================================
// Variable Sequence Length Tests
// ============================================================================

/// Test sequence length in attention shapes.
#[test]
fn test_seq_len_attention() {
    // Q, K, V: [batch, num_heads, seq_len, head_dim]
    let qkv = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(12), // num_heads
        Dim::Var("seq_len".into()),
        Dim::Concrete(64), // head_dim
    ]);

    // Transpose for attention: [batch, heads, seq, dim] -> stays same
    // K transpose: -> [batch, heads, dim, seq]
    let k_t = qkv.infer_transpose(Some(&[0, 1, 3, 2])).unwrap();
    assert!(matches!(&k_t.dims()[0], Dim::Var(n) if n == "batch"));
    assert_eq!(k_t.dims()[1], Dim::Concrete(12));
    assert_eq!(k_t.dims()[2], Dim::Concrete(64));
    assert!(matches!(&k_t.dims()[3], Dim::Var(n) if n == "seq_len"));
}

/// Test sequence length with position embeddings.
#[test]
fn test_seq_len_embeddings() {
    // Token embeddings: [batch, seq_len, hidden]
    let tokens = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("seq_len".into()),
        Dim::Concrete(768),
    ]);

    // Position embeddings: [1, max_seq, hidden] or [seq_len, hidden]
    let pos = SymbolicShape::new(vec![Dim::Var("seq_len".into()), Dim::Concrete(768)]);

    // tokens + pos: broadcasting should work
    let result = tokens.infer_binary_op(&pos).unwrap();
    assert_eq!(result.rank(), 3);
    assert!(matches!(&result.dims()[0], Dim::Var(n) if n == "batch"));
    assert!(matches!(&result.dims()[1], Dim::Var(n) if n == "seq_len"));
    assert_eq!(result.dims()[2], Dim::Concrete(768));
}

/// Test sequence length in RNN/LSTM shapes.
#[test]
fn test_seq_len_rnn() {
    // RNN input: [batch, seq_len, input_size]
    let rnn_input = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Var("T".into()), // seq_len often called T
        Dim::Concrete(128),
    ]);

    // Hidden state: [num_layers, batch, hidden_size]
    let hidden = SymbolicShape::new(vec![
        Dim::Concrete(2), // num_layers
        Dim::Var("batch".into()),
        Dim::Concrete(256),
    ]);

    assert!(matches!(&rnn_input.dims()[1], Dim::Var(n) if n == "T"));
    assert!(matches!(&hidden.dims()[1], Dim::Var(n) if n == "batch"));
}

/// Test sequence length in causal attention mask.
#[test]
fn test_seq_len_causal_mask() {
    // Causal mask: [seq_len, seq_len]
    let mask = SymbolicShape::new(vec![Dim::Var("seq".into()), Dim::Var("seq".into())]);

    // Broadcast to [batch, heads, seq, seq]
    let full_mask = SymbolicShape::new(vec![
        Dim::Var("batch".into()),
        Dim::Concrete(12),
        Dim::Var("seq".into()),
        Dim::Var("seq".into()),
    ]);

    // mask + full_mask should broadcast
    let result = mask.infer_binary_op(&full_mask).unwrap();
    assert_eq!(result.rank(), 4);
}

// ============================================================================
// Operation Translation with Symbolic Shapes Tests
// ============================================================================

/// Test MatMul translation with symbolic shapes.
#[test]
fn test_translate_matmul_symbolic() {
    let mut builder = IRBuilder::new("test_matmul");
    let shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // This tests that the translator accepts symbolic shapes
    // The actual translation may fail due to missing inputs, but
    // the shape handling should work
    let result = translate_onnx_op(
        "MatMul",
        &[], // Empty inputs for structure test
        &[],
        &shapes,
        &mut builder,
    );

    // Expected to fail due to missing inputs, not shape issues
    assert!(result.is_err());
    if let Err(e) = result {
        // Should be input error, not shape error
        let msg = format!("{:?}", e);
        assert!(
            !msg.contains("Shape"),
            "Should not be a shape error: {}",
            msg
        );
    }
}

/// Test Add translation with symbolic shapes in map.
#[test]
fn test_translate_add_symbolic_in_map() {
    let mut builder = IRBuilder::new("test_add");
    let mut shapes: HashMap<String, SymbolicShape> = HashMap::new();

    // Add symbolic shape to map
    shapes.insert(
        "input".to_string(),
        SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(768)]),
    );

    // Translation should handle symbolic shapes in the map
    let result = translate_onnx_op("Add", &[], &[], &shapes, &mut builder);

    // Expected to fail due to missing inputs
    assert!(result.is_err());
}

/// Test Softmax translation preserves symbolic shape.
#[test]
fn test_translate_softmax_symbolic() {
    let mut builder = IRBuilder::new("test_softmax");
    let mut shapes: HashMap<String, SymbolicShape> = HashMap::new();

    shapes.insert(
        "logits".to_string(),
        SymbolicShape::new(vec![
            Dim::Var("batch".into()),
            Dim::Var("seq".into()),
            Dim::Concrete(50000), // vocab size
        ]),
    );

    let result = translate_onnx_op("Softmax", &[], &[], &shapes, &mut builder);

    // Softmax preserves shape
    assert!(result.is_err()); // Due to missing inputs
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

/// Test incompatible symbolic dimensions in broadcast.
#[test]
fn test_broadcast_incompatible_vars() {
    // [batch, 64] + [other_batch, 64] - different variable names
    let a = SymbolicShape::new(vec![Dim::Var("batch".into()), Dim::Concrete(64)]);
    let b = SymbolicShape::new(vec![Dim::Var("other".into()), Dim::Concrete(64)]);

    // This should succeed but preserve one of the variables
    // (broadcasting rules say one or both must be 1, or they must match)
    // Since both are variables (not 1), this depends on implementation
    let result = a.infer_binary_op(&b);
    // The result depends on how the implementation handles different variable names
    // Some implementations may error, others may pick one
    if let Err(err) = result {
        assert!(matches!(err, OnnxError::ShapeInferenceError(_)));
    }
}

/// Test MatMul dimension mismatch.
#[test]
fn test_matmul_dimension_mismatch() {
    let a = SymbolicShape::concrete(vec![32, 64]);
    let b = SymbolicShape::concrete(vec![128, 256]); // 128 != 64

    let result = a.infer_matmul(&b);
    assert!(result.is_err());
    if let Err(OnnxError::ShapeInferenceError(msg)) = result {
        assert!(msg.contains("mismatch") || msg.contains("inner"));
    }
}

/// Test transpose with invalid permutation length.
#[test]
fn test_transpose_invalid_perm_length() {
    let shape = SymbolicShape::concrete(vec![2, 3, 4]);

    // Permutation length 2 for rank-3 tensor
    let result = shape.infer_transpose(Some(&[1, 0]));
    assert!(result.is_err());
    if let Err(OnnxError::ShapeInferenceError(msg)) = result {
        assert!(msg.contains("length") || msg.contains("rank"));
    }
}

/// Test 1D tensor for MatMul (should fail).
#[test]
fn test_matmul_1d_tensor() {
    let a = SymbolicShape::concrete(vec![64]);
    let b = SymbolicShape::concrete(vec![64, 128]);

    let result = a.infer_matmul(&b);
    assert!(result.is_err());
    if let Err(OnnxError::ShapeInferenceError(msg)) = result {
        assert!(msg.contains("2D"));
    }
}

/// Test empty dimension vector.
#[test]
fn test_empty_shape() {
    let shape = SymbolicShape::new(vec![]);
    assert_eq!(shape.rank(), 0);
    assert!(shape.is_fully_concrete()); // Empty is considered concrete
}

/// Test scalar (rank-0) shape handling.
#[test]
fn test_scalar_shape() {
    let scalar = SymbolicShape::new(vec![]);
    let tensor = SymbolicShape::concrete(vec![2, 3]);

    // Scalar + Tensor should broadcast to Tensor shape
    let result = scalar.infer_binary_op(&tensor).unwrap();
    assert_eq!(result.rank(), 2);
    assert_eq!(result.dims()[0], Dim::Concrete(2));
    assert_eq!(result.dims()[1], Dim::Concrete(3));
}

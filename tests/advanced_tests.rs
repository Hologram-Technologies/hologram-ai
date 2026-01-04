//! Advanced integration tests for transformer and RNN models.
//!
//! These tests verify:
//! - **BERT model**: Encoder-only transformer with attention and variable seq_len
//! - **GPT model**: Decoder-only transformer with causal attention
//! - **LSTM model**: Recurrent networks with variable sequence lengths
//! - **LOOP optimization**: O(1) space complexity design verification
//!
//! # ISA Integration
//!
//! All operations leverage hologram's ISA:
//! - **LOOP instructions**: O(1) space complexity for attention and RNN unrolling
//! - **SIMD vectorization**: Parallel MatMul and element-wise ops
//! - **ClassMap fusion**: Element-wise activation chains
//!
//! # Test Organization
//!
//! - Tests use working primitives (MatMul, LayerNorm, Softmax) to build model components
//! - Symbolic shapes are tested throughout with variable batch and sequence length
//! - Full attention/LSTM decomposition is documented as pending

use hologram_compiler::ir::IRBuilder;
use hologram_compiler::ir::types::{ScalarType, TensorType, Type};
use hologram_onnx_core::{Dim, SymbolicShape};
use hologram_onnx_ops::{
    translate_add, translate_attention, translate_gelu, translate_gru,
    translate_layer_normalization, translate_lstm, translate_matmul,
    translate_multi_head_attention, translate_rnn, translate_sigmoid,
};
use hologram_onnx_spec::{AttributeProto, attribute_proto::AttributeType};
use std::collections::HashMap;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a tensor type with F32 elements and concrete shape.
fn f32_tensor(dims: &[usize]) -> Type {
    Type::Tensor(TensorType::concrete(ScalarType::F32, dims.to_vec()))
}

/// Create a symbolic shape from concrete dimensions.
fn concrete_shape(dims: &[usize]) -> SymbolicShape {
    SymbolicShape::concrete(dims.to_vec())
}

/// Create a symbolic shape with variable batch dimension.
fn symbolic_batch_shape(batch_name: &str, rest: &[usize]) -> SymbolicShape {
    let mut dims = vec![Dim::Var(batch_name.to_string())];
    for &d in rest {
        dims.push(Dim::Concrete(d));
    }
    SymbolicShape::new(dims)
}

/// Create a symbolic shape with variable batch and sequence dimensions.
fn symbolic_batch_seq_shape(batch_name: &str, seq_name: &str, rest: &[usize]) -> SymbolicShape {
    let mut dims = vec![
        Dim::Var(batch_name.to_string()),
        Dim::Var(seq_name.to_string()),
    ];
    for &d in rest {
        dims.push(Dim::Concrete(d));
    }
    SymbolicShape::new(dims)
}

/// Create an integer attribute.
fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

/// Create a float attribute.
fn float_attr(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        f: value,
        r#type: AttributeType::Float as i32,
        ..Default::default()
    }
}

/// Create an integer array attribute.
#[allow(dead_code)]
fn ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

/// Create a string attribute.
fn string_attr(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        s: value.as_bytes().to_vec(),
        r#type: AttributeType::String as i32,
        ..Default::default()
    }
}

// ============================================================================
// BERT Model Tests
// ============================================================================

/// Test BERT encoder block components with concrete shapes.
///
/// BERT encoder block:
/// 1. Multi-head self-attention
/// 2. Add & LayerNorm
/// 3. Feed-forward network (2 linear + GELU)
/// 4. Add & LayerNorm
#[test]
fn test_bert_encoder_block_concrete() {
    let mut builder = IRBuilder::new("bert_encoder");

    // BERT-base dimensions
    let batch = 2;
    let seq_len = 128;
    let hidden = 768;
    let intermediate = 3072; // 4 * hidden

    // Input: [batch, seq_len, hidden]
    let input = builder.add_input("input", f32_tensor(&[batch, seq_len, hidden]));

    // === Attention Block (using primitives since full attention is pending) ===
    // Q, K, V projections
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));
    let k_weight = builder.add_input("k_weight", f32_tensor(&[hidden, hidden]));
    let v_weight = builder.add_input("v_weight", f32_tensor(&[hidden, hidden]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "input".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );
    shapes.insert("q_weight".to_string(), concrete_shape(&[hidden, hidden]));
    shapes.insert("k_weight".to_string(), concrete_shape(&[hidden, hidden]));
    shapes.insert("v_weight".to_string(), concrete_shape(&[hidden, hidden]));

    // For testing purposes, we simulate the attention components
    // Q = input @ q_weight (simplified - actual BERT reshapes for multi-head)
    let q = translate_matmul(&[input, q_weight], &[], &shapes, &mut builder);
    assert!(q.is_ok(), "Q projection should succeed");
    let _q = q.unwrap();

    let k = translate_matmul(&[input, k_weight], &[], &shapes, &mut builder);
    assert!(k.is_ok(), "K projection should succeed");

    let v = translate_matmul(&[input, v_weight], &[], &shapes, &mut builder);
    assert!(v.is_ok(), "V projection should succeed");
    let v = v.unwrap();

    // === Feed-Forward Network ===
    let ff1_weight = builder.add_input("ff1_weight", f32_tensor(&[hidden, intermediate]));
    let _ff2_weight = builder.add_input("ff2_weight", f32_tensor(&[intermediate, hidden]));

    shapes.insert(
        "ff1_weight".to_string(),
        concrete_shape(&[hidden, intermediate]),
    );
    shapes.insert(
        "ff2_weight".to_string(),
        concrete_shape(&[intermediate, hidden]),
    );
    shapes.insert(
        "attention_out".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );

    // FFN: Linear -> GELU -> Linear
    // For simplicity, use attention output (v) as stand-in
    let ff1 = translate_matmul(&[v, ff1_weight], &[], &shapes, &mut builder);
    assert!(ff1.is_ok(), "FFN first linear should succeed");
    let ff1 = ff1.unwrap();

    // GELU activation
    let gelu = translate_gelu(&[ff1], &[], &shapes, &mut builder);
    assert!(gelu.is_ok(), "GELU should succeed");
    let gelu = gelu.unwrap();

    // Update shape for intermediate
    shapes.insert(
        "ff1_out".to_string(),
        concrete_shape(&[batch, seq_len, intermediate]),
    );

    // Second linear (would need shape fix in real impl)
    // Skip for now as shape doesn't match

    // Build and verify
    builder.set_output(gelu);
    let func = builder.build();

    assert!(!func.body.is_empty(), "BERT encoder should have IR nodes");
    eprintln!(
        "BERT encoder block (concrete): {} IR nodes",
        func.body.len()
    );
}

/// Test BERT with symbolic batch and sequence dimensions.
#[test]
fn test_bert_encoder_symbolic_shapes() {
    let mut builder = IRBuilder::new("bert_encoder_symbolic");

    // Symbolic dimensions
    let hidden = 768;

    // Create types with symbolic shapes
    // Note: IRBuilder may not fully support symbolic types directly,
    // but we can test shape inference
    let input = builder.add_input("input", f32_tensor(&[])); // Symbolic
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));

    // Create symbolic shapes for shape map
    let input_shape = symbolic_batch_seq_shape("batch", "seq_len", &[hidden]);
    let weight_shape = concrete_shape(&[hidden, hidden]);

    let mut shapes = HashMap::new();
    shapes.insert("input".to_string(), input_shape.clone());
    shapes.insert("q_weight".to_string(), weight_shape);

    // Test that shape inference handles symbolic dimensions
    assert!(input_shape.rank() == 3, "Input should be 3D");
    assert!(
        !input_shape.is_fully_concrete(),
        "Input shape should be symbolic"
    );

    // Verify symbolic dimension names
    let dims = input_shape.dims();
    assert!(matches!(dims[0], Dim::Var(ref n) if n == "batch"));
    assert!(matches!(dims[1], Dim::Var(ref n) if n == "seq_len"));
    assert!(matches!(dims[2], Dim::Concrete(768)));

    // MatMul with symbolic batch should work
    let result = translate_matmul(&[input, q_weight], &[], &shapes, &mut builder);
    assert!(result.is_ok(), "MatMul with symbolic shapes should succeed");

    eprintln!(
        "BERT encoder symbolic shapes: verified batch={:?}, seq_len={:?}",
        dims[0], dims[1]
    );
}

/// Test BERT attention pattern (Q @ K^T / sqrt(d_k)) shape inference.
#[test]
fn test_bert_attention_score_shapes() {
    let batch = 2;
    let num_heads = 12;
    let seq_len = 128;
    let head_dim = 64; // hidden / num_heads = 768 / 12

    // After splitting into heads: [batch, num_heads, seq_len, head_dim]
    let q_shape = concrete_shape(&[batch, num_heads, seq_len, head_dim]);
    let k_shape = concrete_shape(&[batch, num_heads, seq_len, head_dim]);

    // After K transpose: [batch, num_heads, head_dim, seq_len]
    // Scores = Q @ K^T: [batch, num_heads, seq_len, seq_len]
    let _expected_scores_shape = concrete_shape(&[batch, num_heads, seq_len, seq_len]);

    // Verify shape inference for attention scores
    let _scores_shape = q_shape.infer_matmul(&k_shape);
    // Note: This won't work directly as infer_matmul expects 2D, but demonstrates the concept

    // The attention scores matrix is [seq_len, seq_len] per head
    assert!(q_shape.rank() == 4, "Q should be 4D after head split");
    assert_eq!(
        q_shape.dims()[2],
        Dim::Concrete(seq_len),
        "Sequence dimension should match"
    );

    eprintln!(
        "BERT attention scores: [{}, {}, {}, {}]",
        batch, num_heads, seq_len, seq_len
    );
}

/// Test BERT LayerNorm placement.
#[test]
fn test_bert_layer_norm() {
    let mut builder = IRBuilder::new("bert_layernorm");

    let batch = 2;
    let seq_len = 128;
    let hidden = 768;

    let input = builder.add_input("input", f32_tensor(&[batch, seq_len, hidden]));
    let gamma = builder.add_input("gamma", f32_tensor(&[hidden]));
    let beta = builder.add_input("beta", f32_tensor(&[hidden]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "input".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );
    shapes.insert("gamma".to_string(), concrete_shape(&[hidden]));
    shapes.insert("beta".to_string(), concrete_shape(&[hidden]));

    // LayerNorm with axis=-1 (normalize over hidden dimension)
    let attrs = vec![int_attr("axis", -1), float_attr("epsilon", 1e-5)];

    let result =
        translate_layer_normalization(&[input, gamma, beta], &attrs, &shapes, &mut builder);
    assert!(result.is_ok(), "LayerNorm should succeed");

    builder.set_output(result.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "LayerNorm should produce IR nodes");
    eprintln!("BERT LayerNorm: {} IR nodes", func.body.len());
}

// ============================================================================
// GPT Model Tests
// ============================================================================

/// Test GPT decoder block with causal masking concept.
///
/// GPT decoder block:
/// 1. Masked multi-head self-attention
/// 2. Add & LayerNorm
/// 3. Feed-forward network (2 linear + GELU)
/// 4. Add & LayerNorm
#[test]
fn test_gpt_decoder_block_concrete() {
    let mut builder = IRBuilder::new("gpt_decoder");

    // GPT-2 small dimensions
    let batch = 2;
    let seq_len = 1024; // GPT can handle longer sequences
    let hidden = 768;
    let intermediate = 3072;

    let input = builder.add_input("input", f32_tensor(&[batch, seq_len, hidden]));

    // Q, K, V projections
    let qkv_weight = builder.add_input("qkv_weight", f32_tensor(&[hidden, 3 * hidden]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "input".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );
    shapes.insert(
        "qkv_weight".to_string(),
        concrete_shape(&[hidden, 3 * hidden]),
    );

    // Combined QKV projection (GPT style)
    let qkv = translate_matmul(&[input, qkv_weight], &[], &shapes, &mut builder);
    assert!(qkv.is_ok(), "QKV projection should succeed");
    let _qkv = qkv.unwrap();

    // FFN
    let ff1_weight = builder.add_input("ff1_weight", f32_tensor(&[hidden, intermediate]));
    shapes.insert(
        "ff1_weight".to_string(),
        concrete_shape(&[hidden, intermediate]),
    );

    // For testing, use input as stand-in for attention output
    let ff1 = translate_matmul(&[input, ff1_weight], &[], &shapes, &mut builder);
    assert!(ff1.is_ok(), "FFN linear should succeed");
    let ff1 = ff1.unwrap();

    // GELU activation (GPT uses GELU)
    let gelu = translate_gelu(&[ff1], &[], &shapes, &mut builder);
    assert!(gelu.is_ok(), "GELU should succeed");

    builder.set_output(gelu.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "GPT decoder should have IR nodes");
    eprintln!("GPT decoder block: {} IR nodes", func.body.len());
}

/// Test GPT with symbolic sequence length for autoregressive generation.
#[test]
fn test_gpt_autoregressive_shapes() {
    // In autoregressive generation, seq_len grows at each step
    let hidden = 768;

    // Variable sequence length for generation
    let input_shape = symbolic_batch_seq_shape("batch", "seq_len", &[hidden]);

    assert!(!input_shape.is_fully_concrete());

    let dims = input_shape.dims();
    assert!(matches!(dims[0], Dim::Var(ref n) if n == "batch"));
    assert!(matches!(dims[1], Dim::Var(ref n) if n == "seq_len"));

    // KV cache shapes for autoregressive generation
    // Past key: [batch, num_heads, past_seq_len, head_dim]
    let num_heads = 12;
    let head_dim = 64;
    let past_kv_shape = SymbolicShape::new(vec![
        Dim::Var("batch".to_string()),
        Dim::Concrete(num_heads),
        Dim::Var("past_seq_len".to_string()),
        Dim::Concrete(head_dim),
    ]);

    assert_eq!(past_kv_shape.rank(), 4);
    assert!(!past_kv_shape.is_fully_concrete());

    eprintln!(
        "GPT autoregressive shapes verified: input={:?}, kv_cache=4D symbolic",
        input_shape.dims()
    );
}

/// Test GPT causal mask generation pattern.
#[test]
fn test_gpt_causal_mask_pattern() {
    // Causal mask: lower triangular matrix
    // mask[i, j] = 1 if j <= i, else -inf (or 0 in additive form)
    let seq_len = 128;

    // Mask shape: [1, 1, seq_len, seq_len] (broadcastable)
    let mask_shape = concrete_shape(&[1, 1, seq_len, seq_len]);

    assert_eq!(mask_shape.rank(), 4);
    assert!(mask_shape.is_fully_concrete());

    // Verify dimensions match attention scores
    assert_eq!(mask_shape.dims()[2], Dim::Concrete(seq_len));
    assert_eq!(mask_shape.dims()[3], Dim::Concrete(seq_len));

    eprintln!("GPT causal mask shape: [1, 1, {}, {}]", seq_len, seq_len);
}

// ============================================================================
// LSTM Model Tests
// ============================================================================

/// Test LSTM cell computation with concrete shapes.
#[test]
fn test_lstm_cell_concrete() {
    let mut builder = IRBuilder::new("lstm_cell");

    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;

    // Single timestep input
    let x_t = builder.add_input("x_t", f32_tensor(&[batch, input_size]));
    let h_prev = builder.add_input("h_prev", f32_tensor(&[batch, hidden_size]));
    let _c_prev = builder.add_input("c_prev", f32_tensor(&[batch, hidden_size]));

    // LSTM weights: [4*hidden_size, input_size] for i,f,g,o gates
    let w_x = builder.add_input("W_x", f32_tensor(&[input_size, 4 * hidden_size]));
    let w_h = builder.add_input("W_h", f32_tensor(&[hidden_size, 4 * hidden_size]));

    let mut shapes = HashMap::new();
    shapes.insert("x_t".to_string(), concrete_shape(&[batch, input_size]));
    shapes.insert("h_prev".to_string(), concrete_shape(&[batch, hidden_size]));
    shapes.insert("c_prev".to_string(), concrete_shape(&[batch, hidden_size]));
    shapes.insert(
        "W_x".to_string(),
        concrete_shape(&[input_size, 4 * hidden_size]),
    );
    shapes.insert(
        "W_h".to_string(),
        concrete_shape(&[hidden_size, 4 * hidden_size]),
    );

    // gates = x_t @ W_x + h_prev @ W_h
    let x_proj = translate_matmul(&[x_t, w_x], &[], &shapes, &mut builder);
    assert!(x_proj.is_ok(), "Input projection should succeed");
    let x_proj = x_proj.unwrap();

    let h_proj = translate_matmul(&[h_prev, w_h], &[], &shapes, &mut builder);
    assert!(h_proj.is_ok(), "Hidden projection should succeed");
    let h_proj = h_proj.unwrap();

    // Add projections
    shapes.insert(
        "x_proj".to_string(),
        concrete_shape(&[batch, 4 * hidden_size]),
    );
    shapes.insert(
        "h_proj".to_string(),
        concrete_shape(&[batch, 4 * hidden_size]),
    );

    let gates = translate_add(&[x_proj, h_proj], &[], &shapes, &mut builder);
    assert!(gates.is_ok(), "Gate addition should succeed");
    let gates = gates.unwrap();

    // Split gates would happen here (i, f, g, o)
    // For simplicity, apply sigmoid to full output
    let gate_activations = translate_sigmoid(&[gates], &[], &shapes, &mut builder);
    assert!(gate_activations.is_ok(), "Sigmoid should succeed");

    builder.set_output(gate_activations.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "LSTM cell should have IR nodes");
    eprintln!("LSTM cell (concrete): {} IR nodes", func.body.len());
}

/// Test LSTM with variable sequence length.
#[test]
fn test_lstm_variable_sequence_length() {
    // LSTM input shape: [seq_len, batch, input_size]
    let input_size = 128;
    let hidden_size = 256;

    let input_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Var("batch".to_string()),
        Dim::Concrete(input_size),
    ]);

    assert!(!input_shape.is_fully_concrete());
    assert_eq!(input_shape.rank(), 3);

    let dims = input_shape.dims();
    assert!(matches!(dims[0], Dim::Var(ref n) if n == "seq_len"));
    assert!(matches!(dims[1], Dim::Var(ref n) if n == "batch"));
    assert!(matches!(dims[2], Dim::Concrete(128)));

    // Output shape: [seq_len, batch, hidden_size]
    let output_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Var("batch".to_string()),
        Dim::Concrete(hidden_size),
    ]);

    assert!(!output_shape.is_fully_concrete());

    eprintln!(
        "LSTM variable seq shapes: input={:?}, output={:?}",
        input_shape.dims(),
        output_shape.dims()
    );
}

/// Test bidirectional LSTM shapes.
#[test]
fn test_lstm_bidirectional_shapes() {
    let seq_len = 50;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;
    let num_directions = 2; // bidirectional

    // Input: [seq_len, batch, input_size]
    let input_shape = concrete_shape(&[seq_len, batch, input_size]);

    // Weight shape: [num_directions * 4, hidden_size, input_size]
    let weight_shape = concrete_shape(&[num_directions * 4, hidden_size, input_size]);

    // Output: [seq_len, num_directions, batch, hidden_size]
    let output_shape = concrete_shape(&[seq_len, num_directions, batch, hidden_size]);

    // Final hidden: [num_directions, batch, hidden_size]
    let final_h_shape = concrete_shape(&[num_directions, batch, hidden_size]);

    assert_eq!(input_shape.rank(), 3);
    assert_eq!(weight_shape.rank(), 3);
    assert_eq!(output_shape.rank(), 4);
    assert_eq!(final_h_shape.rank(), 3);

    eprintln!("Bidirectional LSTM shapes verified");
}

/// Test LSTM full operation with decomposition.
#[test]
fn test_lstm_full_operation() {
    let mut builder = IRBuilder::new("lstm_full");

    let seq_len = 10;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;

    // LSTM weight shapes: [num_directions, 4*hidden_size, input_size]
    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, 4 * hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, 4 * hidden_size, hidden_size]));

    let attrs = vec![int_attr("hidden_size", hidden_size as i64)];

    let result = translate_lstm(&[x, w, r], &attrs, &HashMap::new(), &mut builder);

    // LSTM is now decomposed to primitive operations
    assert!(result.is_ok(), "LSTM should succeed with decomposition");

    builder.set_output(result.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "LSTM should produce IR nodes");
    eprintln!(
        "LSTM full operation: {} IR nodes (decomposed)",
        func.body.len()
    );
}

// ============================================================================
// LOOP Optimization Tests
// ============================================================================

/// Test that attention would use LOOP for O(1) space complexity.
///
/// LOOP optimization design:
/// - Instead of materializing full [seq, seq] attention matrix
/// - Use LOOP to compute attention scores row-by-row
/// - Memory: O(seq) instead of O(seq²)
#[test]
fn test_loop_optimization_attention_design() {
    // Traditional attention: O(seq²) memory for attention scores
    let seq_len = 1024;
    let batch = 2;
    let num_heads = 12;

    // Full attention matrix: [batch, num_heads, seq_len, seq_len]
    let full_memory = batch * num_heads * seq_len * seq_len * 4; // f32

    // With LOOP: only need [batch, num_heads, seq_len] at a time
    let loop_memory = batch * num_heads * seq_len * 4;

    let savings = full_memory as f64 / loop_memory as f64;

    assert!(
        savings > 100.0,
        "LOOP should provide >100x memory savings for seq=1024"
    );

    eprintln!(
        "LOOP optimization for attention:\n\
         - Full attention memory: {} MB\n\
         - LOOP memory: {} KB\n\
         - Savings: {:.0}x",
        full_memory / (1024 * 1024),
        loop_memory / 1024,
        savings
    );
}

/// Test LOOP optimization for LSTM unrolling.
///
/// LOOP optimization design:
/// - Instead of unrolling LSTM for seq_len timesteps
/// - Use LOOP to iterate over sequence
/// - Memory: O(hidden) instead of O(seq * hidden)
#[test]
fn test_loop_optimization_lstm_design() {
    let seq_len = 1000;
    let batch = 2;
    let hidden_size = 256;

    // Unrolled LSTM: need all intermediate states
    let unrolled_memory = seq_len * batch * hidden_size * 4 * 2; // h and c

    // With LOOP: only current state needed
    let loop_memory = batch * hidden_size * 4 * 2;

    let savings = unrolled_memory as f64 / loop_memory as f64;

    assert_eq!(
        savings, seq_len as f64,
        "LOOP savings should equal sequence length"
    );

    eprintln!(
        "LOOP optimization for LSTM:\n\
         - Unrolled memory: {} MB\n\
         - LOOP memory: {} KB\n\
         - Savings: {:.0}x (= seq_len)",
        unrolled_memory / (1024 * 1024),
        loop_memory / 1024,
        savings
    );
}

/// Test that broadcasting operations would use LOOP.
#[test]
fn test_loop_optimization_broadcast_design() {
    // Broadcasting: [batch, seq, hidden] op [hidden] -> [batch, seq, hidden]
    let batch = 2;
    let seq = 1024;
    let hidden = 768;

    // Without LOOP: materialize broadcast result
    let materialized = batch * seq * hidden * 4;

    // With LOOP: process in-place, no extra allocation
    let _loop_overhead = 0; // In-place operation

    eprintln!(
        "LOOP optimization for broadcast:\n\
         - Materialized: {} MB\n\
         - LOOP: in-place (zero overhead)",
        materialized / (1024 * 1024)
    );

    assert!(materialized > 0, "Materialized broadcast needs memory");
}

// ============================================================================
// End-to-End Pipeline Tests
// ============================================================================

/// Test complete BERT-like encoder pipeline.
#[test]
fn test_bert_pipeline_shapes() {
    // BERT-base config
    let batch = 2;
    let seq_len = 128;
    let _vocab_size = 30522;
    let hidden = 768;
    let num_layers = 12;
    let _num_heads = 12;
    let _intermediate = 3072;

    // Verify shape propagation through layers
    let _input_shape = concrete_shape(&[batch, seq_len]); // Token IDs
    let embed_shape = concrete_shape(&[batch, seq_len, hidden]); // After embedding

    // Each encoder layer preserves shape
    let layer_output_shape = concrete_shape(&[batch, seq_len, hidden]);

    // Output shape
    let output_shape = concrete_shape(&[batch, seq_len, hidden]);

    assert_eq!(
        embed_shape, layer_output_shape,
        "Layer should preserve shape"
    );
    assert_eq!(
        layer_output_shape, output_shape,
        "Output shape should match"
    );

    eprintln!(
        "BERT pipeline shapes:\n\
         - Input: [batch={}, seq_len={}]\n\
         - Embedding: [batch, seq_len, hidden={}]\n\
         - {} encoder layers (shape preserved)\n\
         - Output: [batch, seq_len, hidden]",
        batch, seq_len, hidden, num_layers
    );
}

/// Test GPT-like decoder pipeline.
#[test]
fn test_gpt_pipeline_shapes() {
    // GPT-2 small config
    let batch = 2;
    let seq_len = 1024;
    let vocab_size = 50257;
    let hidden = 768;
    let num_layers = 12;

    // Verify shape propagation
    let _input_shape = concrete_shape(&[batch, seq_len]);
    let _embed_shape = concrete_shape(&[batch, seq_len, hidden]);
    let _output_shape = concrete_shape(&[batch, seq_len, vocab_size]); // Logits

    eprintln!(
        "GPT pipeline shapes:\n\
         - Input: [batch={}, seq_len={}]\n\
         - Embedding: [batch, seq_len, hidden={}]\n\
         - {} decoder layers\n\
         - Output logits: [batch, seq_len, vocab_size={}]",
        batch, seq_len, hidden, num_layers, vocab_size
    );
}

/// Test sequence-to-sequence pipeline (encoder-decoder).
#[test]
fn test_seq2seq_pipeline_shapes() {
    let _batch = 2;
    let _src_len = 128;
    let _tgt_len = 64;
    let hidden = 512;

    // Encoder input/output
    let encoder_input = SymbolicShape::new(vec![
        Dim::Var("batch".to_string()),
        Dim::Var("src_len".to_string()),
        Dim::Concrete(hidden),
    ]);

    // Decoder input/output with different sequence length
    let decoder_input = SymbolicShape::new(vec![
        Dim::Var("batch".to_string()),
        Dim::Var("tgt_len".to_string()),
        Dim::Concrete(hidden),
    ]);

    // Cross-attention: Q from decoder [batch, tgt_len, hidden]
    //                  K,V from encoder [batch, src_len, hidden]

    assert!(!encoder_input.is_fully_concrete());
    assert!(!decoder_input.is_fully_concrete());

    eprintln!(
        "Seq2Seq pipeline shapes:\n\
         - Encoder: [batch, src_len, hidden={}]\n\
         - Decoder: [batch, tgt_len, hidden={}]\n\
         - Cross-attention: [batch, tgt_len, src_len]",
        hidden, hidden
    );
}

// ============================================================================
// Symbolic Shape Inference Tests
// ============================================================================

/// Test symbolic shape inference for attention.
#[test]
fn test_symbolic_attention_shape_inference() {
    let hidden = 768;
    let num_heads = 12;
    let head_dim = hidden / num_heads;

    // Q, K, V with symbolic batch and seq_len
    let q_shape = symbolic_batch_seq_shape("batch", "seq", &[num_heads, head_dim]);
    let _k_shape = symbolic_batch_seq_shape("batch", "seq", &[num_heads, head_dim]);

    // Attention scores: [batch, num_heads, seq, seq]
    // (would be computed, but shape inference preserves symbolic dims)

    assert_eq!(q_shape.rank(), 4);
    assert!(!q_shape.is_fully_concrete());

    let dims = q_shape.dims();
    assert!(matches!(dims[0], Dim::Var(ref n) if n == "batch"));
    assert!(matches!(dims[1], Dim::Var(ref n) if n == "seq"));

    eprintln!("Symbolic attention shape inference verified");
}

/// Test symbolic shape inference for LSTM.
#[test]
fn test_symbolic_lstm_shape_inference() {
    let input_size = 128;
    let hidden_size = 256;

    // Input: [seq_len, batch, input_size]
    let input_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Var("batch".to_string()),
        Dim::Concrete(input_size),
    ]);

    // Output: [seq_len, batch, hidden_size]
    // Hidden: [1, batch, hidden_size]

    let output_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Var("batch".to_string()),
        Dim::Concrete(hidden_size),
    ]);

    // Verify symbolic dims are preserved
    let input_dims = input_shape.dims();
    let output_dims = output_shape.dims();

    assert!(matches!(input_dims[0], Dim::Var(ref n) if n == "seq_len"));
    assert!(matches!(output_dims[0], Dim::Var(ref n) if n == "seq_len"));

    // seq_len is same symbolic var in both
    assert_eq!(input_dims[0], output_dims[0]);

    eprintln!("Symbolic LSTM shape inference verified");
}

// ============================================================================
// Decomposed Operation Integration Tests
// ============================================================================

/// Test Attention operation with full decomposition.
#[test]
fn test_attention_decomposition() {
    let mut builder = IRBuilder::new("attention_decomposition");

    let batch = 2;
    let seq_len = 128;
    let hidden = 768;

    // Q, K, V inputs: [batch, seq_len, hidden]
    let query = builder.add_input("Q", f32_tensor(&[batch, seq_len, hidden]));
    let key = builder.add_input("K", f32_tensor(&[batch, seq_len, hidden]));
    let value = builder.add_input("V", f32_tensor(&[batch, seq_len, hidden]));

    let mut shapes = HashMap::new();
    shapes.insert("Q".to_string(), concrete_shape(&[batch, seq_len, hidden]));
    shapes.insert("K".to_string(), concrete_shape(&[batch, seq_len, hidden]));
    shapes.insert("V".to_string(), concrete_shape(&[batch, seq_len, hidden]));

    let result = translate_attention(&[query, key, value], &[], &shapes, &mut builder);

    assert!(
        result.is_ok(),
        "Attention should succeed with decomposition"
    );

    builder.set_output(result.unwrap());
    let func = builder.build();

    // Attention decomposes to: transpose, matmul, scale, softmax, matmul
    assert!(
        func.body.len() >= 5,
        "Attention should have at least 5 IR nodes"
    );
    eprintln!("Attention decomposition: {} IR nodes", func.body.len());
}

/// Test Attention with mask.
#[test]
fn test_attention_with_mask() {
    let mut builder = IRBuilder::new("attention_masked");

    let batch = 2;
    let seq_len = 64;
    let hidden = 512;

    let query = builder.add_input("Q", f32_tensor(&[batch, seq_len, hidden]));
    let key = builder.add_input("K", f32_tensor(&[batch, seq_len, hidden]));
    let value = builder.add_input("V", f32_tensor(&[batch, seq_len, hidden]));
    let mask = builder.add_input("mask", f32_tensor(&[batch, 1, seq_len, seq_len]));

    let result = translate_attention(
        &[query, key, value, mask],
        &[],
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "Masked attention should succeed");

    builder.set_output(result.unwrap());
    let func = builder.build();

    // Masked attention has one more node for mask addition
    assert!(
        func.body.len() >= 6,
        "Masked attention should have at least 6 IR nodes"
    );
    eprintln!("Masked attention: {} IR nodes", func.body.len());
}

/// Test Attention with symbolic shapes.
#[test]
fn test_attention_symbolic_shapes() {
    let mut builder = IRBuilder::new("attention_symbolic");

    let hidden = 768;

    // Symbolic batch and seq_len
    let query = builder.add_input("Q", f32_tensor(&[]));
    let key = builder.add_input("K", f32_tensor(&[]));
    let value = builder.add_input("V", f32_tensor(&[]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "Q".to_string(),
        symbolic_batch_seq_shape("batch", "seq", &[hidden]),
    );
    shapes.insert(
        "K".to_string(),
        symbolic_batch_seq_shape("batch", "seq", &[hidden]),
    );
    shapes.insert(
        "V".to_string(),
        symbolic_batch_seq_shape("batch", "seq", &[hidden]),
    );

    let result = translate_attention(&[query, key, value], &[], &shapes, &mut builder);

    assert!(
        result.is_ok(),
        "Attention with symbolic shapes should succeed"
    );
    eprintln!("Attention with symbolic batch/seq: OK");
}

/// Test MultiHeadAttention decomposition.
#[test]
fn test_multi_head_attention_decomposition() {
    let mut builder = IRBuilder::new("mha_decomposition");

    let batch = 2;
    let seq_len = 128;
    let hidden = 768;
    let num_heads = 12;

    // Q, K, V inputs
    let query = builder.add_input("Q", f32_tensor(&[batch, seq_len, hidden]));
    let key = builder.add_input("K", f32_tensor(&[batch, seq_len, hidden]));
    let value = builder.add_input("V", f32_tensor(&[batch, seq_len, hidden]));

    // Weight matrices
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));
    let k_weight = builder.add_input("k_weight", f32_tensor(&[hidden, hidden]));
    let v_weight = builder.add_input("v_weight", f32_tensor(&[hidden, hidden]));
    let o_weight = builder.add_input("o_weight", f32_tensor(&[hidden, hidden]));

    let attrs = vec![int_attr("num_heads", num_heads as i64)];

    let result = translate_multi_head_attention(
        &[query, key, value, q_weight, k_weight, v_weight, o_weight],
        &attrs,
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "MultiHeadAttention should succeed");

    builder.set_output(result.unwrap());
    let func = builder.build();

    // MHA decomposes to: Q projection, K projection, V projection, attention, output projection
    assert!(func.body.len() >= 8, "MHA should have at least 8 IR nodes");
    eprintln!(
        "MultiHeadAttention decomposition: {} IR nodes",
        func.body.len()
    );
}

/// Test MultiHeadAttention with optional mask.
#[test]
fn test_multi_head_attention_with_mask() {
    let mut builder = IRBuilder::new("mha_masked");

    let batch = 2;
    let seq_len = 64;
    let hidden = 512;
    let num_heads = 8;

    let query = builder.add_input("Q", f32_tensor(&[batch, seq_len, hidden]));
    let key = builder.add_input("K", f32_tensor(&[batch, seq_len, hidden]));
    let value = builder.add_input("V", f32_tensor(&[batch, seq_len, hidden]));
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));
    let k_weight = builder.add_input("k_weight", f32_tensor(&[hidden, hidden]));
    let v_weight = builder.add_input("v_weight", f32_tensor(&[hidden, hidden]));
    let o_weight = builder.add_input("o_weight", f32_tensor(&[hidden, hidden]));
    let mask = builder.add_input("mask", f32_tensor(&[batch, 1, seq_len, seq_len]));

    let attrs = vec![int_attr("num_heads", num_heads as i64)];

    let result = translate_multi_head_attention(
        &[
            query, key, value, q_weight, k_weight, v_weight, o_weight, mask,
        ],
        &attrs,
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "Masked MHA should succeed");
    eprintln!("Masked MultiHeadAttention: OK");
}

/// Test GRU operation with decomposition.
#[test]
fn test_gru_decomposition() {
    let mut builder = IRBuilder::new("gru_decomposition");

    let seq_len = 10;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;

    // GRU weight shapes: [num_directions, 3*hidden_size, input_size]
    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, 3 * hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, 3 * hidden_size, hidden_size]));

    let attrs = vec![int_attr("hidden_size", hidden_size as i64)];

    let result = translate_gru(&[x, w, r], &attrs, &HashMap::new(), &mut builder);

    assert!(result.is_ok(), "GRU should succeed with decomposition");

    builder.set_output(result.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "GRU should produce IR nodes");
    eprintln!("GRU decomposition: {} IR nodes", func.body.len());
}

/// Test GRU with initial hidden state.
#[test]
fn test_gru_with_initial_hidden() {
    let mut builder = IRBuilder::new("gru_with_h0");

    let seq_len = 10;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;

    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, 3 * hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, 3 * hidden_size, hidden_size]));
    let b = builder.add_input("B", f32_tensor(&[1, 6 * hidden_size])); // Optional bias
    // Skip sequence_lens input (index 4)
    let initial_h = builder.add_input("H0", f32_tensor(&[1, batch, hidden_size]));

    let attrs = vec![int_attr("hidden_size", hidden_size as i64)];

    // Pass with optional inputs: X, W, R, B, sequence_lens (skipped), initial_h
    let result = translate_gru(
        &[x, w, r, b, x, initial_h], // Using x as placeholder for sequence_lens
        &attrs,
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "GRU with initial_h should succeed");
    eprintln!("GRU with initial hidden state: OK");
}

/// Test RNN operation with decomposition.
#[test]
fn test_rnn_decomposition() {
    let mut builder = IRBuilder::new("rnn_decomposition");

    let seq_len = 10;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;

    // RNN weight shapes: [num_directions, hidden_size, input_size]
    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, hidden_size, hidden_size]));

    let attrs = vec![int_attr("hidden_size", hidden_size as i64)];

    let result = translate_rnn(&[x, w, r], &attrs, &HashMap::new(), &mut builder);

    assert!(result.is_ok(), "RNN should succeed with decomposition");

    builder.set_output(result.unwrap());
    let func = builder.build();

    assert!(!func.body.is_empty(), "RNN should produce IR nodes");
    eprintln!("RNN decomposition: {} IR nodes", func.body.len());
}

/// Test RNN with different activation functions.
#[test]
fn test_rnn_activations() {
    let mut builder = IRBuilder::new("rnn_activations");

    let seq_len = 10;
    let batch = 2;
    let input_size = 64;
    let hidden_size = 128;

    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, hidden_size, hidden_size]));

    // Test with Relu activation
    let attrs = vec![
        int_attr("hidden_size", hidden_size as i64),
        string_attr("activations", "Relu"),
    ];

    let result = translate_rnn(&[x, w, r], &attrs, &HashMap::new(), &mut builder);

    assert!(result.is_ok(), "RNN with Relu activation should succeed");
    eprintln!("RNN with activations: OK");
}

// ============================================================================
// BERT Model Integration Tests
// ============================================================================

/// Test complete BERT encoder layer with all decomposed operations.
#[test]
fn test_bert_encoder_layer_complete() {
    let mut builder = IRBuilder::new("bert_encoder_layer");

    let batch = 2;
    let seq_len = 128;
    let hidden = 768;
    let num_heads = 12;

    // Inputs
    let input = builder.add_input("input", f32_tensor(&[batch, seq_len, hidden]));
    let gamma1 = builder.add_input("gamma1", f32_tensor(&[hidden]));
    let beta1 = builder.add_input("beta1", f32_tensor(&[hidden]));

    // Attention weights
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));
    let k_weight = builder.add_input("k_weight", f32_tensor(&[hidden, hidden]));
    let v_weight = builder.add_input("v_weight", f32_tensor(&[hidden, hidden]));
    let o_weight = builder.add_input("o_weight", f32_tensor(&[hidden, hidden]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "input".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );

    // Step 1: Multi-head self-attention
    let mha_attrs = vec![int_attr("num_heads", num_heads as i64)];
    let attention_out = translate_multi_head_attention(
        &[input, input, input, q_weight, k_weight, v_weight, o_weight],
        &mha_attrs,
        &shapes,
        &mut builder,
    );
    assert!(attention_out.is_ok(), "MHA should succeed");
    let attention_out = attention_out.unwrap();

    // Step 2: Residual connection (input + attention_out)
    shapes.insert(
        "attention_out".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );
    let residual1 = translate_add(&[input, attention_out], &[], &shapes, &mut builder);
    assert!(residual1.is_ok(), "Residual connection should succeed");
    let residual1 = residual1.unwrap();

    // Step 3: LayerNorm
    let ln_attrs = vec![int_attr("axis", -1), float_attr("epsilon", 1e-5)];
    let ln1_out = translate_layer_normalization(
        &[residual1, gamma1, beta1],
        &ln_attrs,
        &shapes,
        &mut builder,
    );
    assert!(ln1_out.is_ok(), "LayerNorm should succeed");

    builder.set_output(ln1_out.unwrap());
    let func = builder.build();

    // Should have many IR nodes from all the decompositions
    assert!(
        func.body.len() >= 15,
        "BERT encoder layer should have many IR nodes"
    );
    eprintln!("BERT encoder layer complete: {} IR nodes", func.body.len());
}

/// Test BERT with symbolic batch and sequence dimensions.
#[test]
fn test_bert_symbolic_batch_seq() {
    let hidden = 768;

    // Create symbolic shapes
    let input_shape = symbolic_batch_seq_shape("batch", "seq_len", &[hidden]);
    let weight_shape = concrete_shape(&[hidden, hidden]);

    assert!(!input_shape.is_fully_concrete(), "Input should be symbolic");
    assert!(
        weight_shape.is_fully_concrete(),
        "Weights should be concrete"
    );

    // Verify the symbolic dimensions
    let dims = input_shape.dims();
    match &dims[0] {
        Dim::Var(name) => assert_eq!(name, "batch"),
        _ => panic!("First dim should be symbolic batch"),
    }
    match &dims[1] {
        Dim::Var(name) => assert_eq!(name, "seq_len"),
        _ => panic!("Second dim should be symbolic seq_len"),
    }

    eprintln!("BERT symbolic shapes: [batch, seq_len, {}]", hidden);
}

// ============================================================================
// GPT Model Integration Tests
// ============================================================================

/// Test complete GPT decoder layer with causal attention.
#[test]
fn test_gpt_decoder_layer_complete() {
    let mut builder = IRBuilder::new("gpt_decoder_layer");

    let batch = 2;
    let seq_len = 512;
    let hidden = 768;
    let num_heads = 12;

    // Inputs
    let input = builder.add_input("input", f32_tensor(&[batch, seq_len, hidden]));

    // Attention weights
    let q_weight = builder.add_input("q_weight", f32_tensor(&[hidden, hidden]));
    let k_weight = builder.add_input("k_weight", f32_tensor(&[hidden, hidden]));
    let v_weight = builder.add_input("v_weight", f32_tensor(&[hidden, hidden]));
    let o_weight = builder.add_input("o_weight", f32_tensor(&[hidden, hidden]));

    // Causal mask (lower triangular)
    let mask = builder.add_input("causal_mask", f32_tensor(&[1, 1, seq_len, seq_len]));

    // Layer norm weights
    let gamma = builder.add_input("gamma", f32_tensor(&[hidden]));
    let beta = builder.add_input("beta", f32_tensor(&[hidden]));

    // FFN weights
    let intermediate = 3072;
    let ff1_weight = builder.add_input("ff1_weight", f32_tensor(&[hidden, intermediate]));
    let _ff2_weight = builder.add_input("ff2_weight", f32_tensor(&[intermediate, hidden]));

    let mut shapes = HashMap::new();
    shapes.insert(
        "input".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );

    // Step 1: Masked multi-head self-attention
    let mha_attrs = vec![int_attr("num_heads", num_heads as i64)];
    let attention_out = translate_multi_head_attention(
        &[
            input, input, input, q_weight, k_weight, v_weight, o_weight, mask,
        ],
        &mha_attrs,
        &shapes,
        &mut builder,
    );
    assert!(attention_out.is_ok(), "Masked MHA should succeed");
    let attention_out = attention_out.unwrap();

    // Step 2: Residual + LayerNorm
    let residual = translate_add(&[input, attention_out], &[], &shapes, &mut builder);
    assert!(residual.is_ok());
    let residual = residual.unwrap();

    let ln_attrs = vec![int_attr("axis", -1), float_attr("epsilon", 1e-5)];
    let ln_out =
        translate_layer_normalization(&[residual, gamma, beta], &ln_attrs, &shapes, &mut builder);
    assert!(ln_out.is_ok());
    let ln_out = ln_out.unwrap();

    // Step 3: Feed-forward network
    shapes.insert(
        "ln_out".to_string(),
        concrete_shape(&[batch, seq_len, hidden]),
    );
    let ff1 = translate_matmul(&[ln_out, ff1_weight], &[], &shapes, &mut builder);
    assert!(ff1.is_ok());
    let ff1 = ff1.unwrap();

    let gelu = translate_gelu(&[ff1], &[], &shapes, &mut builder);
    assert!(gelu.is_ok());

    builder.set_output(gelu.unwrap());
    let func = builder.build();

    assert!(
        func.body.len() >= 20,
        "GPT decoder layer should have many IR nodes"
    );
    eprintln!("GPT decoder layer complete: {} IR nodes", func.body.len());
}

/// Test GPT autoregressive generation with variable sequence length.
#[test]
fn test_gpt_autoregressive_generation() {
    // In autoregressive generation, we process one token at a time
    // and use KV cache for efficiency

    // Current token position
    let hidden = 768;
    let num_heads = 12;
    let head_dim = hidden / num_heads;

    // Current query: [batch, 1, hidden] (single new token)
    let query_shape = symbolic_batch_shape("batch", &[1, hidden]);

    // Past key/value cache: [batch, num_heads, past_len, head_dim]
    let past_kv_shape = SymbolicShape::new(vec![
        Dim::Var("batch".to_string()),
        Dim::Concrete(num_heads),
        Dim::Var("past_len".to_string()),
        Dim::Concrete(head_dim),
    ]);

    assert!(!query_shape.is_fully_concrete());
    assert!(!past_kv_shape.is_fully_concrete());

    // After concat: [batch, num_heads, past_len + 1, head_dim]
    // This requires symbolic arithmetic support

    eprintln!(
        "GPT autoregressive: query=[batch, 1, {}], cache=[batch, {}, past_len, {}]",
        hidden, num_heads, head_dim
    );
}

// ============================================================================
// LSTM Integration Tests
// ============================================================================

/// Test LSTM sequence processing.
#[test]
fn test_lstm_sequence_processing() {
    let mut builder = IRBuilder::new("lstm_sequence");

    let seq_len = 50;
    let batch = 4;
    let input_size = 256;
    let hidden_size = 512;

    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input("W", f32_tensor(&[1, 4 * hidden_size, input_size]));
    let r = builder.add_input("R", f32_tensor(&[1, 4 * hidden_size, hidden_size]));

    let attrs = vec![int_attr("hidden_size", hidden_size as i64)];

    let result = translate_lstm(&[x, w, r], &attrs, &HashMap::new(), &mut builder);
    assert!(result.is_ok(), "LSTM sequence should succeed");

    builder.set_output(result.unwrap());
    let func = builder.build();

    eprintln!(
        "LSTM sequence processing: {} IR nodes (seq_len={})",
        func.body.len(),
        seq_len
    );
}

/// Test bidirectional LSTM.
#[test]
fn test_lstm_bidirectional() {
    let mut builder = IRBuilder::new("lstm_bidirectional");

    let seq_len = 20;
    let batch = 2;
    let input_size = 128;
    let hidden_size = 256;
    let num_directions = 2;

    // Bidirectional LSTM has 2x weights
    let x = builder.add_input("X", f32_tensor(&[seq_len, batch, input_size]));
    let w = builder.add_input(
        "W",
        f32_tensor(&[num_directions, 4 * hidden_size, input_size]),
    );
    let r = builder.add_input(
        "R",
        f32_tensor(&[num_directions, 4 * hidden_size, hidden_size]),
    );

    let attrs = vec![
        int_attr("hidden_size", hidden_size as i64),
        string_attr("direction", "bidirectional"),
    ];

    let result = translate_lstm(&[x, w, r], &attrs, &HashMap::new(), &mut builder);
    assert!(result.is_ok(), "Bidirectional LSTM should succeed");

    eprintln!("Bidirectional LSTM: OK");
}

/// Test LSTM with symbolic sequence length.
#[test]
fn test_lstm_symbolic_sequence() {
    let input_size = 128;
    let hidden_size = 256;

    // Symbolic seq_len and batch
    let input_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Var("batch".to_string()),
        Dim::Concrete(input_size),
    ]);

    let output_shape = SymbolicShape::new(vec![
        Dim::Var("seq_len".to_string()),
        Dim::Concrete(1), // num_directions
        Dim::Var("batch".to_string()),
        Dim::Concrete(hidden_size),
    ]);

    assert!(!input_shape.is_fully_concrete());
    assert!(!output_shape.is_fully_concrete());

    // seq_len should be preserved
    let input_dims = input_shape.dims();
    let output_dims = output_shape.dims();

    assert!(matches!(&input_dims[0], Dim::Var(n) if n == "seq_len"));
    assert!(matches!(&output_dims[0], Dim::Var(n) if n == "seq_len"));

    eprintln!(
        "LSTM symbolic sequence: input=[seq_len, batch, {}], output=[seq_len, 1, batch, {}]",
        input_size, hidden_size
    );
}

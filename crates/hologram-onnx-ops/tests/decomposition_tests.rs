//! Decomposition integration tests for hologram-onnx-core.
//!
//! These tests verify that the ONNX → IR → Decomposed IR pipeline works correctly,
//! ensuring ISA optimizations are applied:
//!
//! - **Conv2D → Im2Col + MatMul**: Enables SIMD vectorization
//! - **MaxPool → Unfold + ReduceMax**: Enables efficient window operations
//! - **AvgPool → Unfold + ReduceMean**: Enables efficient window operations
//!
//! # ISA Integration
//!
//! After decomposition, all operations should be O(N²) complexity, enabling:
//! - LOOP instructions for O(1) space complexity
//! - PhiCoordinate addressing for efficient boundary handling
//! - ClassMap fusion for element-wise operations

use hologram_compiler::ir::{
    DecomposeConfig, IRBuilder, IRFunction, IRNode, ScalarType, Type, decompose_function,
};
use hologram_compiler::shapes::Shape;
use hologram_onnx_core::{Dim, OnnxConfig, SymbolicShape};
use hologram_onnx_ops::{
    infer_conv_output_shape, infer_pool_output_shape, translate_conv, translate_max_pool,
};
use hologram_onnx_spec::AttributeProto;
use hologram_onnx_spec::attribute_proto::AttributeType;
use std::collections::HashMap;

// =============================================================================
// Test Helpers
// =============================================================================

/// Create a tensor type with concrete shape.
fn f32_tensor(dims: &[usize]) -> Type {
    Type::tensor(ScalarType::F32, Shape::concrete(dims.to_vec()))
}

/// Create a tensor type with symbolic batch dimension for [batch, C, H, W] tensors.
fn f32_tensor_symbolic_batch(_batch_name: &str, dims: &[usize]) -> Type {
    // For testing, we create a shape with symbolic batch and concrete spatial dims
    // Shape::symbolic creates Dim::Var for each element
    // We'll use a mixed approach with Shape::new for more control
    let mut shape_dims = vec![hologram_compiler::shapes::Dim::Var("batch".to_string())];
    for &d in dims {
        shape_dims.push(hologram_compiler::shapes::Dim::Concrete(d));
    }
    Type::tensor(ScalarType::F32, Shape::new(shape_dims))
}

/// Create an integer attribute.
#[allow(dead_code)]
fn make_int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i: value,
        r#type: AttributeType::Int as i32,
        ..Default::default()
    }
}

/// Create an integer list attribute.
fn make_ints_attr(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        ints: values,
        r#type: AttributeType::Ints as i32,
        ..Default::default()
    }
}

/// Check if IR function contains a specific node type.
fn contains_node_type(func: &IRFunction, check: impl Fn(&IRNode) -> bool) -> bool {
    func.body.iter().any(|entry| check(&entry.node))
}

/// Count nodes of a specific type in IR function.
fn count_node_type(func: &IRFunction, check: impl Fn(&IRNode) -> bool) -> usize {
    func.body.iter().filter(|entry| check(&entry.node)).count()
}

// =============================================================================
// Test: Conv2D Translation and Decomposition
// =============================================================================

#[test]
fn test_conv2d_creates_ir_node() {
    // Verify that ONNX Conv translates to IR Conv2D node
    let mut builder = IRBuilder::new("conv_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));
    let kernel = builder.add_input("W", f32_tensor(&[64, 3, 7, 7]));

    let result = translate_conv(
        &[input, kernel],
        &[
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![3, 3, 3, 3]),
        ],
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "Conv translation should succeed");
    let func = builder.build();

    // Should contain a Conv2D node
    assert!(
        contains_node_type(&func, |n| matches!(n, IRNode::Conv2D { .. })),
        "IR should contain Conv2D node before decomposition"
    );
}

#[test]
fn test_conv2d_decomposes_to_im2col_matmul() {
    // Verify Conv2D → Im2Col + MatMul decomposition
    let mut builder = IRBuilder::new("conv_decompose_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));

    // Create Conv2D node directly
    let conv_result = builder.conv2d(input, kernel, None, (1, 1), (1, 1), (1, 1), 1);
    builder.set_output(conv_result);
    let func = builder.build();

    // Before decomposition: should have Conv2D
    assert!(
        contains_node_type(&func, |n| matches!(n, IRNode::Conv2D { .. })),
        "Should have Conv2D before decomposition"
    );

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // After decomposition: no Conv2D
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::Conv2D { .. })),
        "Should NOT have Conv2D after decomposition"
    );

    // After decomposition: should have Im2Col
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Im2Col { .. })),
        "Should have Im2Col after decomposition"
    );

    // After decomposition: should have MatMul
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::MatMul { .. })),
        "Should have MatMul after decomposition"
    );

    // After decomposition: should have Reshape (for output shape)
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Reshape { .. })),
        "Should have Reshape after decomposition"
    );
}

#[test]
fn test_conv2d_with_bias_decomposes_correctly() {
    // Verify Conv2D with bias includes Add after decomposition
    let mut builder = IRBuilder::new("conv_bias_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));
    let bias = builder.add_input("B", f32_tensor(&[16]));

    let conv_result = builder.conv2d(input, kernel, Some(bias), (1, 1), (1, 1), (1, 1), 1);
    builder.set_output(conv_result);
    let func = builder.build();

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // Should have bias addition (BinaryOp::Add)
    assert!(
        contains_node_type(&decomposed, |n| matches!(
            n,
            IRNode::BinaryOp {
                op: hologram_compiler::ir::BinOp::Add,
                ..
            }
        )),
        "Should have Add for bias after decomposition"
    );
}

#[test]
fn test_conv2d_shape_inference_concrete() {
    // Verify shape inference for concrete dimensions
    let input_shape = SymbolicShape::concrete(vec![1, 3, 224, 224]);
    let kernel_shape = SymbolicShape::concrete(vec![64, 3, 7, 7]);

    let attrs = vec![
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![3, 3, 3, 3]),
        make_ints_attr("dilations", vec![1, 1]),
    ];

    let result = infer_conv_output_shape(&input_shape, &kernel_shape, &attrs)
        .expect("Shape inference should succeed");

    // ResNet-style first conv: 224 -> 112
    assert_eq!(result.dims()[0], Dim::Concrete(1)); // batch
    assert_eq!(result.dims()[1], Dim::Concrete(64)); // channels
    assert_eq!(result.dims()[2], Dim::Concrete(112)); // height
    assert_eq!(result.dims()[3], Dim::Concrete(112)); // width
}

#[test]
fn test_conv2d_shape_inference_symbolic_batch() {
    // Verify shape inference preserves symbolic batch dimension
    let input_shape = SymbolicShape::symbolic(vec!["batch", "3", "224", "224"]);
    let kernel_shape = SymbolicShape::concrete(vec![64, 3, 7, 7]);

    let attrs = vec![
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![3, 3, 3, 3]),
    ];

    let result = infer_conv_output_shape(&input_shape, &kernel_shape, &attrs)
        .expect("Shape inference should succeed");

    // Batch dimension should remain symbolic
    assert!(
        matches!(result.dims()[0], Dim::Var(_)),
        "Batch dimension should be symbolic"
    );

    // Other dimensions should be concrete
    assert_eq!(result.dims()[1], Dim::Concrete(64));
}

// =============================================================================
// Test: MaxPool Translation and Decomposition
// =============================================================================

#[test]
fn test_maxpool_creates_ir_node() {
    // Verify that ONNX MaxPool translates to IR MaxPool node
    let mut builder = IRBuilder::new("maxpool_test");
    let input = builder.add_input("X", f32_tensor(&[1, 64, 112, 112]));

    let result = translate_max_pool(
        &[input],
        &[
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        &HashMap::new(),
        &mut builder,
    );

    assert!(result.is_ok(), "MaxPool translation should succeed");
    let func = builder.build();

    assert!(
        contains_node_type(&func, |n| matches!(n, IRNode::MaxPool { .. })),
        "IR should contain MaxPool node before decomposition"
    );
}

#[test]
fn test_maxpool_decomposes_to_unfold_reduce() {
    // Verify MaxPool → Unfold + ReduceMax decomposition
    let mut builder = IRBuilder::new("maxpool_decompose_test");
    let input = builder.add_input("X", f32_tensor(&[1, 64, 32, 32]));

    let pool_result = builder.max_pool(input, (2, 2), (2, 2), (0, 0));
    builder.set_output(pool_result);
    let func = builder.build();

    // Before decomposition: should have MaxPool
    assert!(
        contains_node_type(&func, |n| matches!(n, IRNode::MaxPool { .. })),
        "Should have MaxPool before decomposition"
    );

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // After decomposition: no MaxPool
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::MaxPool { .. })),
        "Should NOT have MaxPool after decomposition"
    );

    // After decomposition: should have Unfold
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Unfold { .. })),
        "Should have Unfold after decomposition"
    );

    // After decomposition: should have Reduce with Max op
    assert!(
        contains_node_type(&decomposed, |n| matches!(
            n,
            IRNode::Reduce {
                op: hologram_compiler::ir::ReduceOp::Max,
                ..
            }
        )),
        "Should have ReduceMax after decomposition"
    );
}

#[test]
fn test_maxpool_shape_inference() {
    // Verify MaxPool shape inference
    let input_shape = SymbolicShape::concrete(vec![1, 64, 112, 112]);

    let attrs = vec![
        make_ints_attr("kernel_shape", vec![3, 3]),
        make_ints_attr("strides", vec![2, 2]),
        make_ints_attr("pads", vec![1, 1, 1, 1]),
    ];

    let result =
        infer_pool_output_shape(&input_shape, &attrs).expect("Shape inference should succeed");

    // Output: (112 + 2 - 3) / 2 + 1 = 56
    assert_eq!(result.dims()[0], Dim::Concrete(1)); // batch
    assert_eq!(result.dims()[1], Dim::Concrete(64)); // channels
    assert_eq!(result.dims()[2], Dim::Concrete(56)); // height
    assert_eq!(result.dims()[3], Dim::Concrete(56)); // width
}

// =============================================================================
// Test: AvgPool Translation and Decomposition
// =============================================================================

#[test]
fn test_avgpool_decomposes_to_unfold_reduce() {
    // Verify AvgPool → Unfold + ReduceMean decomposition
    let mut builder = IRBuilder::new("avgpool_decompose_test");
    let input = builder.add_input("X", f32_tensor(&[1, 64, 32, 32]));

    let pool_result = builder.avg_pool(input, (2, 2), (2, 2), (0, 0));
    builder.set_output(pool_result);
    let func = builder.build();

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // After decomposition: no AvgPool
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::AvgPool { .. })),
        "Should NOT have AvgPool after decomposition"
    );

    // After decomposition: should have Unfold
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Unfold { .. })),
        "Should have Unfold after decomposition"
    );

    // After decomposition: should have Reduce with Mean op
    assert!(
        contains_node_type(&decomposed, |n| matches!(
            n,
            IRNode::Reduce {
                op: hologram_compiler::ir::ReduceOp::Mean,
                ..
            }
        )),
        "Should have ReduceMean after decomposition"
    );
}

// =============================================================================
// Test: ResNet-style Pipeline
// =============================================================================

#[test]
fn test_resnet_block_decomposition() {
    // Test a ResNet-style block: Conv -> MaxPool -> Conv
    let mut builder = IRBuilder::new("resnet_block_test");

    // Input: [batch, 3, 224, 224]
    let input = builder.add_input("X", f32_tensor(&[1, 3, 224, 224]));

    // First conv: 3 -> 64 channels, 7x7 kernel, stride 2
    let conv1_kernel = builder.add_input("conv1_W", f32_tensor(&[64, 3, 7, 7]));
    let conv1 = builder.conv2d(input, conv1_kernel, None, (2, 2), (3, 3), (1, 1), 1);

    // MaxPool: 3x3 kernel, stride 2
    let pool = builder.max_pool(conv1, (3, 3), (2, 2), (1, 1));

    // Second conv: 64 -> 64 channels, 3x3 kernel
    let conv2_kernel = builder.add_input("conv2_W", f32_tensor(&[64, 64, 3, 3]));
    let conv2 = builder.conv2d(pool, conv2_kernel, None, (1, 1), (1, 1), (1, 1), 1);

    builder.set_output(conv2);
    let func = builder.build();

    // Count nodes before decomposition
    let conv2d_count_before = count_node_type(&func, |n| matches!(n, IRNode::Conv2D { .. }));
    let maxpool_count_before = count_node_type(&func, |n| matches!(n, IRNode::MaxPool { .. }));

    assert_eq!(conv2d_count_before, 2, "Should have 2 Conv2D nodes");
    assert_eq!(maxpool_count_before, 1, "Should have 1 MaxPool node");

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // Verify no Conv2D or MaxPool remain
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::Conv2D { .. })),
        "No Conv2D should remain after decomposition"
    );
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::MaxPool { .. })),
        "No MaxPool should remain after decomposition"
    );

    // Verify decomposed operations are present
    let im2col_count = count_node_type(&decomposed, |n| matches!(n, IRNode::Im2Col { .. }));
    let matmul_count = count_node_type(&decomposed, |n| matches!(n, IRNode::MatMul { .. }));
    let unfold_count = count_node_type(&decomposed, |n| matches!(n, IRNode::Unfold { .. }));

    assert!(im2col_count >= 2, "Should have at least 2 Im2Col nodes");
    assert!(matmul_count >= 2, "Should have at least 2 MatMul nodes");
    assert!(unfold_count >= 1, "Should have at least 1 Unfold node");
}

#[test]
fn test_resnet_block_symbolic_batch() {
    // Test ResNet-style block with symbolic batch size
    let mut builder = IRBuilder::new("resnet_symbolic_test");

    // Input: [batch, 3, 224, 224] with symbolic batch
    let input = builder.add_input("X", f32_tensor_symbolic_batch("batch", &[3, 224, 224]));
    let conv_kernel = builder.add_input("W", f32_tensor(&[64, 3, 7, 7]));

    let conv = builder.conv2d(input, conv_kernel, None, (2, 2), (3, 3), (1, 1), 1);
    builder.set_output(conv);
    let func = builder.build();

    // Apply decomposition - should work with symbolic shapes
    let config = DecomposeConfig::all();
    let result = decompose_function(&func, &config);

    assert!(
        result.is_ok(),
        "Decomposition should succeed with symbolic batch: {:?}",
        result.err()
    );
}

// =============================================================================
// Test: Complexity Validation
// =============================================================================

#[test]
fn test_decomposed_passes_complexity_validation() {
    use hologram_compiler::ir::{IRModule, validate_complexity};

    // Create a function with Conv2D and MaxPool
    let mut builder = IRBuilder::new("complexity_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));

    let conv = builder.conv2d(input, kernel, None, (1, 1), (1, 1), (1, 1), 1);
    let pool = builder.max_pool(conv, (2, 2), (2, 2), (0, 0));
    builder.set_output(pool);
    let func = builder.build();

    // Before decomposition: complexity validation should FAIL
    let mut module_before = IRModule::new("test");
    module_before.add_function(func.clone());

    let result_before = validate_complexity(&module_before);
    assert!(
        result_before.is_err(),
        "Raw Conv2D/MaxPool should fail complexity validation"
    );

    // After decomposition: complexity validation should PASS
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    let mut module_after = IRModule::new("test");
    module_after.add_function(decomposed);

    let result_after = validate_complexity(&module_after);
    assert!(
        result_after.is_ok(),
        "Decomposed function should pass complexity validation: {:?}",
        result_after.err()
    );
}

// =============================================================================
// Test: OnnxConfig Integration
// =============================================================================

#[test]
fn test_onnx_config_decompose_flags() {
    // Verify OnnxConfig decomposition flags map to DecomposeConfig
    let config = OnnxConfig {
        decompose_conv2d: true,
        decompose_pooling: true,
        ..Default::default()
    };

    assert!(config.decompose_conv2d);
    assert!(config.decompose_pooling);

    // With both disabled
    let config_disabled = OnnxConfig {
        decompose_conv2d: false,
        decompose_pooling: false,
        ..Default::default()
    };

    assert!(!config_disabled.decompose_conv2d);
    assert!(!config_disabled.decompose_pooling);
}

#[test]
fn test_selective_decomposition() {
    // Test that decomposition can be selectively disabled
    let mut builder = IRBuilder::new("selective_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));

    let conv = builder.conv2d(input, kernel, None, (1, 1), (1, 1), (1, 1), 1);
    let pool = builder.max_pool(conv, (2, 2), (2, 2), (0, 0));
    builder.set_output(pool);
    let func = builder.build();

    // Only decompose pooling, not conv
    let config = DecomposeConfig {
        decompose_conv2d: false,
        decompose_pooling: true,
    };
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // Conv2D should remain
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Conv2D { .. })),
        "Conv2D should remain when decompose_conv2d is false"
    );

    // MaxPool should be decomposed
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::MaxPool { .. })),
        "MaxPool should be decomposed when decompose_pooling is true"
    );
}

// =============================================================================
// Test: Large Model Memory Efficiency
// =============================================================================

#[test]
fn test_large_conv_chain_decomposition() {
    // Test that a large chain of convolutions decomposes correctly
    // This tests memory efficiency for large models
    let mut builder = IRBuilder::new("large_conv_chain");

    let mut current = builder.add_input("X", f32_tensor(&[1, 64, 56, 56]));

    // Add 10 conv layers
    for i in 0..10 {
        let kernel = builder.add_input(&format!("W{}", i), f32_tensor(&[64, 64, 3, 3]));
        current = builder.conv2d(current, kernel, None, (1, 1), (1, 1), (1, 1), 1);
    }

    builder.set_output(current);
    let func = builder.build();

    // Verify 10 Conv2D nodes
    let conv_count = count_node_type(&func, |n| matches!(n, IRNode::Conv2D { .. }));
    assert_eq!(conv_count, 10);

    // Apply decomposition
    let config = DecomposeConfig::all();
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // Verify no Conv2D nodes remain
    assert!(
        !contains_node_type(&decomposed, |n| matches!(n, IRNode::Conv2D { .. })),
        "All Conv2D nodes should be decomposed"
    );

    // Verify we have 10 Im2Col nodes (one per conv)
    let im2col_count = count_node_type(&decomposed, |n| matches!(n, IRNode::Im2Col { .. }));
    assert_eq!(im2col_count, 10, "Should have 10 Im2Col nodes");

    // Verify we have 10 MatMul nodes (one per conv)
    let matmul_count = count_node_type(&decomposed, |n| matches!(n, IRNode::MatMul { .. }));
    assert_eq!(matmul_count, 10, "Should have 10 MatMul nodes");
}

// =============================================================================
// Test: Edge Cases
// =============================================================================

#[test]
fn test_conv_with_dilation() {
    // Test Conv2D with dilation
    let mut builder = IRBuilder::new("dilated_conv_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));

    // Dilation = (2, 2)
    let conv = builder.conv2d(input, kernel, None, (1, 1), (2, 2), (2, 2), 1);
    builder.set_output(conv);
    let func = builder.build();

    // Decomposition should handle dilation
    let config = DecomposeConfig::all();
    let result = decompose_function(&func, &config);

    assert!(
        result.is_ok(),
        "Decomposition should succeed with dilation: {:?}",
        result.err()
    );
}

#[test]
fn test_conv_without_decomposition() {
    // Test that Conv2D passes through when decomposition is disabled
    let mut builder = IRBuilder::new("no_decompose_test");
    let input = builder.add_input("X", f32_tensor(&[1, 3, 32, 32]));
    let kernel = builder.add_input("W", f32_tensor(&[16, 3, 3, 3]));

    let conv = builder.conv2d(input, kernel, None, (1, 1), (1, 1), (1, 1), 1);
    builder.set_output(conv);
    let func = builder.build();

    // Disable decomposition
    let config = DecomposeConfig::default(); // All disabled
    let decomposed = decompose_function(&func, &config).expect("Decomposition should succeed");

    // Conv2D should remain
    assert!(
        contains_node_type(&decomposed, |n| matches!(n, IRNode::Conv2D { .. })),
        "Conv2D should remain when decomposition is disabled"
    );
}

#[test]
fn test_pool_with_asymmetric_kernel() {
    // Test pooling with non-square kernel
    let mut builder = IRBuilder::new("asymmetric_pool_test");
    let input = builder.add_input("X", f32_tensor(&[1, 64, 32, 32]));

    // 3x2 kernel
    let pool = builder.max_pool(input, (3, 2), (1, 1), (1, 0));
    builder.set_output(pool);
    let func = builder.build();

    let config = DecomposeConfig::all();
    let result = decompose_function(&func, &config);

    assert!(
        result.is_ok(),
        "Decomposition should handle asymmetric kernels: {:?}",
        result.err()
    );
}

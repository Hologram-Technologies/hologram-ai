/// Standalone test case for hologram team to reproduce ResNet18 shape mismatch bug
///
/// This test creates a minimal graph that mimics ResNet18's final layers:
/// GlobalAveragePool → Flatten → MatMul (+ bias)
///
/// Expected to compile successfully, but currently fails with:
/// ShapeMismatch { node_id: 273, expected: [1000], actual: [1, 512] }

use hologram::compiler::{compile, CompilerConfig, DType, OpKind, OpNode, OperationGraph};

#[test]
fn test_resnet18_final_layers() {
    let mut graph = OperationGraph::default();

    // 1. Input after all Conv layers: [1, 512, 7, 7]
    let input = OpNode::new(0, OpKind::Input, vec![1, 512, 7, 7], DType::F32);
    graph.nodes.push(input);
    graph.inputs.push(0);

    // 2. GlobalAveragePool: [1, 512, 7, 7] → [1, 512, 1, 1]
    let mut gap = OpNode::new(1, OpKind::GlobalAveragePool, vec![1, 512, 1, 1], DType::F32);
    gap.inputs = vec![0];
    graph.nodes.push(gap);

    // 3. Flatten (axis=1): [1, 512, 1, 1] → [1, 512]
    // Everything before axis=1 (dim 0) stays: 1
    // Everything from axis=1 onwards flattens: 512 * 1 * 1 = 512
    // Output: [1, 512]
    let mut flatten = OpNode::new(2, OpKind::Flatten { start_dim: 1 }, vec![1, 512], DType::F32);
    flatten.inputs = vec![1];
    graph.nodes.push(flatten);

    // 4. Weight constant: [512, 1000]
    let weight = OpNode::new(3, OpKind::Constant, vec![512, 1000], DType::F32);
    graph.nodes.push(weight);

    // 5. MatMul: [1, 512] × [512, 1000] → [1, 1000]
    let mut matmul = OpNode::new(
        4,
        OpKind::MatMul { m: 1, k: 512, n: 1000 },
        vec![1, 1000],
        DType::F32,
    );
    matmul.inputs = vec![2, 3];
    graph.nodes.push(matmul);

    // 6. Bias constant: [1000]
    let bias = OpNode::new(5, OpKind::Constant, vec![1000], DType::F32);
    graph.nodes.push(bias);

    // 7. Add: [1, 1000] + [1000] → [1, 1000] (with broadcasting)
    let mut add = OpNode::new(6, OpKind::Add, vec![1, 1000], DType::F32);
    add.inputs = vec![4, 5];
    graph.nodes.push(add);

    graph.outputs.push(6);

    // Compile
    let config = CompilerConfig::default();
    let result = compile(&graph, &config);

    match result {
        Ok(plan) => {
            println!("✅ Compilation successful!");
            println!("   Plan has {} ops", plan.ops.len());
        }
        Err(e) => {
            println!("❌ Compilation failed: {:?}", e);

            // This is the expected error we're debugging:
            // ShapeMismatch { node_id: X, expected: [1000], actual: [1, 512] }

            panic!("Compilation failed. This test demonstrates the ResNet18 shape mismatch bug.");
        }
    }
}

#[test]
fn test_flatten_shape_preservation() {
    // Test hypothesis 1: Does Flatten preserve dimensions before start_dim?
    let mut graph = OperationGraph::default();

    // Input: [1, 512, 1, 1]
    let input = OpNode::new(0, OpKind::Input, vec![1, 512, 1, 1], DType::F32);
    graph.nodes.push(input);
    graph.inputs.push(0);

    // Flatten with start_dim=1: Should produce [1, 512]
    let mut flatten = OpNode::new(1, OpKind::Flatten { start_dim: 1 }, vec![1, 512], DType::F32);
    flatten.inputs = vec![0];
    graph.nodes.push(flatten);

    graph.outputs.push(1);

    let config = CompilerConfig::default();
    let result = compile(&graph, &config);

    match result {
        Ok(_) => println!("✅ Flatten preserves prefix dimensions correctly"),
        Err(e) => {
            println!("❌ Flatten bug: Expected output [1, 512] but got error: {:?}", e);
            // If this fails, Flatten is likely producing [512] instead of [1, 512]
        }
    }
}

#[test]
fn test_matmul_with_batch_size_one() {
    // Test hypothesis 2: Does MatMul keep batch dimension when m=1?
    let mut graph = OperationGraph::default();

    // Input A: [1, 512]
    let input_a = OpNode::new(0, OpKind::Input, vec![1, 512], DType::F32);
    graph.nodes.push(input_a);
    graph.inputs.push(0);

    // Input B: [512, 1000]
    let input_b = OpNode::new(1, OpKind::Constant, vec![512, 1000], DType::F32);
    graph.nodes.push(input_b);

    // MatMul: [1, 512] × [512, 1000] → Should produce [1, 1000]
    let mut matmul = OpNode::new(
        2,
        OpKind::MatMul { m: 1, k: 512, n: 1000 },
        vec![1, 1000],
        DType::F32,
    );
    matmul.inputs = vec![0, 1];
    graph.nodes.push(matmul);

    graph.outputs.push(2);

    let config = CompilerConfig::default();
    let result = compile(&graph, &config);

    match result {
        Ok(_) => println!("✅ MatMul preserves batch dimension when m=1"),
        Err(e) => {
            println!("❌ MatMul bug: Expected output [1, 1000] but got error: {:?}", e);
            // If this fails, MatMul is likely producing [1000] instead of [1, 1000]
        }
    }
}

#[test]
fn test_add_broadcasting() {
    // Test hypothesis 3: Does Add support broadcasting [1, n] + [n]?
    let mut graph = OperationGraph::default();

    // Input A: [1, 1000]
    let input_a = OpNode::new(0, OpKind::Input, vec![1, 1000], DType::F32);
    graph.nodes.push(input_a);
    graph.inputs.push(0);

    // Input B (bias): [1000]
    let bias = OpNode::new(1, OpKind::Constant, vec![1000], DType::F32);
    graph.nodes.push(bias);

    // Add: [1, 1000] + [1000] → Should produce [1, 1000] with broadcasting
    let mut add = OpNode::new(2, OpKind::Add, vec![1, 1000], DType::F32);
    add.inputs = vec![0, 1];
    graph.nodes.push(add);

    graph.outputs.push(2);

    let config = CompilerConfig::default();
    let result = compile(&graph, &config);

    match result {
        Ok(_) => println!("✅ Add supports broadcasting [1, n] + [n]"),
        Err(e) => {
            println!("❌ Add broadcasting bug: Expected [1, 1000] + [1000] → [1, 1000] but got error: {:?}", e);
            // If this fails, Add doesn't support broadcasting and expects exact shape match
        }
    }
}

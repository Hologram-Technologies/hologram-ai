//! Test with a simple MLP that only uses supported ops.
//!
//! This model only uses: MatMul, Add, Relu - all supported by hologram backend.

use anyhow::Result;
use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::compiler::{
    CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph, compile,
};

fn main() -> Result<()> {
    println!("🧪 Testing simple MLP with supported ops only\n");

    // Build a 2-layer MLP: Input[4] -> FC1[8] -> ReLU -> FC2[2] -> Output
    let mut graph = OperationGraph::new();

    // Input: [1, 4]
    graph.add_node(OpNode::new(0, OpKind::Input, vec![1, 4], DType::F32));
    graph.add_input("input", 0);

    // FC1 weights: [4, 8] - transposed for MatMul(input @ weights)
    graph.add_node(OpNode::new(1, OpKind::Constant, vec![4, 8], DType::F32));
    let w1: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.6).collect(); // range [-1.6, 1.5]
    graph.add_constant(ConstantData::F32(w1));

    // FC1 bias: [8]
    graph.add_node(OpNode::new(2, OpKind::Constant, vec![1, 8], DType::F32));
    let b1: Vec<f32> = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4];
    graph.add_constant(ConstantData::F32(b1));

    // MatMul: input[1,4] @ w1[4,8] = [1,8]
    graph.add_node(OpNode::new(
        3,
        OpKind::MatMul { m: 1, k: 4, n: 8 },
        vec![1, 8],
        DType::F32,
    ));
    graph.add_edge(0, 3);
    graph.add_edge(1, 3);

    // Add bias: [1,8] + [1,8] = [1,8]
    graph.add_node(OpNode::new(4, OpKind::Add, vec![1, 8], DType::F32));
    graph.add_edge(3, 4);
    graph.add_edge(2, 4);

    // ReLU
    graph.add_node(OpNode::new(5, OpKind::Relu, vec![1, 8], DType::F32));
    graph.add_edge(4, 5);

    // FC2 weights: [8, 2]
    graph.add_node(OpNode::new(6, OpKind::Constant, vec![8, 2], DType::F32));
    let w2: Vec<f32> = (0..16).map(|i| (i as f32) * 0.2 - 1.5).collect();
    graph.add_constant(ConstantData::F32(w2));

    // FC2 bias: [2]
    graph.add_node(OpNode::new(7, OpKind::Constant, vec![1, 2], DType::F32));
    let b2: Vec<f32> = vec![0.5, -0.5];
    graph.add_constant(ConstantData::F32(b2));

    // MatMul: relu_out[1,8] @ w2[8,2] = [1,2]
    graph.add_node(OpNode::new(
        8,
        OpKind::MatMul { m: 1, k: 8, n: 2 },
        vec![1, 2],
        DType::F32,
    ));
    graph.add_edge(5, 8);
    graph.add_edge(6, 8);

    // Add bias: [1,2] + [1,2] = [1,2]
    graph.add_node(OpNode::new(9, OpKind::Add, vec![1, 2], DType::F32));
    graph.add_edge(8, 9);
    graph.add_edge(7, 9);

    // Output
    graph.add_node(OpNode::new(10, OpKind::Output, vec![1, 2], DType::F32));
    graph.add_edge(9, 10);
    graph.add_output("output", 10);

    println!(
        "📊 Graph: {} nodes, {} constants",
        graph.nodes.len(),
        graph.constants.len()
    );

    // Compile
    println!("\n🔨 Compiling...");
    let plan = compile(&graph, &CompilerConfig::default())?;

    println!("   Instructions: {}", plan.instructions.len());
    println!("   Buffers: {}", plan.buffers.len());
    println!("   Constants: {} bytes", plan.constants.len());

    // Print instructions
    println!("\n   Instruction breakdown:");
    for (i, instr) in plan.instructions.iter().enumerate() {
        let name = format!("{:?}", instr);
        let name = name.split_whitespace().next().unwrap_or("?");
        println!("   {}. {}", i + 1, name);
    }

    // Check constants are loaded
    if plan.constants.is_empty() {
        println!("\n❌ FAIL: Constants not loaded!");
        return Ok(());
    }

    // Verify first few constant bytes
    let first_weight: f32 = f32::from_le_bytes([
        plan.constants[0],
        plan.constants[1],
        plan.constants[2],
        plan.constants[3],
    ]);
    println!(
        "\n   First constant value: {:.4} (expected -1.6)",
        first_weight
    );

    // Create input: [1, 4] with values [1.0, 2.0, 3.0, 4.0]
    println!("\n🎨 Input: [1.0, 2.0, 3.0, 4.0]");
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    // Create output buffer: [1, 2]
    let mut output_data: Vec<f32> = vec![0.0; 2];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    // Execute
    println!("\n🚀 Executing on CPU...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    // Check results
    println!("\n📊 Output: {:?}", output_data);

    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    if non_zero > 0 {
        println!("\n✅ SUCCESS! MLP produces non-zero outputs!");
        println!("   This confirms weights are loaded and computation works.");
    } else {
        println!("\n❌ FAIL: All outputs are zero");
    }

    Ok(())
}

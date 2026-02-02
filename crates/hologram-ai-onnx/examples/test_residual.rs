//! Test residual block structure like ResNet.
//! Conv -> BN -> ReLU -> Conv -> BN -> Add(skip) -> ReLU

use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::compiler::{
    CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph, compile,
};

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing ResNet-style residual block\n");

    let mut graph = OperationGraph::new();

    // Input: [1, 64, 56, 56] (after initial conv+pool in ResNet)
    let spatial = 56;
    let channels = 64;

    // Node 0: Input
    graph.add_node(OpNode::new(
        0,
        OpKind::Input,
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_input("input", 0);

    // === First Conv3x3 ===
    // Node 1: Conv weights [64, 64, 3, 3]
    graph.add_node(OpNode::new(
        1,
        OpKind::Constant,
        vec![channels, channels, 3, 3],
        DType::F32,
    ));
    graph.add_constant(ConstantData::F32(vec![0.01; channels * channels * 9]));

    // Node 2: Conv output
    graph.add_node(OpNode::new(
        2,
        OpKind::Conv2d {
            kernel: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 1,
        },
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(0, 2);
    graph.add_edge(1, 2);

    // === BatchNorm 1 ===
    // Nodes 3-6: BN params (scale, bias, mean, var)
    for i in 3..=6 {
        graph.add_node(OpNode::new(i, OpKind::Constant, vec![channels], DType::F32));
    }
    graph.add_constant(ConstantData::F32(vec![1.0; channels])); // scale
    graph.add_constant(ConstantData::F32(vec![0.0; channels])); // bias
    graph.add_constant(ConstantData::F32(vec![0.0; channels])); // mean
    graph.add_constant(ConstantData::F32(vec![1.0; channels])); // var

    // Node 7: BN output
    graph.add_node(OpNode::new(
        7,
        OpKind::BatchNormalization { epsilon: 1e-5 },
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(2, 7);
    graph.add_edge(3, 7);
    graph.add_edge(4, 7);
    graph.add_edge(5, 7);
    graph.add_edge(6, 7);

    // === ReLU 1 ===
    // Node 8: ReLU
    graph.add_node(OpNode::new(
        8,
        OpKind::Relu,
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(7, 8);

    // === Second Conv3x3 ===
    // Node 9: Conv weights
    graph.add_node(OpNode::new(
        9,
        OpKind::Constant,
        vec![channels, channels, 3, 3],
        DType::F32,
    ));
    graph.add_constant(ConstantData::F32(vec![0.01; channels * channels * 9]));

    // Node 10: Conv output
    graph.add_node(OpNode::new(
        10,
        OpKind::Conv2d {
            kernel: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 1,
        },
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(8, 10);
    graph.add_edge(9, 10);

    // === BatchNorm 2 ===
    // Nodes 11-14: BN params
    for i in 11..=14 {
        graph.add_node(OpNode::new(i, OpKind::Constant, vec![channels], DType::F32));
    }
    graph.add_constant(ConstantData::F32(vec![1.0; channels]));
    graph.add_constant(ConstantData::F32(vec![0.0; channels]));
    graph.add_constant(ConstantData::F32(vec![0.0; channels]));
    graph.add_constant(ConstantData::F32(vec![1.0; channels]));

    // Node 15: BN output
    graph.add_node(OpNode::new(
        15,
        OpKind::BatchNormalization { epsilon: 1e-5 },
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(10, 15);
    graph.add_edge(11, 15);
    graph.add_edge(12, 15);
    graph.add_edge(13, 15);
    graph.add_edge(14, 15);

    // === Add (residual connection) ===
    // Node 16: Add input + bn2_out
    graph.add_node(OpNode::new(
        16,
        OpKind::Add,
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(0, 16); // skip connection from input
    graph.add_edge(15, 16); // from bn2

    // === ReLU 2 ===
    // Node 17: ReLU
    graph.add_node(OpNode::new(
        17,
        OpKind::Relu,
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(16, 17);

    // === Output ===
    // Node 18: Output
    graph.add_node(OpNode::new(
        18,
        OpKind::Output,
        vec![1, channels, spatial, spatial],
        DType::F32,
    ));
    graph.add_edge(17, 18);
    graph.add_output("output", 18);

    println!("📊 Graph: {} nodes", graph.nodes.len());

    let plan = compile(&graph, &CompilerConfig::default())?;
    println!("   Instructions: {}", plan.instructions.len());
    println!("   Buffers: {}", plan.buffers.len());
    println!("   Constants: {} bytes", plan.constants.len());

    // Print instructions
    println!("\n📋 Instructions:");
    for (i, instr) in plan.instructions.iter().enumerate() {
        let s = format!("{:?}", instr);
        let s = if s.len() > 70 {
            format!("{}...", &s[..70])
        } else {
            s
        };
        println!("   {}. {}", i + 1, s);
    }

    // Execute
    let input_size = channels * spatial * spatial;
    let input_data: Vec<f32> = vec![1.0; input_size];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut output_data: Vec<f32> = vec![0.0; input_size];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    println!("\n🚀 Executing...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    let max_val = output_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("\n📊 Output:");
    println!("   Non-zero: {}/{}", non_zero, input_size);
    println!("   Range: [{:.6}, {:.6}]", min_val, max_val);
    println!("   First 5: {:?}", &output_data[..5]);

    if non_zero > 0 {
        println!("\n✅ Residual block works!");
    } else {
        println!("\n❌ FAIL: All zeros");
    }

    Ok(())
}

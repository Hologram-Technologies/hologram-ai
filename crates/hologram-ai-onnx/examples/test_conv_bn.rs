//! Test Conv2d -> BatchNorm chain with ResNet18 dimensions.

use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::compiler::{
    CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph, compile,
};

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Conv2d -> BatchNorm chain\n");

    let mut graph = OperationGraph::new();

    // Node 0: Input [1, 3, 224, 224]
    graph.add_node(OpNode::new(
        0,
        OpKind::Input,
        vec![1, 3, 224, 224],
        DType::F32,
    ));
    graph.add_input("input", 0);

    // Node 1: Conv weights [64, 3, 7, 7]
    graph.add_node(OpNode::new(
        1,
        OpKind::Constant,
        vec![64, 3, 7, 7],
        DType::F32,
    ));
    graph.add_constant(ConstantData::F32(vec![0.01; 64 * 3 * 7 * 7]));

    // Node 2: Conv2d output [1, 64, 112, 112]
    graph.add_node(OpNode::new(
        2,
        OpKind::Conv2d {
            kernel: (7, 7),
            stride: (2, 2),
            padding: (3, 3),
            dilation: (1, 1),
            groups: 1,
        },
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(0, 2);
    graph.add_edge(1, 2);

    // BatchNorm parameters for 64 channels
    // Node 3: scale [64] = all 1.0
    graph.add_node(OpNode::new(3, OpKind::Constant, vec![64], DType::F32));
    graph.add_constant(ConstantData::F32(vec![1.0; 64]));

    // Node 4: bias [64] = all 0.0
    graph.add_node(OpNode::new(4, OpKind::Constant, vec![64], DType::F32));
    graph.add_constant(ConstantData::F32(vec![0.0; 64]));

    // Node 5: mean [64] = all 0.0
    graph.add_node(OpNode::new(5, OpKind::Constant, vec![64], DType::F32));
    graph.add_constant(ConstantData::F32(vec![0.0; 64]));

    // Node 6: var [64] = all 1.0
    graph.add_node(OpNode::new(6, OpKind::Constant, vec![64], DType::F32));
    graph.add_constant(ConstantData::F32(vec![1.0; 64]));

    // Node 7: BatchNorm output [1, 64, 112, 112]
    // With scale=1, bias=0, mean=0, var=1, this is identity
    graph.add_node(OpNode::new(
        7,
        OpKind::BatchNormalization { epsilon: 1e-5 },
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(2, 7); // conv output -> bn
    graph.add_edge(3, 7); // scale
    graph.add_edge(4, 7); // bias
    graph.add_edge(5, 7); // mean
    graph.add_edge(6, 7); // var

    // Node 8: Output
    graph.add_node(OpNode::new(
        8,
        OpKind::Output,
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(7, 8);
    graph.add_output("output", 8);

    println!("📊 Graph: {} nodes", graph.nodes.len());

    let plan = compile(&graph, &CompilerConfig::default())?;
    println!("   Instructions: {}", plan.instructions.len());
    println!("   Buffers: {}", plan.buffers.len());
    println!("   Constants: {} bytes", plan.constants.len());

    // Print instructions
    println!("\n📋 Instructions:");
    for (i, instr) in plan.instructions.iter().enumerate() {
        let s = format!("{:?}", instr);
        let s = if s.len() > 80 {
            format!("{}...", &s[..80])
        } else {
            s
        };
        println!("   {}. {}", i + 1, s);
    }

    // Execute
    let input_data: Vec<f32> = vec![1.0; 3 * 224 * 224];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let output_size = 64 * 112 * 112;
    let mut output_data: Vec<f32> = vec![0.0; output_size];
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
    println!("   Non-zero: {}/{}", non_zero, output_size);
    println!("   Range: [{:.6}, {:.6}]", min_val, max_val);

    if non_zero > 0 {
        println!("\n✅ Conv2d -> BatchNorm chain works!");
    } else {
        println!("\n❌ FAIL: All zeros");
    }

    Ok(())
}

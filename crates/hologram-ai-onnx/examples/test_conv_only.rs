//! Test just Conv2d with ResNet18's first layer dimensions.
//! If this fails, the bug is in Conv2d execution for these specific dimensions.

use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::compiler::{
    CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph, compile,
};

fn main() -> anyhow::Result<()> {
    println!("🧪 Testing Conv2d with ResNet18 first layer dimensions\n");

    // ResNet18 first conv: [1, 3, 224, 224] -> [1, 64, 112, 112]
    // kernel=7x7, stride=2, padding=3
    let mut graph = OperationGraph::new();

    // Node 0: Input [1, 3, 224, 224]
    graph.add_node(OpNode::new(
        0,
        OpKind::Input,
        vec![1, 3, 224, 224],
        DType::F32,
    ));
    graph.add_input("input", 0);

    // Node 1: Conv weights [64, 3, 7, 7] = 9408 floats
    graph.add_node(OpNode::new(
        1,
        OpKind::Constant,
        vec![64, 3, 7, 7],
        DType::F32,
    ));
    // Use simple weights: all 0.01
    let weights: Vec<f32> = vec![0.01; 64 * 3 * 7 * 7];
    graph.add_constant(ConstantData::F32(weights));

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
    graph.add_edge(0, 2); // input -> conv
    graph.add_edge(1, 2); // weight -> conv

    // Node 3: Output
    graph.add_node(OpNode::new(
        3,
        OpKind::Output,
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(2, 3);
    graph.add_output("output", 3);

    println!("📊 Graph: {} nodes", graph.nodes.len());

    // Compile
    let plan = compile(&graph, &CompilerConfig::default())?;
    println!("   Instructions: {}", plan.instructions.len());
    println!("   Buffers: {}", plan.buffers.len());
    println!("   Constants: {} bytes", plan.constants.len());

    // Create input: all 1.0 for easy verification
    let input_size = 1 * 3 * 224 * 224;
    let input_data: Vec<f32> = vec![1.0; input_size];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    // Create output buffer
    let output_size = 1 * 64 * 112 * 112;
    let mut output_data: Vec<f32> = vec![0.0; output_size];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    println!("\n🚀 Executing Conv2d...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    // Analyze output
    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    let max_val = output_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);

    println!("\n📊 Output analysis:");
    println!("   Non-zero: {}/{}", non_zero, output_size);
    println!("   Range: [{:.6}, {:.6}]", min_val, max_val);
    println!("   First 5: {:?}", &output_data[..5]);

    // Expected: with all-1.0 input and all-0.01 weights, each output pixel is:
    // sum over 3 channels * 7*7 kernel * 0.01 weight * 1.0 input = 3 * 49 * 0.01 = 1.47
    // But with padding, edge pixels have fewer contributing inputs.
    // Center pixels should be close to 1.47

    if non_zero > 0 {
        println!("\n✅ Conv2d works! Non-zero outputs.");
    } else {
        println!("\n❌ FAIL: All zeros. Conv2d bug confirmed.");
    }

    Ok(())
}

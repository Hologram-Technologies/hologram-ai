//! Test ResNet-like chain with Conv -> BN -> ReLU -> MaxPool -> residual blocks.

use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::compiler::{
    CompilerConfig, ConstantData, DType, OpKind, OpNode, OperationGraph, compile,
};

fn main() -> anyhow::Result<()> {
    println!("Testing ResNet-style full chain\n");

    let mut graph = OperationGraph::new();
    let mut node_id = 0u32;
    let mut next_id = || {
        let id = node_id;
        node_id += 1;
        id
    };

    // Input: [1, 3, 224, 224]
    let input_id = next_id();
    graph.add_node(OpNode::new(
        input_id,
        OpKind::Input,
        vec![1, 3, 224, 224],
        DType::F32,
    ));
    graph.add_input("input", input_id);

    // === Initial Conv7x7 stride 2 ===
    let conv1_weight = next_id();
    graph.add_node(OpNode::new(
        conv1_weight,
        OpKind::Constant,
        vec![64, 3, 7, 7],
        DType::F32,
    ));
    graph.add_constant(ConstantData::F32(vec![0.01; 64 * 3 * 7 * 7]));

    let conv1_out = next_id();
    graph.add_node(OpNode::new(
        conv1_out,
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
    graph.add_edge(input_id, conv1_out);
    graph.add_edge(conv1_weight, conv1_out);

    // === BN 1 ===
    let bn1_scale = next_id();
    let bn1_bias = next_id();
    let bn1_mean = next_id();
    let bn1_var = next_id();
    for id in [bn1_scale, bn1_bias, bn1_mean, bn1_var] {
        graph.add_node(OpNode::new(id, OpKind::Constant, vec![64], DType::F32));
    }
    graph.add_constant(ConstantData::F32(vec![1.0; 64]));
    graph.add_constant(ConstantData::F32(vec![0.0; 64]));
    graph.add_constant(ConstantData::F32(vec![0.0; 64]));
    graph.add_constant(ConstantData::F32(vec![1.0; 64]));

    let bn1_out = next_id();
    graph.add_node(OpNode::new(
        bn1_out,
        OpKind::BatchNormalization { epsilon: 1e-5 },
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(conv1_out, bn1_out);
    graph.add_edge(bn1_scale, bn1_out);
    graph.add_edge(bn1_bias, bn1_out);
    graph.add_edge(bn1_mean, bn1_out);
    graph.add_edge(bn1_var, bn1_out);

    // === ReLU ===
    let relu1 = next_id();
    graph.add_node(OpNode::new(
        relu1,
        OpKind::Relu,
        vec![1, 64, 112, 112],
        DType::F32,
    ));
    graph.add_edge(bn1_out, relu1);

    // === MaxPool 3x3 stride 2 ===
    let maxpool = next_id();
    graph.add_node(OpNode::new(
        maxpool,
        OpKind::MaxPool {
            kernel: (3, 3),
            stride: (2, 2),
            padding: (1, 1),
        },
        vec![1, 64, 56, 56],
        DType::F32,
    ));
    graph.add_edge(relu1, maxpool);

    let mut prev_output = maxpool;
    let spatial = 56;
    let channels = 64;

    // === 8 residual blocks (like ResNet18) ===
    for _block in 0..8 {
        // Conv 1
        let conv_w1 = next_id();
        graph.add_node(OpNode::new(
            conv_w1,
            OpKind::Constant,
            vec![channels, channels, 3, 3],
            DType::F32,
        ));
        graph.add_constant(ConstantData::F32(vec![0.01; channels * channels * 9]));

        let conv1 = next_id();
        graph.add_node(OpNode::new(
            conv1,
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
        graph.add_edge(prev_output, conv1);
        graph.add_edge(conv_w1, conv1);

        // BN 1
        let s = next_id();
        let b = next_id();
        let m = next_id();
        let v = next_id();
        for id in [s, b, m, v] {
            graph.add_node(OpNode::new(
                id,
                OpKind::Constant,
                vec![channels],
                DType::F32,
            ));
        }
        graph.add_constant(ConstantData::F32(vec![1.0; channels]));
        graph.add_constant(ConstantData::F32(vec![0.0; channels]));
        graph.add_constant(ConstantData::F32(vec![0.0; channels]));
        graph.add_constant(ConstantData::F32(vec![1.0; channels]));

        let bn1 = next_id();
        graph.add_node(OpNode::new(
            bn1,
            OpKind::BatchNormalization { epsilon: 1e-5 },
            vec![1, channels, spatial, spatial],
            DType::F32,
        ));
        graph.add_edge(conv1, bn1);
        graph.add_edge(s, bn1);
        graph.add_edge(b, bn1);
        graph.add_edge(m, bn1);
        graph.add_edge(v, bn1);

        let r1 = next_id();
        graph.add_node(OpNode::new(
            r1,
            OpKind::Relu,
            vec![1, channels, spatial, spatial],
            DType::F32,
        ));
        graph.add_edge(bn1, r1);

        // Conv 2
        let conv_w2 = next_id();
        graph.add_node(OpNode::new(
            conv_w2,
            OpKind::Constant,
            vec![channels, channels, 3, 3],
            DType::F32,
        ));
        graph.add_constant(ConstantData::F32(vec![0.01; channels * channels * 9]));

        let conv2 = next_id();
        graph.add_node(OpNode::new(
            conv2,
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
        graph.add_edge(r1, conv2);
        graph.add_edge(conv_w2, conv2);

        // BN 2
        let s2 = next_id();
        let b2 = next_id();
        let m2 = next_id();
        let v2 = next_id();
        for id in [s2, b2, m2, v2] {
            graph.add_node(OpNode::new(
                id,
                OpKind::Constant,
                vec![channels],
                DType::F32,
            ));
        }
        graph.add_constant(ConstantData::F32(vec![1.0; channels]));
        graph.add_constant(ConstantData::F32(vec![0.0; channels]));
        graph.add_constant(ConstantData::F32(vec![0.0; channels]));
        graph.add_constant(ConstantData::F32(vec![1.0; channels]));

        let bn2 = next_id();
        graph.add_node(OpNode::new(
            bn2,
            OpKind::BatchNormalization { epsilon: 1e-5 },
            vec![1, channels, spatial, spatial],
            DType::F32,
        ));
        graph.add_edge(conv2, bn2);
        graph.add_edge(s2, bn2);
        graph.add_edge(b2, bn2);
        graph.add_edge(m2, bn2);
        graph.add_edge(v2, bn2);

        // Add (residual)
        let add = next_id();
        graph.add_node(OpNode::new(
            add,
            OpKind::Add,
            vec![1, channels, spatial, spatial],
            DType::F32,
        ));
        graph.add_edge(prev_output, add);
        graph.add_edge(bn2, add);

        let r2 = next_id();
        graph.add_node(OpNode::new(
            r2,
            OpKind::Relu,
            vec![1, channels, spatial, spatial],
            DType::F32,
        ));
        graph.add_edge(add, r2);

        prev_output = r2;
    }

    // === GlobalAveragePool ===
    let gap = next_id();
    graph.add_node(OpNode::new(
        gap,
        OpKind::GlobalAveragePool,
        vec![1, channels, 1, 1],
        DType::F32,
    ));
    graph.add_edge(prev_output, gap);

    // === Flatten ===
    let flatten = next_id();
    graph.add_node(OpNode::new(
        flatten,
        OpKind::Flatten { start_dim: 1 },
        vec![1, channels],
        DType::F32,
    ));
    graph.add_edge(gap, flatten);

    // === Output ===
    let output_id = next_id();
    graph.add_node(OpNode::new(
        output_id,
        OpKind::Output,
        vec![1, channels],
        DType::F32,
    ));
    graph.add_edge(flatten, output_id);
    graph.add_output("output", output_id);

    println!("Graph: {} nodes", graph.nodes.len());

    let plan = compile(&graph, &CompilerConfig::default())?;
    println!("Buffers: {}", plan.buffers.len());
    println!("Instructions: {}", plan.instructions.len());
    println!("Constants: {} bytes", plan.constants.len());

    // Execute
    let input_data: Vec<f32> = vec![1.0; 3 * 224 * 224];
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut output_data: Vec<f32> = vec![0.0; channels];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    println!("\nExecuting...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    println!("Output: {}/{} non-zero", non_zero, channels);
    println!("First 5: {:?}", &output_data[..5]);

    if non_zero > 0 {
        println!("\n✅ SUCCESS!");
    } else {
        println!("\n❌ FAIL: All zeros");
    }

    Ok(())
}

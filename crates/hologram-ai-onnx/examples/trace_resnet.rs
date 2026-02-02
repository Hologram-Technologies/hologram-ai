//! Trace ResNet18 execution to find where zeros appear.

use anyhow::{Context, Result};
use hologram::BackendPlan;
use hologram::backend::cpu::BufferManager;
use hologram::backend::cpu::ExecutionContext;
use hologram::backend::cpu::dispatch_instruction;
use hologram::holo::HolbReader;
use hologram::holo::IsaInstruction;
use hologram::holo::types::BufferType;
use std::fs;

fn main() -> Result<()> {
    let onnx_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx".to_string()
    });

    println!("Compiling ResNet18 from: {}", onnx_path);
    let onnx_bytes = fs::read(&onnx_path).context("Failed to read ONNX model")?;
    let holb_bytes = hologram_ai_onnx::compile_onnx(&onnx_bytes)?;

    let reader = HolbReader::from_bytes(&holb_bytes)?;
    let plan: BackendPlan = rkyv::from_bytes(reader.graph())
        .map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))?;

    println!(
        "Plan: {} instructions, {} buffers, {} bytes constants\n",
        plan.instructions.len(),
        plan.buffers.len(),
        plan.constants.len()
    );

    // Create input data
    let input_data: Vec<f32> = (0..3 * 224 * 224)
        .map(|i| (i % 100) as f32 / 100.0)
        .collect();
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    // Create buffer manager
    let mut buffers = BufferManager::new(&plan.buffers, &[&input_bytes], &plan.constants)?;

    // Verify input buffer has data
    {
        let input_buf = buffers.read(0)?;
        let nz = input_buf.iter().filter(|&&b| b != 0).count();
        println!(
            "[INIT] Input buffer 0: {}/{} non-zero bytes",
            nz,
            input_buf.len()
        );
    }

    // Verify first few constant buffers
    println!("\n[INIT] Checking constant buffers:");
    for (i, meta) in plan.buffers.iter().enumerate() {
        if meta.buffer_type == BufferType::Constant {
            let data = buffers.read(i as u32)?;
            let nz = data.iter().filter(|&&b| b != 0).count();
            if nz == 0 {
                println!("  Buffer {} (Constant, {} bytes): ALL ZEROS!", i, meta.size);
            }
            if i < 10 || nz == 0 {
                println!(
                    "  Buffer {} (Constant, {} bytes): {}/{} non-zero",
                    i,
                    meta.size,
                    nz,
                    data.len()
                );
            }
        }
    }

    // Count instruction types
    println!("\n[STATS] Instruction type counts:");
    let mut instr_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for instr in &plan.instructions {
        let (_, op_name) = get_dst_info(instr);
        *instr_counts.entry(op_name).or_insert(0) += 1;
    }
    let mut sorted: Vec<_> = instr_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (name, count) in &sorted {
        println!("  {}: {}", name, count);
    }

    // Check for MatMul buffer - what is buffer 103?
    println!("\n[DEBUG] MatMul analysis:");
    for instr in &plan.instructions {
        if let IsaInstruction::MatMul { a, b, c, m, k, n } = instr {
            println!(
                "  MatMul: c={}, a={}, b={}, m={}, k={}, n={}",
                c, a, b, m, k, n
            );
            let b_meta = &plan.buffers[*b as usize];
            println!(
                "  Buffer {} type: {:?}, size: {}",
                b, b_meta.buffer_type, b_meta.size
            );
        }
    }

    // Analyze constant buffers - find which node ID maps to buffer 103
    println!("\n[DEBUG] node_buffer_map analysis for buffer 103:");
    for (node_id, buf_opt) in plan.node_buffer_map.iter().enumerate() {
        if let Some(buf_idx) = buf_opt {
            if *buf_idx == 103 {
                println!("  Node {} -> Buffer {}", node_id, buf_idx);
            }
        }
    }

    // Count total constant bytes expected
    let total_const_buf_bytes: usize = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Constant)
        .map(|b| b.size)
        .sum();
    let num_const_bufs = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Constant)
        .count();
    println!(
        "\n[DEBUG] Constants: {} buffers, {} total bytes, blob has {} bytes",
        num_const_bufs,
        total_const_buf_bytes,
        plan.constants.len()
    );

    // Execute with tracing
    println!(
        "\n[EXEC] Executing {} instructions:",
        plan.instructions.len()
    );

    let mut context = ExecutionContext::new(&mut buffers, &plan.node_buffer_map);

    for (i, instruction) in plan.instructions.iter().enumerate() {
        // Execute instruction
        dispatch_instruction(instruction, &mut context)?;

        // Get destination buffer info
        let (dst, op_name) = get_dst_info(instruction);

        // Check if destination buffer has non-zero data
        let dst_data = context.read_buffer(dst)?;
        let nz = dst_data.iter().filter(|&&b| b != 0).count();
        let total = dst_data.len();

        // Print first 20 or any that produce zeros
        if i < 20 || nz == 0 {
            let status = if nz == 0 { "ZEROS!" } else { "ok" };
            println!(
                "  [{:3}] {}: dst={} -> {}/{} non-zero [{}]",
                i, op_name, dst, nz, total, status
            );

            // For Conv2d and BatchNorm, show input buffer states
            if nz == 0 {
                print_input_buffers(instruction, &context);
            }
        }

        // Stop if we find first zero output (excluding Copy which might legitimately be zero)
        if nz == 0 && !matches!(instruction, IsaInstruction::Copy { .. }) {
            println!("\n  First zero-producing instruction found!");
            break;
        }
    }

    // Check final output
    println!("\n[RESULT] Checking output buffers:");
    for (i, meta) in plan.buffers.iter().enumerate() {
        if meta.buffer_type == BufferType::Output {
            let data = context.read_buffer(i as u32)?;
            let floats: &[f32] = bytemuck::cast_slice(data);
            let nz = floats.iter().filter(|&&x| x.abs() > 1e-10).count();
            println!("  Output buffer {}: {}/{} non-zero", i, nz, floats.len());
            println!("  First 5: {:?}", &floats[..5.min(floats.len())]);
        }
    }

    Ok(())
}

fn get_dst_info(instruction: &IsaInstruction) -> (u32, &'static str) {
    match instruction {
        IsaInstruction::Conv2d { dst, .. } => (*dst, "Conv2d"),
        IsaInstruction::BatchNorm { dst, .. } => (*dst, "BatchNorm"),
        IsaInstruction::Relu { dst, .. } => (*dst, "Relu"),
        IsaInstruction::MaxPool2d { dst, .. } => (*dst, "MaxPool2d"),
        IsaInstruction::GlobalAvgPool { dst, .. } => (*dst, "GlobalAvgPool"),
        IsaInstruction::Add { dst, .. } => (*dst, "Add"),
        IsaInstruction::MatMul { c, .. } => (*c, "MatMul"),
        IsaInstruction::Copy { dst, .. } => (*dst, "Copy"),
        IsaInstruction::Reshape { dst, .. } => (*dst, "Reshape"),
        IsaInstruction::Transpose { dst, .. } => (*dst, "Transpose"),
        _ => (0, "Unknown"),
    }
}

fn print_input_buffers(instruction: &IsaInstruction, ctx: &ExecutionContext) {
    match instruction {
        IsaInstruction::Conv2d {
            input,
            weight,
            bias,
            ..
        } => {
            if let Ok(data) = ctx.read_buffer(*input) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                println!(
                    "      -> input buf {}: {}/{} non-zero",
                    input,
                    nz,
                    data.len()
                );
            }
            if let Ok(data) = ctx.read_buffer(*weight) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                println!(
                    "      -> weight buf {}: {}/{} non-zero",
                    weight,
                    nz,
                    data.len()
                );
            }
            if let Some(b) = bias {
                if let Ok(data) = ctx.read_buffer(*b) {
                    let nz = data.iter().filter(|&&b| b != 0).count();
                    println!("      -> bias buf {}: {}/{} non-zero", b, nz, data.len());
                }
            }
        }
        IsaInstruction::BatchNorm {
            input,
            scale,
            bias,
            mean,
            var,
            ..
        } => {
            for (name, idx) in [
                ("input", *input),
                ("scale", *scale),
                ("bias", *bias),
                ("mean", *mean),
                ("var", *var),
            ] {
                if let Ok(data) = ctx.read_buffer(idx) {
                    let nz = data.iter().filter(|&&b| b != 0).count();
                    println!(
                        "      -> {} buf {}: {}/{} non-zero",
                        name,
                        idx,
                        nz,
                        data.len()
                    );
                }
            }
        }
        IsaInstruction::Add { a, b, .. } => {
            if let Ok(data) = ctx.read_buffer(*a) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                println!("      -> a buf {}: {}/{} non-zero", a, nz, data.len());
            }
            if let Ok(data) = ctx.read_buffer(*b) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                println!("      -> b buf {}: {}/{} non-zero", b, nz, data.len());
            }
        }
        IsaInstruction::MatMul { a, b, c, m, k, n } => {
            println!(
                "      MatMul: c={}, a={}, b={}, m={}, k={}, n={}",
                c, a, b, m, k, n
            );
            if let Ok(data) = ctx.read_buffer(*a) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                let floats: &[f32] = bytemuck::cast_slice(data);
                println!(
                    "      -> a buf {}: {}/{} non-zero, first 3: {:?}",
                    a,
                    nz,
                    data.len(),
                    &floats[..3.min(floats.len())]
                );
            }
            if let Ok(data) = ctx.read_buffer(*b) {
                let nz = data.iter().filter(|&&b| b != 0).count();
                let floats: &[f32] = bytemuck::cast_slice(data);
                println!(
                    "      -> b buf {}: {}/{} non-zero, first 3: {:?}",
                    b,
                    nz,
                    data.len(),
                    &floats[..3.min(floats.len())]
                );
            }
        }
        _ => {}
    }
}

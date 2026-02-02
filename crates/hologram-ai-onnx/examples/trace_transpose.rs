//! Trace buffer types around Transpose/MatMul
use anyhow::{Context, Result};
use hologram::BackendPlan;
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

    // Find Transpose instructions
    println!("\n[TRANSPOSE] Looking for Transpose instructions:");
    for (i, instr) in plan.instructions.iter().enumerate() {
        if let IsaInstruction::Transpose { dst, src, .. } = instr {
            let dst_meta = &plan.buffers[*dst as usize];
            let src_meta = &plan.buffers[*src as usize];
            println!("  Instr {}: Transpose src={} -> dst={}", i, src, dst);
            println!(
                "    src buf {}: {:?}, {} bytes",
                src, src_meta.buffer_type, src_meta.size
            );
            println!(
                "    dst buf {}: {:?}, {} bytes",
                dst, dst_meta.buffer_type, dst_meta.size
            );
        }
    }

    // Find MatMul instructions
    println!("\n[MATMUL] Looking for MatMul instructions:");
    for (i, instr) in plan.instructions.iter().enumerate() {
        if let IsaInstruction::MatMul { a, b, c, m, k, n } = instr {
            let a_meta = &plan.buffers[*a as usize];
            let b_meta = &plan.buffers[*b as usize];
            let c_meta = &plan.buffers[*c as usize];
            println!(
                "  Instr {}: MatMul a={}, b={}, c={}, m={}, k={}, n={}",
                i, a, b, c, m, k, n
            );
            println!(
                "    a buf {}: {:?}, {} bytes",
                a, a_meta.buffer_type, a_meta.size
            );
            println!(
                "    b buf {}: {:?}, {} bytes",
                b, b_meta.buffer_type, b_meta.size
            );
            println!(
                "    c buf {}: {:?}, {} bytes",
                c, c_meta.buffer_type, c_meta.size
            );
        }
    }

    // Check buffer 103
    println!("\n[BUFFER 103] Details:");
    if plan.buffers.len() > 103 {
        let meta = &plan.buffers[103];
        println!("  Buffer 103: {:?}, {} bytes", meta.buffer_type, meta.size);
    }

    // Find which nodes use buffer 103
    println!("\n[NODE BUFFER MAP] Entries mapping to buffer 103:");
    for (node_id, buf_opt) in plan.node_buffer_map.iter().enumerate() {
        if let Some(buf_idx) = buf_opt {
            if *buf_idx == 103 {
                println!("  Node {} -> Buffer {}", node_id, buf_idx);
            }
        }
    }

    Ok(())
}

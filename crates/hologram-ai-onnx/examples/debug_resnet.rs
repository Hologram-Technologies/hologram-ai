//! Debug ResNet18 execution - trace where zeros appear.

use anyhow::{Context, Result};
use hologram::BackendPlan;
use hologram::holo::HolbReader;
use hologram::holo::IsaInstruction;
use hologram::holo::types::BufferType;
use std::fs;

fn main() -> Result<()> {
    let onnx_path = "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx";

    println!("📦 Compiling ResNet18...");
    let onnx_bytes = fs::read(onnx_path).context("Failed to read ONNX model")?;
    let holb_bytes = hologram_ai_onnx::compile_onnx(&onnx_bytes)?;

    let reader = HolbReader::from_bytes(&holb_bytes)?;
    let plan: BackendPlan = rkyv::from_bytes(reader.graph())
        .map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))?;

    println!(
        "   Constants: {} bytes ({:.1} MB)",
        plan.constants.len(),
        plan.constants.len() as f64 / 1_000_000.0
    );
    println!("   Buffers: {}", plan.buffers.len());

    // Build constant offset map: buffer_idx -> offset in constants blob
    let mut const_offset_map: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    let mut offset = 0;
    for (i, buf) in plan.buffers.iter().enumerate() {
        if buf.buffer_type == BufferType::Constant {
            const_offset_map.insert(i, offset);
            offset += buf.size;
        }
    }

    // Helper to read f32 data from a buffer
    let read_const = |buf_idx: u32| -> Option<Vec<f32>> {
        let idx = buf_idx as usize;
        let buf = plan.buffers.get(idx)?;
        if buf.buffer_type != BufferType::Constant {
            return None;
        }
        let offset = *const_offset_map.get(&idx)?;
        let end = offset + buf.size;
        if end > plan.constants.len() {
            return None;
        }
        Some(
            plan.constants[offset..end]
                .chunks(4)
                .filter_map(|c| {
                    if c.len() == 4 {
                        Some(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    } else {
                        None
                    }
                })
                .collect(),
        )
    };

    // Show first 10 instructions
    println!("\n📋 First 10 instructions:");
    for (i, instr) in plan.instructions.iter().take(10).enumerate() {
        let s = format!("{:?}", instr);
        // Truncate for display
        let s = if s.len() > 100 {
            format!("{}...", &s[..100])
        } else {
            s
        };
        println!("   {}. {}", i + 1, s);
    }

    // Analyze first Conv2d
    println!("\n📋 First Conv2d details:");
    for instr in &plan.instructions {
        if let IsaInstruction::Conv2d {
            dst,
            input,
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            ..
        } = instr
        {
            println!(
                "   dst={}, input={}, weight={}, bias={:?}",
                dst, input, weight, bias
            );
            println!(
                "   in_ch={}, out_ch={}, kernel={}x{}",
                in_channels, out_channels, kernel_h, kernel_w
            );

            let input_buf = &plan.buffers[*input as usize];
            println!(
                "   Input buf {}: {:?}, {} bytes",
                input, input_buf.buffer_type, input_buf.size
            );

            let weight_buf = &plan.buffers[*weight as usize];
            println!(
                "   Weight buf {}: {:?}, {} bytes",
                weight, weight_buf.buffer_type, weight_buf.size
            );

            if let Some(data) = read_const(*weight) {
                let nz = data.iter().filter(|&&v| v.abs() > 1e-10).count();
                println!("   Weight first 5: {:?}", &data[..5.min(data.len())]);
                println!("   Weight non-zero: {}/{}", nz, data.len());
            }
            break;
        }
    }

    // Analyze first BatchNorm
    println!("\n📋 First BatchNorm:");
    for instr in &plan.instructions {
        if let IsaInstruction::BatchNorm {
            dst,
            input,
            scale,
            bias,
            mean,
            var,
            channels,
            spatial,
            ..
        } = instr
        {
            println!("   dst={}, input={}", dst, input);
            println!(
                "   scale={}, bias={}, mean={}, var={}",
                scale, bias, mean, var
            );
            println!("   channels={}, spatial={}", channels, spatial);

            for (name, idx) in [
                ("scale", *scale),
                ("bias", *bias),
                ("mean", *mean),
                ("var", *var),
            ] {
                let buf = &plan.buffers[idx as usize];
                print!("   {} buf {}: {:?}", name, idx, buf.buffer_type);
                if let Some(data) = read_const(idx) {
                    let nz = data.iter().filter(|&&v| v.abs() > 1e-10).count();
                    println!(
                        ", first 3: {:?}, {}/{} nz",
                        &data[..3.min(data.len())],
                        nz,
                        data.len()
                    );
                } else {
                    println!(" - NOT A CONSTANT!");
                }
            }
            break;
        }
    }

    // Check constant buffer count vs constants blob size
    let constant_buffer_count = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Constant)
        .count();
    let total_const_bytes: usize = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Constant)
        .map(|b| b.size)
        .sum();

    println!("\n📊 Constant buffer analysis:");
    println!("   Constant buffers: {}", constant_buffer_count);
    println!("   Sum of buffer sizes: {} bytes", total_const_bytes);
    println!("   Constants blob: {} bytes", plan.constants.len());

    if total_const_bytes != plan.constants.len() {
        println!("   ⚠️  MISMATCH! Sum of buffer sizes != constants blob size!");
        println!(
            "   Diff: {} bytes",
            (total_const_bytes as i64 - plan.constants.len() as i64).abs()
        );
    } else {
        println!("   ✓ Sizes match");
    }

    // Execute
    println!("\n🚀 Executing...");
    let input_data: Vec<f32> = (0..3 * 224 * 224)
        .map(|i| (i % 100) as f32 / 100.0)
        .collect();
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut output_data: Vec<f32> = vec![0.0; 1000];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    use hologram::backend::{Backend, cpu::CpuBackend};
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    println!("\n📊 Output: {}/1000 non-zero", non_zero);

    if non_zero == 0 {
        println!("❌ All zeros");
    } else {
        println!("✅ SUCCESS!");
    }

    Ok(())
}

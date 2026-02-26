//! Full ResNet18 inference with actual execution.

use anyhow::{Context, Result};
use hologram::BackendPlan;
use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::holo::HolbReader;
use std::fs;

// ImageNet normalization constants
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Deserialize BackendPlan from rkyv bytes (rkyv 0.7 API).
fn deserialize_plan(bytes: &[u8]) -> Result<BackendPlan> {
    let archived = unsafe { rkyv::archived_root::<BackendPlan>(bytes) };
    let plan: BackendPlan = rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible)?;
    Ok(plan)
}

fn main() -> Result<()> {
    let onnx_path = "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx";

    // Compile ONNX to holb
    println!("📦 Compiling ResNet18...");
    let onnx_bytes = fs::read(onnx_path).context("Failed to read ONNX model")?;
    let holb_bytes = hologram_ai_onnx::compile_onnx(&onnx_bytes)?;
    println!("   HOLB size: {} KB", holb_bytes.len() / 1024);

    // Load plan
    let reader = HolbReader::from_bytes(&holb_bytes)?;
    let plan = deserialize_plan(reader.graph())?;

    println!("   Constants: {} bytes", plan.constants.len());
    println!("   Instructions: {}", plan.instructions.len());
    println!("   Buffers: {}", plan.buffers.len());

    // Debug: Count buffer types
    use hologram::holo::types::BufferType;
    let input_count = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Input)
        .count();
    let output_count = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Output)
        .count();
    let constant_count = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Constant)
        .count();
    let workspace_count = plan
        .buffers
        .iter()
        .filter(|b| b.buffer_type == BufferType::Workspace)
        .count();
    println!(
        "   Buffer types: {} input, {} output, {} constant, {} workspace",
        input_count, output_count, constant_count, workspace_count
    );

    // Check if we have constant buffers
    if constant_count == 0 {
        println!("   ⚠️  WARNING: No constant buffers registered!");
    }

    // Check constant data sample (first few bytes)
    if plan.constants.len() >= 16 {
        let sample: Vec<f32> = (0..4)
            .map(|i| {
                let offset = i * 4;
                f32::from_le_bytes([
                    plan.constants[offset],
                    plan.constants[offset + 1],
                    plan.constants[offset + 2],
                    plan.constants[offset + 3],
                ])
            })
            .collect();
        println!("   First 4 constants (as f32): {:?}", sample);
    }

    // Debug: Print instruction types
    println!("\n   Instructions breakdown:");
    use std::collections::HashMap;
    let mut instr_counts: HashMap<String, usize> = HashMap::new();
    for instr in &plan.instructions {
        let name = format!("{:?}", instr);
        let name = name
            .split_whitespace()
            .next()
            .unwrap_or("Unknown")
            .to_string();
        *instr_counts.entry(name).or_default() += 1;
    }
    for (name, count) in &instr_counts {
        println!("   - {}: {}", name, count);
    }

    // Create input - mid-gray normalized image
    println!("\n🎨 Creating input tensor [1, 3, 224, 224]...");
    let input_data = create_sample_input();
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    // Create output buffer [1, 1000]
    let output_size = 1000;
    let mut output_data: Vec<f32> = vec![0.0; output_size];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    // Execute!
    println!("\n🚀 Running inference on CPU...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    // Analyze output
    println!("\n📊 Results:");
    let non_zero = output_data.iter().filter(|&&x| x.abs() > 1e-10).count();
    println!("   Non-zero outputs: {}/{}", non_zero, output_size);

    let max_val = output_data
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    println!("   Output range: [{:.4}, {:.4}]", min_val, max_val);

    // Top-5 predictions
    let mut indexed: Vec<(usize, f32)> = output_data.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\n🏆 Top 5 predictions (class index, logit):");
    for (i, (class_idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("   {}. Class {}: {:.4}", i + 1, class_idx, logit);
    }

    // Apply softmax for probabilities
    let exp_sum: f32 = output_data.iter().map(|x| (x - max_val).exp()).sum();
    let top_prob = (indexed[0].1 - max_val).exp() / exp_sum;
    println!("\n   Top-1 probability: {:.2}%", top_prob * 100.0);

    if non_zero > 0 {
        println!("\n✅ SUCCESS! Model produces non-zero outputs - weights are loaded!");
    } else {
        println!("\n❌ FAIL: All outputs are zero - weights may not be loaded correctly");
    }

    Ok(())
}

/// Create a normalized mid-gray input [1, 3, 224, 224]
fn create_sample_input() -> Vec<f32> {
    let mut data = Vec::with_capacity(3 * 224 * 224);

    for c in 0..3 {
        for _h in 0..224 {
            for _w in 0..224 {
                // Mid-gray (0.5) normalized with ImageNet stats
                let normalized = (0.5 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                data.push(normalized);
            }
        }
    }

    data
}

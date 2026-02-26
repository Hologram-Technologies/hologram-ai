//! Run ResNet18 and show top-5 predictions.

use anyhow::{Context, Result};
use hologram::BackendPlan;
use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::holo::HolbReader;
use std::fs;

/// Deserialize BackendPlan from rkyv bytes (rkyv 0.7 API).
fn deserialize_plan(bytes: &[u8]) -> Result<BackendPlan> {
    let archived = unsafe { rkyv::archived_root::<BackendPlan>(bytes) };
    let plan: BackendPlan = rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible)?;
    Ok(plan)
}

fn main() -> Result<()> {
    let onnx_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx".to_string()
    });

    println!("Loading ResNet18 from: {}", onnx_path);
    let onnx_bytes = fs::read(&onnx_path).context("Failed to read ONNX model")?;

    println!("Compiling...");
    let holb_bytes = hologram_ai_onnx::compile_onnx(&onnx_bytes)?;
    println!(
        "Compiled to {} bytes ({:.1} MB)",
        holb_bytes.len(),
        holb_bytes.len() as f64 / 1_000_000.0
    );

    // Load the plan
    let reader = HolbReader::from_bytes(&holb_bytes)?;
    let plan = deserialize_plan(reader.graph())?;

    // Create test input (random pattern)
    let input_data: Vec<f32> = (0..3 * 224 * 224)
        .map(|i| ((i % 255) as f32) / 255.0)
        .collect();
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut output_data: Vec<f32> = vec![0.0; 1000];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    println!("Running inference...");
    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    // Find top-5 predictions
    let mut indexed: Vec<(usize, f32)> = output_data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop-5 predictions (class index, logit):");
    for (i, (class_idx, logit)) in indexed.iter().take(5).enumerate() {
        println!("  {}. Class {:4}: {:.4}", i + 1, class_idx, logit);
    }

    // Apply softmax for probabilities
    let max_logit = indexed[0].1;
    let exp_sum: f32 = output_data.iter().map(|&x| (x - max_logit).exp()).sum();
    let top_prob = 1.0 / exp_sum;
    println!("\nTop-1 probability: {:.2}%", top_prob * 100.0);

    Ok(())
}

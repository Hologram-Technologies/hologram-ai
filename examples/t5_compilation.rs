//! T5 text-to-text model compilation example.
//!
//! This example demonstrates compiling Google's T5 encoder-decoder model.
//! T5 is a sequence-to-sequence model for text generation tasks like:
//! - Translation
//! - Summarization
//! - Question answering
//!
//! ## Prerequisites
//!
//! Download T5-small ONNX models first:
//! ```bash
//! pip install optimum[exporters]
//! optimum-cli export onnx \
//!   --model google/t5-small \
//!   --task text2text-generation-with-past \
//!   /workspace/models/t5-small/
//! ```
//!
//! Run with: `cargo run --example t5_compilation`

use hologram_onnx::{OnnxCompiler, OnnxConfig};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== T5 Text-to-Text Model Compilation ===\n");

    // Check if models exist
    let encoder_path = "/workspace/models/t5-small/encoder_model.onnx";
    let decoder_path = "/workspace/models/t5-small/decoder_model.onnx";

    if !Path::new(encoder_path).exists() || !Path::new(decoder_path).exists() {
        eprintln!("❌ T5 models not found!");
        eprintln!("\nPlease download T5-small ONNX models first:\n");
        eprintln!("  pip install optimum[exporters]");
        eprintln!("  optimum-cli export onnx \\");
        eprintln!("    --model google/t5-small \\");
        eprintln!("    --task text2text-generation-with-past \\");
        eprintln!("    /workspace/models/t5-small/\n");
        return Ok(());
    }

    // Configure compilation for T5 architecture
    // T5 has ~600-800 nodes per component, so partitioning is recommended
    let config = OnnxConfig {
        enable_partitioning: true,
        partition_size: 200,           // Handle 600-800 nodes efficiently
        memory_budget: Some(2048),     // 2GB for T5-small
        pack_weights: true,            // Pack weights for faster runtime
        weight_threshold: 4096,        // External weights for large tensors
        decompose_conv2d: false,       // T5 doesn't use Conv2D
        decompose_pooling: false,      // T5 doesn't use pooling
        enable_resize_upscaling: false,
    };

    println!("Configuration:");
    println!("  - Partitioning: enabled ({} nodes/partition)", config.partition_size);
    println!("  - Memory budget: {} MB", config.memory_budget.unwrap_or(0));
    println!("  - Weight packing: {}\n", config.pack_weights);

    // Create compiler
    let compiler = OnnxCompiler::with_config(config);

    // Compile encoder
    println!("1. Compiling T5 Encoder");
    println!("   Loading {}...", encoder_path);

    let encoder_bytes = std::fs::read(encoder_path)?;
    println!("   ONNX model size: {} MB", encoder_bytes.len() / 1_000_000);

    println!("   Translating ONNX → hologram-ir...");
    let (encoder_holo, encoder_weights) = compiler.compile(&encoder_bytes)?;

    println!("   ✓ Encoder compiled!");
    println!("     - IR graph: {} bytes", encoder_holo.len());
    println!("     - Weights: {} bytes\n", encoder_weights.len());

    // Write encoder files
    let encoder_out = "/workspace/models/t5-small/encoder";
    std::fs::write(format!("{}.holo", encoder_out), &encoder_holo)?;
    if !encoder_weights.is_empty() {
        std::fs::write(format!("{}.weights", encoder_out), &encoder_weights)?;
    }

    // Compile decoder
    println!("2. Compiling T5 Decoder");
    println!("   Loading {}...", decoder_path);

    let decoder_bytes = std::fs::read(decoder_path)?;
    println!("   ONNX model size: {} MB", decoder_bytes.len() / 1_000_000);

    println!("   Translating ONNX → hologram-ir...");
    let (decoder_holo, decoder_weights) = compiler.compile(&decoder_bytes)?;

    println!("   ✓ Decoder compiled!");
    println!("     - IR graph: {} bytes", decoder_holo.len());
    println!("     - Weights: {} bytes\n", decoder_weights.len());

    // Write decoder files
    let decoder_out = "/workspace/models/t5-small/decoder";
    std::fs::write(format!("{}.holo", decoder_out), &decoder_holo)?;
    if !decoder_weights.is_empty() {
        std::fs::write(format!("{}.weights", decoder_out), &decoder_weights)?;
    }

    println!("✓ T5 Compilation Complete!\n");
    println!("Output files:");
    println!("  - {}.holo", encoder_out);
    if !encoder_weights.is_empty() {
        println!("  - {}.weights", encoder_out);
    }
    println!("  - {}.holo", decoder_out);
    if !decoder_weights.is_empty() {
        println!("  - {}.weights", decoder_out);
    }

    println!("\n📝 Notes:");
    println!("  - T5 encoder: Processes input text into hidden states");
    println!("  - T5 decoder: Generates output text token-by-token");
    println!("  - Full generation requires auto-regressive loop (runtime feature)");
    println!("  - See configs/examples/t5.toml for pipeline configuration");

    Ok(())
}

//! Inspect an ONNX model to see what operations it uses.

use hologram_ai_onnx::parse_model;
use std::collections::BTreeSet;
use std::env;
use std::fs;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.onnx>", args[0]);
        std::process::exit(1);
    }

    let model_path = &args[1];
    println!("📦 Loading model: {}", model_path);

    let onnx_bytes = fs::read(model_path)?;
    println!("   Size: {} MB", onnx_bytes.len() / 1024 / 1024);

    let model = parse_model(&onnx_bytes)?;
    let graph = model.graph.as_ref().expect("Model has no graph");

    // Collect all unique operation types
    let mut op_types = BTreeSet::new();
    for node in &graph.node {
        op_types.insert(node.op_type.clone());
    }

    println!("\n🔍 Operations used in this model:");
    for op in &op_types {
        println!("   - {}", op);
    }

    println!("\nTotal: {} unique operations", op_types.len());
    println!("Nodes: {}", graph.node.len());
    println!("Inputs: {}", graph.input.len());
    println!("Outputs: {}", graph.output.len());
    println!("Initializers: {}", graph.initializer.len());

    Ok(())
}

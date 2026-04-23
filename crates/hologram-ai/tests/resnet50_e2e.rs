//! End-to-end tests for ResNet-50 v2 — compile and run ONNX.
//!
//! Run with:
//!   cargo test -p hologram-ai --features e2e -- resnet --nocapture
//!
//! Model expected at (relative to workspace root):
//!   models/resnet50-v2-7.onnx

#![cfg(feature = "e2e")]

mod common;

use hologram_ai::compiler::{ModelCompiler, ModelSource};
use hologram_ai::validate::validate_model;
use std::path::PathBuf;

fn resnet_model_path() -> PathBuf {
    common::workspace_path("models/resnet50-v2-7.onnx")
}

#[test]
fn resnet50_onnx_compiles() {
    let model = resnet_model_path();
    if !model.exists() {
        return;
    }

    let report = validate_model(&model);
    assert!(
        report.compilation_ok,
        "ResNet-50 ONNX should compile: {:?}",
        report.error
    );
    // After BatchNorm decomposition + constant folding: ~225 nodes
    assert!(
        report.compiled_node_count > 100,
        "expected > 100 compiled nodes, got {}",
        report.compiled_node_count
    );
}

#[test]
fn resnet50_onnx_executes() {
    let model = resnet_model_path();
    if !model.exists() {
        return;
    }

    // Compile
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model))
        .expect("compilation failed");

    // Load plan directly from archive bytes.
    let plan = hologram::load_auto(&archive.bytes).expect("loading plan failed");

    // Build input: [1, 3, 224, 224] of deterministic f32
    let input_len = 1 * 3 * 224 * 224; // 150528 floats
    let input_data: Vec<f32> = (0..input_len).map(|i| ((i as f32) * 0.001).sin()).collect();
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();

    let mut inputs = hologram::GraphInputs::new();
    inputs.set_with_shape(0, input_bytes, vec![1, 3, 224, 224]);

    // Execute via tape executor (EnumTape — zero-overhead dispatch).
    let tape = hologram::build_tape_from_plan(&plan).expect("building execution tape");
    let outputs = hologram::execute_tape(&tape, &plan, &inputs).expect("execution failed");

    // Check output: should be [1, 1000] = 1000 floats = 4000 bytes
    let (_name, out_bytes) = outputs.get(0).expect("no output at index 0");
    let out_floats = common::bytes_to_f32(out_bytes);
    assert_eq!(
        out_floats.len(),
        1000,
        "ResNet-50 output should be 1000 classes, got {}",
        out_floats.len()
    );

    // All values should be finite
    assert!(
        out_floats.iter().all(|v| v.is_finite()),
        "output contains non-finite values"
    );
}

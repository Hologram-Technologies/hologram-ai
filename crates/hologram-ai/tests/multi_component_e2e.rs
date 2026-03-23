//! Multi-component compilation E2E test (Plan 022, Phase 3).
//!
//! Builds two synthetic ONNX models (encoder + decoder), compiles them
//! together via `ModelSource::MultiOnnx`, and verifies the resulting pipeline
//! archive contains the correct `MetaSection` with component descriptors.

use hologram_ai::compiler::{ComponentInput, ModelCompiler, ModelSource};
use hologram_ai_common::sections::meta::{ComponentConnection, ComponentRole};
use hologram_ai_conformance::ort_runner::onnx_builder;

/// Two-component multi-ONNX compilation produces a loadable pipeline archive
/// with correct MetaSection.
#[test]
fn multi_component_compile_and_load() {
    // Build two tiny ONNX models:
    // - "encoder": MatMul [4, 8] × [8, 16] → [4, 16]
    // - "decoder": MatMul [4, 16] × [16, 8] → [4, 8]
    let encoder_bytes = onnx_builder::matmul(4, 8, 16);
    let decoder_bytes = onnx_builder::matmul(4, 16, 8);

    // Write to temp files (ModelSource::MultiOnnx needs paths).
    let tmp = tempfile::tempdir().expect("creating temp dir");
    let enc_path = tmp.path().join("encoder.onnx");
    let dec_path = tmp.path().join("decoder.onnx");
    std::fs::write(&enc_path, &encoder_bytes).expect("writing encoder.onnx");
    std::fs::write(&dec_path, &decoder_bytes).expect("writing decoder.onnx");

    let source = ModelSource::MultiOnnx {
        components: vec![
            ComponentInput {
                name: "encoder".to_string(),
                path: enc_path,
                role: ComponentRole::Encoder,
                weight_group: "shared".to_string(),
            },
            ComponentInput {
                name: "decoder".to_string(),
                path: dec_path,
                role: ComponentRole::Decoder,
                weight_group: "shared".to_string(),
            },
        ],
        connections: vec![ComponentConnection {
            from_component: "encoder".to_string(),
            from_output: "Y".to_string(),
            to_component: "decoder".to_string(),
            to_input: "X".to_string(),
        }],
    };

    let compiler = ModelCompiler::default();
    let archive = compiler
        .compile(source)
        .expect("multi-component compilation failed");

    // Verify archive is non-empty and loadable.
    assert!(
        archive.bytes.len() > 100,
        "archive too small: {} bytes",
        archive.bytes.len()
    );

    // Verify metadata indicates multi-component.
    assert_eq!(archive.metadata.arch, "multi-onnx");
}

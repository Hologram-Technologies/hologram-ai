//! Integration tests for the validate command.
//!
//! Uses committed ONNX oracle artifacts (oracles/onnx/) so these work in CI.

use hologram_ai::validate::validate_model;
use std::path::PathBuf;

fn fixture_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/hologram-ai -> crates/
    p.pop(); // crates/ -> repo root
    p.push("oracles/onnx");
    p.push(name);
    p
}

#[test]
fn validate_identity_onnx() {
    let path = fixture_path("identity.onnx");
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (run scripts/gen-fixtures.py)",
            path.display()
        );
        return;
    }
    let report = validate_model(&path);
    assert!(
        report.compilation_ok,
        "identity.onnx should compile: {:?}",
        report.error
    );
    assert!(report.total_nodes > 0);
}

#[test]
fn validate_tiny_mlp_onnx() {
    let path = fixture_path("tiny-mlp.onnx");
    if !path.exists() {
        eprintln!(
            "skipping: {} not found (run scripts/gen-fixtures.py)",
            path.display()
        );
        return;
    }
    let report = validate_model(&path);
    assert!(
        report.compilation_ok,
        "tiny-mlp.onnx should compile: {:?}",
        report.error
    );
    assert!(
        report.total_nodes >= 2,
        "expected at least 2 nodes (Gather + MatMul)"
    );
    assert!(
        report.total_params >= 2,
        "expected at least 2 params (embed + linear weights)"
    );
    assert!(report.compiled_node_count > 0, "expected compiled nodes");
}

#[test]
fn validate_nonexistent_file() {
    let report = validate_model(std::path::Path::new("/tmp/does-not-exist.onnx"));
    assert!(!report.compilation_ok);
    assert!(report.error.is_some());
}

#[test]
fn validate_unsupported_format() {
    let report = validate_model(std::path::Path::new("/tmp/model.safetensors"));
    assert!(!report.compilation_ok);
    assert!(report.error.unwrap().contains("unsupported"));
}

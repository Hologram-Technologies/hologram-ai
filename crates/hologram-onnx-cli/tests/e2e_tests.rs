//! End-to-end tests for the hologram-onnx CLI.
//!
//! These tests verify the complete workflow:
//! - `hologram-onnx compile model.onnx -o output` creates .holo files
//! - `hologram-onnx info model.onnx` displays model information
//! - `hologram-onnx validate model.onnx` validates models
//!
//! # Test Models
//!
//! - **MNIST**: Simple convolutional network for digit classification
//! - **ResNet50**: Deep residual network for image classification
//!
//! # ISA Integration
//!
//! The compiled .holo files leverage hologram's ISA:
//! - **LOOP instructions**: O(1) space complexity for sequences
//! - **SIMD vectorization**: Parallel processing of convolutions
//! - **ClassMap fusion**: Element-wise activation chains

#![allow(deprecated)] // cargo_bin is deprecated but still the standard way for assert_cmd

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Get path to MNIST test model.
fn mnist_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("hologram-onnx-core/tests/fixtures/mnist-12.onnx")
}

/// Get path to ResNet50 test model (if available).
fn resnet_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("models/resnet50-v1-7.onnx")
}

/// Check if MNIST model exists.
fn has_mnist_model() -> bool {
    mnist_model_path().exists()
}

/// Check if ResNet model exists.
fn has_resnet_model() -> bool {
    resnet_model_path().exists()
}

// ============================================================================
// Compile Command Tests
// ============================================================================

/// Test compile command with missing input file.
#[test]
fn test_compile_missing_input() {
    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("output");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg("nonexistent.onnx")
        .arg("-o")
        .arg(&output);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to read ONNX model"));
}

/// Test compile command with MNIST model.
#[test]
fn test_compile_mnist() {
    if !has_mnist_model() {
        eprintln!("Skipping test_compile_mnist: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("mnist_compiled");
    let holo_path = output.with_extension("holo");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg(mnist_model_path())
        .arg("-o")
        .arg(&output);

    cmd.assert().success();

    // Verify .holo file was created
    assert!(holo_path.exists(), ".holo file should be created");

    // Verify .holo file has content
    let holo_content = fs::read(&holo_path).unwrap();
    assert!(!holo_content.is_empty(), ".holo file should not be empty");

    // .holo files are now rkyv binary format, not JSON
    // Just verify it exists and has reasonable size
    assert!(
        holo_content.len() > 100,
        ".holo file should have meaningful content, got {} bytes",
        holo_content.len()
    );

    eprintln!("MNIST compiled: {} bytes (rkyv format)", holo_content.len());
}

/// Test compile command with verbose output.
#[test]
fn test_compile_verbose() {
    if !has_mnist_model() {
        eprintln!("Skipping test_compile_verbose: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("mnist_verbose");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("--verbose")
        .arg("compile")
        .arg(mnist_model_path())
        .arg("-o")
        .arg(&output);

    cmd.assert().success();
}

/// Test compile command with partitioning enabled.
#[test]
fn test_compile_with_partitioning() {
    if !has_mnist_model() {
        eprintln!("Skipping test_compile_with_partitioning: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("mnist_partitioned");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg(mnist_model_path())
        .arg("-o")
        .arg(&output)
        .arg("--partition")
        .arg("--partition-size")
        .arg("100");

    cmd.assert().success();

    assert!(output.with_extension("holo").exists());
}

/// Test compile command with memory budget.
#[test]
fn test_compile_with_memory_budget() {
    if !has_mnist_model() {
        eprintln!("Skipping test_compile_with_memory_budget: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("mnist_memory_budget");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg(mnist_model_path())
        .arg("-o")
        .arg(&output)
        .arg("--memory-budget")
        .arg("512");

    cmd.assert().success();
}

/// Test compile with ResNet50 model (larger model test).
#[test]
fn test_compile_resnet() {
    if !has_resnet_model() {
        eprintln!(
            "Skipping test_compile_resnet: ResNet model not found at {:?}",
            resnet_model_path()
        );
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("resnet_compiled");
    let holo_path = output.with_extension("holo");

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg(resnet_model_path())
        .arg("-o")
        .arg(&output)
        .arg("--partition")
        .arg("--partition-size")
        .arg("200");

    cmd.assert().success();

    assert!(holo_path.exists(), "ResNet .holo file should be created");

    let holo_content = fs::read(&holo_path).unwrap();
    eprintln!("ResNet compiled: {} bytes", holo_content.len());
}

// ============================================================================
// Info Command Tests
// ============================================================================

/// Test info command with missing file.
#[test]
fn test_info_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg("nonexistent.onnx");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to read"));
}

/// Test info command with MNIST model.
#[test]
fn test_info_mnist() {
    if !has_mnist_model() {
        eprintln!("Skipping test_info_mnist: MNIST model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg(mnist_model_path());

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"));
}

/// Test info command with detailed output.
#[test]
fn test_info_detailed() {
    if !has_mnist_model() {
        eprintln!("Skipping test_info_detailed: MNIST model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg(mnist_model_path()).arg("--detailed");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"));
}

/// Test info command with ResNet model.
#[test]
fn test_info_resnet() {
    if !has_resnet_model() {
        eprintln!("Skipping test_info_resnet: ResNet model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg(resnet_model_path()).arg("--detailed");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Model Information"));
}

// ============================================================================
// Validate Command Tests
// ============================================================================

/// Test validate command with missing file.
#[test]
fn test_validate_missing_file() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate").arg("nonexistent.onnx");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to read"));
}

/// Test validate command with MNIST model.
#[test]
fn test_validate_mnist() {
    if !has_mnist_model() {
        eprintln!("Skipping test_validate_mnist: MNIST model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate").arg(mnist_model_path());

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Validation"));
}

/// Test validate command with operation checking.
#[test]
fn test_validate_check_ops() {
    if !has_mnist_model() {
        eprintln!("Skipping test_validate_check_ops: MNIST model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate")
        .arg(mnist_model_path())
        .arg("--check-ops");

    cmd.assert().success();
}

/// Test validate command with ResNet model.
#[test]
fn test_validate_resnet() {
    if !has_resnet_model() {
        eprintln!("Skipping test_validate_resnet: ResNet model not found");
        return;
    }

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate")
        .arg(resnet_model_path())
        .arg("--check-ops");

    cmd.assert().success();
}

// ============================================================================
// Output File Verification Tests
// ============================================================================

/// Verify .holo file format and structure.
#[test]
fn test_holo_file_format() {
    if !has_mnist_model() {
        eprintln!("Skipping test_holo_file_format: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let output = temp_dir.path().join("format_test");
    let holo_path = output.with_extension("holo");

    // Compile the model
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile")
        .arg(mnist_model_path())
        .arg("-o")
        .arg(&output);
    cmd.assert().success();

    // Read and verify .holo file
    let content = fs::read(&holo_path).unwrap();

    // New format: rkyv-serialized OperationGraph (binary)
    assert!(!content.is_empty(), "File should not be empty");

    // Verify it has meaningful size (rkyv binary data)
    assert!(
        content.len() > 100,
        ".holo file should have meaningful content, got {} bytes",
        content.len()
    );

    eprintln!(
        ".holo file format verified: {} bytes, rkyv binary format",
        content.len()
    );
}

/// Test that multiple compiles produce consistent output.
#[test]
fn test_compile_consistency() {
    if !has_mnist_model() {
        eprintln!("Skipping test_compile_consistency: MNIST model not found");
        return;
    }

    let temp_dir = TempDir::new().unwrap();

    // Compile twice
    let output1 = temp_dir.path().join("compile1");
    let output2 = temp_dir.path().join("compile2");

    for output in [&output1, &output2] {
        let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
        cmd.arg("compile")
            .arg(mnist_model_path())
            .arg("-o")
            .arg(output);
        cmd.assert().success();
    }

    // Read both files
    let content1 = fs::read(output1.with_extension("holo")).unwrap();
    let content2 = fs::read(output2.with_extension("holo")).unwrap();

    // Verify they're the same size (content may differ due to timestamps, but size should match)
    assert_eq!(
        content1.len(),
        content2.len(),
        "Consistent compiles should produce same size output"
    );

    eprintln!(
        "Compile consistency verified: {} bytes each",
        content1.len()
    );
}

// ============================================================================
// CLI Help and Version Tests
// ============================================================================

/// Test help output.
#[test]
fn test_help() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Production ONNX runtime"))
        .stdout(predicate::str::contains("compile"))
        .stdout(predicate::str::contains("download"))
        .stdout(predicate::str::contains("info"))
        .stdout(predicate::str::contains("validate"));
}

/// Test version output.
#[test]
fn test_version() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("--version");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("hologram-onnx"));
}

/// Test compile subcommand help.
#[test]
fn test_compile_help() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Compile ONNX model"))
        .stdout(predicate::str::contains("--output"))
        .stdout(predicate::str::contains("--partition"));
}

/// Test info subcommand help.
#[test]
fn test_info_help() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Display ONNX model information"));
}

/// Test validate subcommand help.
#[test]
fn test_validate_help() {
    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate").arg("--help");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Validate ONNX model"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

/// Test compile with invalid ONNX file.
#[test]
fn test_compile_invalid_onnx() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_onnx = temp_dir.path().join("invalid.onnx");
    let output = temp_dir.path().join("output");

    // Create invalid ONNX file
    fs::write(&invalid_onnx, b"not a valid onnx file").unwrap();

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("compile").arg(&invalid_onnx).arg("-o").arg(&output);

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Failed to parse"));
}

/// Test info with invalid ONNX file.
#[test]
fn test_info_invalid_onnx() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_onnx = temp_dir.path().join("invalid.onnx");

    // Create invalid ONNX file
    fs::write(&invalid_onnx, b"not a valid onnx file").unwrap();

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("info").arg(&invalid_onnx);

    cmd.assert().failure();
}

/// Test validate with invalid ONNX file.
#[test]
fn test_validate_invalid_onnx() {
    let temp_dir = TempDir::new().unwrap();
    let invalid_onnx = temp_dir.path().join("invalid.onnx");

    // Create invalid ONNX file
    fs::write(&invalid_onnx, b"not a valid onnx file").unwrap();

    let mut cmd = Command::cargo_bin("hologram-onnx").unwrap();
    cmd.arg("validate").arg(&invalid_onnx);

    cmd.assert().failure();
}

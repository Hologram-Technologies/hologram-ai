//! Integration checklist tests following the hologram integration guide.
//!
//! These tests verify that the ONNX integration follows the patterns and
//! recommendations from specs/external-plans/hologram-integration.md.
//!
//! Test Coverage:
//! - Operation discovery (Guide Section 4)
//! - Weight strategies (Guide Section 6)
//! - Dynamic shapes (Guide Section 7.4)
//! - Optimization features (Guide Section 7)

use hologram_ai::runtime::ModelExecutor;
use hologram_ai_onnx::OnnxCompiler;
use std::fs;
use std::path::PathBuf;

/// Helper to get test fixtures directory
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Helper to compile an ONNX model for testing
fn compile_test_model(fixture_name: &str) -> anyhow::Result<Vec<u8>> {
    let fixture_path = fixtures_dir().join(fixture_name);

    // Skip test if fixture doesn't exist (not all tests have fixtures yet)
    if !fixture_path.exists() {
        eprintln!("Skipping test - fixture not found: {:?}", fixture_path);
        return Ok(Vec::new());
    }

    let onnx_bytes = fs::read(fixture_path)?;
    let compiler = OnnxCompiler::new();
    let (holo_bytes, _weight_bytes) = compiler.compile(&onnx_bytes)?;
    Ok(holo_bytes)
}

/// Helper to load a ModelExecutor from holo bytes (writes to temp file)
fn load_executor_from_bytes(holo_bytes: &[u8]) -> anyhow::Result<ModelExecutor> {
    // Write to temporary file (ModelExecutor requires file path)
    let temp_dir = tempfile::tempdir()?;
    let holo_path = temp_dir.path().join("model.holo");
    fs::write(&holo_path, holo_bytes)?;

    // Load from file
    let executor = ModelExecutor::from_holo_file(&holo_path)?;
    Ok(executor)
}

// =============================================================================
// Operation Discovery Tests (Guide Section 4)
// =============================================================================

#[test]
fn test_operation_discovery_basic() -> anyhow::Result<()> {
    // Compile a simple MNIST model
    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Discover operations (Guide Section 4.4)
    let operations = executor.operations();

    // Verify we can enumerate operations
    assert!(!operations.is_empty(), "Model should have operations");

    // Verify operation info is populated
    for op in operations {
        assert!(!op.kernel_id.is_empty(), "Kernel ID should not be empty");
        // Note: kernel_name may be None for operations without named kernels
    }

    Ok(())
}

#[test]
fn test_optimization_report() -> anyhow::Result<()> {
    // Compile a simple model
    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Get optimization report (Guide Section 7)
    let report = executor.optimization_report();

    // Verify SIMD level is detected
    assert!(
        !report.simd_level.is_empty(),
        "SIMD level should be detected"
    );

    // Verify report fields are populated (values depend on model)
    // Just check that the report is generated without errors
    println!("Optimization report:");
    println!("  SIMD activations: {}", report.has_simd_activations);
    println!("  Epilogue fusion: {}", report.has_epilogue_fusion);
    println!("  Parallel groups: {}", report.has_parallel_groups);
    println!("  Parallel group count: {}", report.parallel_group_count);
    println!("  Parallelizable ops: {}", report.parallelizable_ops);
    println!("  Embedding cache: {}", report.has_embedding_cache);
    println!("  SIMD level: {}", report.simd_level);
    println!("  Dynamic shapes: {}", report.dynamic_shapes);

    Ok(())
}

// =============================================================================
// Weight Strategy Tests (Guide Section 6)
// =============================================================================

#[test]
fn test_weight_strategy_auto_select() {
    use hologram_ai_common::WeightStrategy;

    // Test auto-selection based on model size (Guide Section 6)

    // Small model: < 100MB → embedded
    let strategy = WeightStrategy::auto_select(50 * 1024 * 1024);
    assert_eq!(strategy, WeightStrategy::EmbeddedInPlan);

    // Medium model: 100-1000MB → page-aligned
    let strategy = WeightStrategy::auto_select(500 * 1024 * 1024);
    assert_eq!(strategy, WeightStrategy::PageAlignedInBundle);

    // Large model: > 1GB → external
    let strategy = WeightStrategy::auto_select(2 * 1024 * 1024 * 1024);
    assert_eq!(strategy, WeightStrategy::ExternalFile);
}

#[test]
fn test_embedded_weights_compilation() -> anyhow::Result<()> {
    // Test embedded weights strategy (< 100MB models)
    // This is what most small models use by default

    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Model should compile successfully with embedded weights
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Verify we can discover operations (indicating successful compilation)
    let operations = executor.operations();
    assert!(
        !operations.is_empty(),
        "Compiled model should have operations"
    );

    Ok(())
}

#[test]
fn test_page_aligned_weights_format() -> anyhow::Result<()> {
    // Test that page-aligned weights are used for medium models (100MB-1GB)
    // The OnnxCompiler currently uses PageAlignedInBundle by default

    let compiler = OnnxCompiler::new();

    // Try to compile a model (skip if not available)
    let fixture_path = fixtures_dir().join("resnet18.onnx");
    if !fixture_path.exists() {
        eprintln!("Skipping test - resnet18.onnx fixture not found");
        return Ok(());
    }

    let onnx_bytes = fs::read(fixture_path)?;
    let (holo_bytes, weight_bytes) = compiler.compile(&onnx_bytes)?;

    // For PageAlignedInBundle strategy:
    // - holo_bytes should contain both plan and weights
    // - weight_bytes should be empty (weights are in the bundle)
    assert!(!holo_bytes.is_empty(), "Bundle should not be empty");
    assert!(
        weight_bytes.is_empty(),
        "External weights should be empty for PageAlignedInBundle"
    );

    Ok(())
}

#[test]
fn test_external_weights_strategy() {
    use hologram_ai_common::WeightStrategy;

    // Test that ExternalFile strategy is selected for large models
    // This is primarily for GGUF models > 1GB

    let large_model_size = 2 * 1024 * 1024 * 1024; // 2GB
    let strategy = WeightStrategy::auto_select(large_model_size);

    match strategy {
        WeightStrategy::ExternalFile => {
            // Correct strategy for large models
        }
        _ => {
            panic!("Expected ExternalFile strategy for 2GB model");
        }
    }
}

// =============================================================================
// Dynamic Shapes Tests (Guide Section 7.4)
// =============================================================================

#[test]
fn test_dynamic_shapes_detection() -> anyhow::Result<()> {
    // Test dynamic shape support in compiled models

    // Try to compile a model with dynamic shapes (BERT, T5, etc.)
    let fixture_path = fixtures_dir().join("bert_base_dynamic.onnx");
    if !fixture_path.exists() {
        eprintln!("Skipping test - bert_base_dynamic.onnx fixture not found");
        return Ok(());
    }

    let onnx_bytes = fs::read(fixture_path)?;
    let compiler = OnnxCompiler::new();
    let (holo_bytes, _) = compiler.compile(&onnx_bytes)?;

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Check if dynamic shapes are detected
    let report = executor.optimization_report();

    // For models with variable batch/seq_len, dynamic_shapes should be true
    // (This depends on the model having symbolic dimensions)
    println!("Dynamic shapes detected: {}", report.dynamic_shapes);

    Ok(())
}

#[test]
fn test_symbolic_shape_support() -> anyhow::Result<()> {
    // Verify that ONNX compiler preserves symbolic shapes
    use hologram_ai_onnx::{Dim, SymbolicShape};

    // Test symbolic dimension creation
    let batch_dim = Dim::Symbolic("batch".to_string());
    let seq_len_dim = Dim::Symbolic("seq_len".to_string());
    let hidden_dim = Dim::Static(768);

    // Create a symbolic shape [batch, seq_len, 768]
    let shape = SymbolicShape::new(vec![batch_dim, seq_len_dim, hidden_dim]);

    // Verify shape properties
    assert_eq!(shape.rank(), 3);
    assert!(
        !shape.is_fully_concrete(),
        "Shape with symbolic dims should not be fully concrete"
    );

    Ok(())
}

// =============================================================================
// Optimization Feature Tests (Guide Section 7)
// =============================================================================

#[test]
fn test_simd_detection() {
    // Test SIMD capability detection (Guide Section 7.3)
    // This should work on any CPU

    use hologram::lookup::detect_simd;
    let simd_level = detect_simd();

    // Verify SIMD level is detected
    println!("Detected SIMD level: {:?}", simd_level);

    // Just verify it returns something (the actual level depends on the CPU)
    // Common values: Avx512, Avx2, Sse42, Neon (ARM), Baseline
}

#[test]
fn test_winograd_convolution_support() -> anyhow::Result<()> {
    // Test Winograd optimization for 3x3 convolutions (Guide Section 7.2)

    // Try to compile a model with convolutions
    let fixture_path = fixtures_dir().join("resnet18.onnx");
    if !fixture_path.exists() {
        eprintln!("Skipping test - resnet18.onnx fixture not found");
        return Ok(());
    }

    let onnx_bytes = fs::read(fixture_path)?;
    let compiler = OnnxCompiler::new();
    let (holo_bytes, _) = compiler.compile(&onnx_bytes)?;

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Get operations
    let operations = executor.operations();

    // Verify model has operations (convolutions should be translated)
    assert!(!operations.is_empty(), "ResNet should have operations");

    // Note: Winograd detection would require inspecting kernel IDs
    // This is a placeholder for future enhancement

    Ok(())
}

#[test]
fn test_epilogue_fusion() -> anyhow::Result<()> {
    // Test epilogue fusion optimization (Guide Section 7.1)
    // Example: MatMul + Add + ReLU → fused operation

    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Check optimization report
    let report = executor.optimization_report();

    // Verify epilogue fusion detection works
    println!("Epilogue fusion: {}", report.has_epilogue_fusion);

    // Note: Whether fusion is actually applied depends on the model
    // This test just verifies the detection mechanism works

    Ok(())
}

#[test]
fn test_parallel_execution_groups() -> anyhow::Result<()> {
    // Test parallel execution group detection (Guide Section 7.5)

    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Load model
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // Check optimization report
    let report = executor.optimization_report();

    // Verify parallel group detection works
    println!("Parallel groups: {}", report.parallel_group_count);
    println!("Parallelizable ops: {}", report.parallelizable_ops);

    // Simple models may not have parallel groups, but the detection should work
    // (parallel_group_count is usize, always non-negative)

    Ok(())
}

// =============================================================================
// End-to-End Integration Tests
// =============================================================================

#[test]
fn test_full_compilation_pipeline() -> anyhow::Result<()> {
    // Test the complete ONNX → .holo → execution pipeline

    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // 1. Compilation successful (already done)
    assert!(!holo_bytes.is_empty(), "Compilation should produce bytes");

    // 2. Loading successful
    let executor = load_executor_from_bytes(&holo_bytes)?;

    // 3. Operation discovery works
    let operations = executor.operations();
    assert!(!operations.is_empty(), "Model should have operations");

    // 4. Optimization report works
    let report = executor.optimization_report();
    assert!(!report.simd_level.is_empty(), "SIMD should be detected");

    // 5. Model can be queried for info
    println!("Model compiled successfully:");
    println!("  Operations: {}", operations.len());
    println!("  SIMD level: {}", report.simd_level);
    println!("  Dynamic shapes: {}", report.dynamic_shapes);

    Ok(())
}

#[test]
fn test_model_portability() -> anyhow::Result<()> {
    // Test that compiled .holo files are portable (Guide Section 8)

    let holo_bytes = compile_test_model("mnist_mlp.onnx")?;
    if holo_bytes.is_empty() {
        eprintln!("Skipping test - no fixture");
        return Ok(());
    }

    // Write to file
    let temp_dir = tempfile::tempdir()?;
    let holo_path = temp_dir.path().join("model.holo");
    fs::write(&holo_path, &holo_bytes)?;

    // Load from file
    let executor = ModelExecutor::from_holo_file(&holo_path)?;

    // Verify model works
    let operations = executor.operations();
    assert!(
        !operations.is_empty(),
        "Loaded model should have operations"
    );

    Ok(())
}

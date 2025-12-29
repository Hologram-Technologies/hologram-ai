//! Stable Diffusion model integration tests.
//!
//! Tests the compilation pipeline for Stable Diffusion image generation models:
//! - ONNX parsing → IR translation → decomposition → .holo serialization
//! - Multi-component architecture (UNet, VAE, CLIP)
//! - Large model handling (3000+ nodes)
//! - Image output handler integration
//!
//! Note: Stable Diffusion models must be downloaded separately.
//! Tests will skip if models are not available.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use hologram_compiler::ir::IRBuilder;
use hologram_compiler::shapes::Dim;
use hologram_onnx_core::{
    parse_model, validate_model,
    OnnxConfig, SymbolicShape,
};
use hologram_onnx_ops::translate_onnx_op;
use tempfile::TempDir;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Stable Diffusion component paths.
#[allow(dead_code)]
struct SDComponentPaths {
    unet: Option<PathBuf>,
    vae_encoder: Option<PathBuf>,
    vae_decoder: Option<PathBuf>,
    text_encoder: Option<PathBuf>,
    safety_checker: Option<PathBuf>,
}

/// Get possible paths for SD components.
fn sd_component_paths() -> SDComponentPaths {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let find_component = |names: &[&str]| -> Option<PathBuf> {
        for name in names {
            let path = base.join(format!("models/{}", name));
            if path.exists() {
                return Some(path);
            }
            // Also check subdirectory
            let subpath = base.join(format!("models/stable-diffusion/{}", name));
            if subpath.exists() {
                return Some(subpath);
            }
        }
        None
    };

    SDComponentPaths {
        unet: find_component(&["unet.onnx", "unet/model.onnx"]),
        vae_encoder: find_component(&["vae_encoder.onnx", "vae_encoder/model.onnx"]),
        vae_decoder: find_component(&["vae_decoder.onnx", "vae_decoder/model.onnx"]),
        text_encoder: find_component(&["text_encoder.onnx", "text_encoder/model.onnx", "clip.onnx"]),
        safety_checker: find_component(&["safety_checker.onnx", "safety_checker/model.onnx"]),
    }
}

/// Check if any SD component exists.
fn has_any_sd_component() -> bool {
    let paths = sd_component_paths();
    paths.unet.is_some()
        || paths.vae_encoder.is_some()
        || paths.vae_decoder.is_some()
        || paths.text_encoder.is_some()
}

/// Load SD component model bytes.
fn load_sd_component(path: &Option<PathBuf>) -> Option<Vec<u8>> {
    path.as_ref().and_then(|p| fs::read(p).ok())
}

// ============================================================================
// UNet Tests
// ============================================================================

/// Test UNet parsing (main diffusion model).
#[test]
fn test_sd_unet_parsing() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.unet) else {
        eprintln!("Skipping test_sd_unet_parsing: UNet model not found");
        eprintln!("Download SD ONNX models to: models/stable-diffusion/");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse UNet model");

    let graph = model.graph.as_ref().expect("Model should have graph");

    eprintln!("UNet parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len());

    // UNet is a large model (typically 3000+ nodes)
    assert!(graph.node.len() > 1000, "UNet should have many nodes (got {})", graph.node.len());
}

/// Test UNet validation.
#[test]
fn test_sd_unet_validation() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.unet) else {
        eprintln!("Skipping test_sd_unet_validation: UNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse UNet model");
    let result = validate_model(&model);

    assert!(result.is_ok(), "UNet validation should pass: {:?}", result.err());
}

/// Test UNet input shapes (latent, timestep, encoder_hidden_states).
#[test]
fn test_sd_unet_input_shapes() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.unet) else {
        eprintln!("Skipping test_sd_unet_input_shapes: UNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse UNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    for input in &graph.input {
        // Skip initializers
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }

        if let Ok(shape) = SymbolicShape::from_value_info(input) {
            let dims = shape.dims();
            eprintln!("UNet input '{}': {:?}", input.name, dims);

            // UNet expects:
            // - sample: [batch, 4, H/8, W/8] latent
            // - timestep: [batch] or scalar
            // - encoder_hidden_states: [batch, seq_len, hidden]
        }
    }
}

/// Test UNet operation coverage.
#[test]
fn test_sd_unet_operation_coverage() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.unet) else {
        eprintln!("Skipping test_sd_unet_operation_coverage: UNet model not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse UNet model");
    let graph = model.graph.as_ref().expect("Model should have graph");

    let mut builder = IRBuilder::new("unet_test");
    let shapes: HashMap<String, SymbolicShape> = HashMap::new();

    let mut op_counts: HashMap<String, usize> = HashMap::new();
    let mut supported_count = 0;
    let mut unsupported_ops: Vec<String> = Vec::new();

    for node in &graph.node {
        *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;

        let result = translate_onnx_op(
            &node.op_type,
            &[],
            &node.attribute,
            &shapes,
            &mut builder,
        );

        match &result {
            Err(hologram_onnx_core::OnnxError::UnsupportedOp { op_type, .. }) => {
                if !unsupported_ops.contains(op_type) {
                    unsupported_ops.push(op_type.clone());
                }
            }
            _ => {
                supported_count += 1;
            }
        }
    }

    eprintln!("UNet operation breakdown (top 15):");
    let mut sorted_ops: Vec<_> = op_counts.iter().collect();
    sorted_ops.sort_by(|a, b| b.1.cmp(a.1));
    for (op, count) in &sorted_ops[..sorted_ops.len().min(15)] {
        eprintln!("  {}: {}", op, count);
    }

    if !unsupported_ops.is_empty() {
        eprintln!("Unsupported operations: {:?}", unsupported_ops);
    }

    let support_ratio = supported_count as f64 / graph.node.len() as f64;
    eprintln!("Support ratio: {:.1}%", support_ratio * 100.0);
}

/// Test UNet compilation with partitioning (required for large models).
#[test]
fn test_sd_unet_compilation() {
    let paths = sd_component_paths();
    let unet_path = match &paths.unet {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_sd_unet_compilation: UNet model not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("unet_compiled");

    use std::process::Command;

    // UNet requires partitioning due to size
    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            unet_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--partition",
            "--partition-size",
            "200",
            "--memory-budget",
            "2048", // 2GB for UNet
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "UNet compilation should succeed");

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    let holo_content = fs::read(&holo_path).expect("Should read .holo file");
    eprintln!("UNet compiled successfully: {} bytes", holo_content.len());
}

// ============================================================================
// VAE Tests
// ============================================================================

/// Test VAE decoder parsing (latent to image).
#[test]
fn test_sd_vae_decoder_parsing() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.vae_decoder) else {
        eprintln!("Skipping test_sd_vae_decoder_parsing: VAE decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse VAE decoder");

    let graph = model.graph.as_ref().expect("Model should have graph");
    eprintln!("VAE decoder parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len());
}

/// Test VAE decoder input/output shapes.
#[test]
fn test_sd_vae_decoder_shapes() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.vae_decoder) else {
        eprintln!("Skipping test_sd_vae_decoder_shapes: VAE decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse VAE decoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Input shape: [batch, 4, H/8, W/8]
    for input in &graph.input {
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }
        if let Ok(shape) = SymbolicShape::from_value_info(input) {
            eprintln!("VAE decoder input '{}': {:?}", input.name, shape.dims());
        }
    }

    // Output shape: [batch, 3, H, W]
    for output in &graph.output {
        if let Ok(shape) = SymbolicShape::from_value_info(output) {
            eprintln!("VAE decoder output '{}': {:?}", output.name, shape.dims());

            // Check for RGB output
            let dims = shape.dims();
            if dims.len() >= 2 && matches!(dims[1], Dim::Concrete(3)) {
                eprintln!("  Confirmed RGB output (3 channels)");
            }
        }
    }
}

/// Test VAE decoder compilation.
#[test]
fn test_sd_vae_decoder_compilation() {
    let paths = sd_component_paths();
    let vae_path = match &paths.vae_decoder {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_sd_vae_decoder_compilation: VAE decoder not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("vae_decoder_compiled");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            vae_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "VAE decoder compilation should succeed");

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    eprintln!("VAE decoder compiled successfully");
}

// ============================================================================
// Text Encoder Tests
// ============================================================================

/// Test text encoder parsing (CLIP).
#[test]
fn test_sd_text_encoder_parsing() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.text_encoder) else {
        eprintln!("Skipping test_sd_text_encoder_parsing: Text encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse text encoder");

    let graph = model.graph.as_ref().expect("Model should have graph");
    eprintln!("Text encoder parsed: {} nodes, {} initializers",
        graph.node.len(),
        graph.initializer.len());
}

/// Test text encoder input shapes (token IDs).
#[test]
fn test_sd_text_encoder_shapes() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.text_encoder) else {
        eprintln!("Skipping test_sd_text_encoder_shapes: Text encoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse text encoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    for input in &graph.input {
        if graph.initializer.iter().any(|i| i.name == input.name) {
            continue;
        }
        if let Ok(shape) = SymbolicShape::from_value_info(input) {
            eprintln!("Text encoder input '{}': {:?}", input.name, shape.dims());
            // Input is typically [batch, seq_len] token IDs
        }
    }

    for output in &graph.output {
        if let Ok(shape) = SymbolicShape::from_value_info(output) {
            eprintln!("Text encoder output '{}': {:?}", output.name, shape.dims());
            // Output is typically [batch, seq_len, hidden_dim]
        }
    }
}

/// Test text encoder compilation.
#[test]
fn test_sd_text_encoder_compilation() {
    let paths = sd_component_paths();
    let text_path = match &paths.text_encoder {
        Some(path) => path,
        None => {
            eprintln!("Skipping test_sd_text_encoder_compilation: Text encoder not found");
            return;
        }
    };

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let output = temp_dir.path().join("text_encoder_compiled");

    use std::process::Command;

    let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
        .args([
            "compile",
            text_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .status()
        .expect("Failed to run hologram-onnx compile");

    assert!(status.success(), "Text encoder compilation should succeed");

    let holo_path = output.with_extension("holo");
    assert!(holo_path.exists(), ".holo file should be created");

    eprintln!("Text encoder compiled successfully");
}

// ============================================================================
// Multi-Stage Pipeline Tests
// ============================================================================

/// Test compilation of all SD components.
#[test]
fn test_sd_full_pipeline_compilation() {
    let paths = sd_component_paths();

    if !has_any_sd_component() {
        eprintln!("Skipping test_sd_full_pipeline_compilation: No SD components found");
        return;
    }

    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    use std::process::Command;

    // Compile each available component
    let components = [
        (&paths.text_encoder, "text_encoder"),
        (&paths.vae_encoder, "vae_encoder"),
        (&paths.unet, "unet"),
        (&paths.vae_decoder, "vae_decoder"),
    ];

    let mut compiled_count = 0;

    for (path, name) in components {
        let Some(model_path) = path else {
            eprintln!("Skipping {}: not found", name);
            continue;
        };

        let output = temp_dir.path().join(format!("{}_compiled", name));

        let mut args = vec![
            "compile".to_string(),
            model_path.to_str().unwrap().to_string(),
            "-o".to_string(),
            output.to_str().unwrap().to_string(),
        ];

        // UNet needs partitioning
        if name == "unet" {
            args.extend([
                "--partition".to_string(),
                "--partition-size".to_string(),
                "200".to_string(),
            ]);
        }

        let status = Command::new(env!("CARGO_BIN_EXE_hologram-onnx"))
            .args(&args)
            .status()
            .expect("Failed to run hologram-onnx compile");

        if status.success() {
            compiled_count += 1;
            let holo_path = output.with_extension("holo");
            let size = fs::metadata(&holo_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("{} compiled: {} bytes", name, size);
        } else {
            eprintln!("{} compilation failed", name);
        }
    }

    eprintln!("SD pipeline: {}/4 components compiled", compiled_count);
}

/// Test SD configuration for large models.
#[test]
fn test_sd_config_validation() {
    // Configuration for Stable Diffusion
    let config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: true, // Required for UNet
        partition_size: 200,       // Reasonable for 3000+ node UNet
        decompose_conv2d: true,
        decompose_pooling: true,
        memory_budget: Some(4096), // 4GB for full pipeline
    };

    assert!(config.validate().is_ok(), "SD config should be valid");

    // Test memory-constrained configuration
    let constrained_config = OnnxConfig {
        weight_threshold: 4096,
        enable_partitioning: true,
        partition_size: 100, // Smaller partitions for memory constraints
        decompose_conv2d: true,
        decompose_pooling: true,
        memory_budget: Some(2048), // 2GB limit
    };

    assert!(constrained_config.validate().is_ok(), "Constrained config should be valid");
}

/// Test SD weight sizes.
#[test]
fn test_sd_weight_sizes() {
    let paths = sd_component_paths();

    if !has_any_sd_component() {
        eprintln!("Skipping test_sd_weight_sizes: No SD components found");
        return;
    }

    let components = [
        (&paths.text_encoder, "text_encoder"),
        (&paths.vae_encoder, "vae_encoder"),
        (&paths.unet, "unet"),
        (&paths.vae_decoder, "vae_decoder"),
    ];

    let mut total_size = 0usize;

    for (path, name) in components {
        let Some(onnx_bytes) = load_sd_component(path) else {
            continue;
        };

        let model = parse_model(&onnx_bytes).expect("Failed to parse model");
        let graph = model.graph.as_ref().expect("Model should have graph");

        let weight_bytes: usize = graph.initializer.iter()
            .map(|init| {
                if !init.raw_data.is_empty() {
                    init.raw_data.len()
                } else {
                    let elem_size = match init.data_type {
                        1 => 4, // FLOAT
                        7 => 8, // INT64
                        _ => 4,
                    };
                    let num_elements: usize = init.dims.iter().map(|&d| d as usize).product();
                    num_elements * elem_size
                }
            })
            .sum();

        total_size += weight_bytes;

        eprintln!("{} weights: {} MB ({} initializers)",
            name,
            weight_bytes / (1024 * 1024),
            graph.initializer.len());
    }

    eprintln!("Total SD weights: {} GB", total_size as f64 / (1024.0 * 1024.0 * 1024.0));
}

/// Test variable image resolution support.
#[test]
fn test_sd_variable_resolution() {
    let paths = sd_component_paths();
    let Some(onnx_bytes) = load_sd_component(&paths.vae_decoder) else {
        eprintln!("Skipping test_sd_variable_resolution: VAE decoder not found");
        return;
    };

    let model = parse_model(&onnx_bytes).expect("Failed to parse VAE decoder");
    let graph = model.graph.as_ref().expect("Model should have graph");

    // Find latent input
    let latent_input = graph.input.iter()
        .filter(|i| !graph.initializer.iter().any(|init| init.name == i.name))
        .find_map(|i| SymbolicShape::from_value_info(i).ok());

    let Some(shape) = latent_input else {
        eprintln!("Could not find latent input shape");
        return;
    };

    // Test various resolutions
    // Latent is 1/8 of image resolution
    let resolutions = [
        (64, 64),   // 512x512 image
        (72, 72),   // 576x576 image
        (96, 96),   // 768x768 image
        (128, 128), // 1024x1024 image
    ];

    for (h, w) in resolutions {
        let mut concrete_dims: Vec<Dim> = Vec::new();
        let dims = shape.dims();

        for (i, dim) in dims.iter().enumerate() {
            match i {
                0 => concrete_dims.push(Dim::Concrete(1)), // batch
                1 => concrete_dims.push(dim.clone()),       // channels (4)
                2 => concrete_dims.push(Dim::Concrete(h)),  // height
                3 => concrete_dims.push(Dim::Concrete(w)),  // width
                _ => concrete_dims.push(dim.clone()),
            }
        }

        let concrete_shape = SymbolicShape::new(concrete_dims.into_iter().map(|d| {
            match d {
                Dim::Concrete(n) => hologram_onnx_core::Dim::Concrete(n),
                Dim::Var(name) => hologram_onnx_core::Dim::Var(name),
                Dim::Expr(expr) => hologram_onnx_core::Dim::Expr(expr),
            }
        }).collect());

        eprintln!("SD latent {}x{} (image {}x{}): {:?}",
            h, w, h * 8, w * 8, concrete_shape.dims());
    }
}

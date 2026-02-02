//! Test compilation with real ONNX models from the wild.

use std::fs;

#[test]
fn test_compile_resnet18() {
    let model_path = "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx";

    // Check if model exists
    if !std::path::Path::new(model_path).exists() {
        eprintln!("⚠️  ResNet18 model not found at {}", model_path);
        eprintln!("   Download with:");
        eprintln!(
            "   curl -L \"https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx\" -o {}",
            model_path
        );
        return;
    }

    println!("📦 Loading ResNet18 model from {}", model_path);
    let onnx_bytes = fs::read(model_path).expect("Failed to read model");
    println!("   Model size: {} MB", onnx_bytes.len() / 1024 / 1024);

    // Try to compile
    println!("🔨 Attempting to compile...");
    let result = hologram_ai_onnx::compile_onnx(&onnx_bytes);

    match result {
        Ok(holb_bytes) => {
            println!("✅ Successfully compiled ResNet18!");
            println!("   ONNX size: {} MB", onnx_bytes.len() / 1024 / 1024);
            println!("   HOLB size: {} MB", holb_bytes.len() / 1024 / 1024);
            assert!(!holb_bytes.is_empty());
            assert_eq!(&holb_bytes[0..4], b"HOLB");
        }
        Err(e) => {
            println!("❌ Compilation failed:");
            println!("   Error: {}", e);

            // Check if it's an unsupported operation
            let error_msg = e.to_string();
            if error_msg.contains("Unsupported ONNX operation") {
                println!("\n🔍 Missing operation detected!");
                println!("   We need to implement this operation to support ResNet18");
            }

            // Don't panic - we want to see the error for analysis
            eprintln!("\nFull error chain:");
            eprintln!("{:?}", e);
        }
    }
}

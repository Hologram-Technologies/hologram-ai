//! Classify an image using ResNet18.

use anyhow::{Context, Result};
use hologram::BackendPlan;
use hologram::backend::{Backend, cpu::CpuBackend};
use hologram::holo::HolbReader;
use image::GenericImageView;
use std::fs;

// ImageNet normalization constants
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

// Top ImageNet class names (subset for common predictions)
const IMAGENET_CLASSES: &[&str] = &[
    "tench",
    "goldfish",
    "great white shark",
    "tiger shark",
    "hammerhead", // 0-4
    "electric ray",
    "stingray",
    "cock",
    "hen",
    "ostrich", // 5-9
               // ... skip to cat classes
];

// Cat-related class indices in ImageNet
const CAT_CLASSES: &[(usize, &str)] = &[
    (281, "tabby cat"),
    (282, "tiger cat"),
    (283, "Persian cat"),
    (284, "Siamese cat"),
    (285, "Egyptian cat"),
    (286, "cougar/mountain lion"),
    (287, "lynx"),
    (288, "leopard"),
    (289, "snow leopard"),
    (290, "jaguar"),
    (291, "lion"),
    (292, "tiger"),
];

fn get_class_name(idx: usize) -> String {
    // Check if it's a cat class
    for (cat_idx, name) in CAT_CLASSES {
        if *cat_idx == idx {
            return name.to_string();
        }
    }
    // Common ImageNet classes
    match idx {
        0 => "tench".to_string(),
        1 => "goldfish".to_string(),
        207 => "golden retriever".to_string(),
        208 => "Labrador retriever".to_string(),
        232 => "border collie".to_string(),
        259 => "Pomeranian".to_string(),
        263 => "Pembroke Welsh Corgi".to_string(),
        388 => "giant panda".to_string(),
        409 => "analog clock".to_string(),
        417 => "balloon".to_string(),
        531 => "digital watch".to_string(),
        539 => "doormat".to_string(),
        553 => "fire engine".to_string(),
        607 => "jigsaw puzzle".to_string(),
        673 => "mouse".to_string(),
        717 => "pickup truck".to_string(),
        753 => "racket".to_string(),
        779 => "school bus".to_string(),
        804 => "ski".to_string(),
        828 => "streetcar".to_string(),
        847 => "tank".to_string(),
        849 => "teapot".to_string(),
        850 => "teddy bear".to_string(),
        876 => "tub".to_string(),
        880 => "umbrella".to_string(),
        898 => "water bottle".to_string(),
        920 => "traffic light".to_string(),
        948 => "strawberry".to_string(),
        949 => "orange".to_string(),
        950 => "lemon".to_string(),
        951 => "fig".to_string(),
        952 => "pineapple".to_string(),
        953 => "banana".to_string(),
        954 => "jackfruit".to_string(),
        _ => format!("class_{}", idx),
    }
}

fn main() -> Result<()> {
    let image_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/tmp/claude-1000/-workspace/467e61d2-7fce-494d-9b59-dc23e7754a46/scratchpad/cat.jpg"
            .to_string()
    });

    let onnx_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/tmp/claude-1000/-workspace/d42433ba-a7d4-4e38-bcb7-e07ce8361e75/scratchpad/onnx_models/resnet18.onnx".to_string()
    });

    println!("Image: {}", image_path);
    println!("Model: {}", onnx_path);

    // Load and preprocess image
    println!("\nLoading image...");
    let img = image::open(&image_path).context("Failed to load image")?;
    let (orig_w, orig_h) = img.dimensions();
    println!("  Original size: {}x{}", orig_w, orig_h);

    // Resize to 224x224
    let resized = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Convert to CHW format with ImageNet normalization
    let mut input_data = vec![0.0f32; 3 * 224 * 224];
    for y in 0..224 {
        for x in 0..224 {
            let pixel = rgb.get_pixel(x, y);
            for c in 0..3 {
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - MEAN[c]) / STD[c];
                input_data[c * 224 * 224 + y as usize * 224 + x as usize] = normalized;
            }
        }
    }
    println!("  Preprocessed to 224x224, normalized");

    // Load and compile model
    println!("\nCompiling model...");
    let onnx_bytes = fs::read(&onnx_path).context("Failed to read ONNX model")?;
    let holb_bytes = hologram_ai_onnx::compile_onnx(&onnx_bytes)?;

    let reader = HolbReader::from_bytes(&holb_bytes)?;
    let plan: BackendPlan = rkyv::from_bytes(reader.graph())
        .map_err(|e| anyhow::anyhow!("Deserialize error: {}", e))?;
    println!(
        "  Compiled: {} instructions, {} buffers",
        plan.instructions.len(),
        plan.buffers.len()
    );

    // Run inference
    println!("\nRunning inference...");
    let input_bytes: Vec<u8> = bytemuck::cast_slice(&input_data).to_vec();
    let mut output_data: Vec<f32> = vec![0.0; 1000];
    let output_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut output_data);

    let backend = CpuBackend::new();
    backend.execute_plan(&plan, &[&input_bytes], &mut [output_bytes])?;

    // Find top-5 predictions
    let mut indexed: Vec<(usize, f32)> = output_data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Apply softmax for probabilities
    let max_logit = indexed[0].1;
    let exp_vals: Vec<f32> = output_data.iter().map(|&x| (x - max_logit).exp()).collect();
    let exp_sum: f32 = exp_vals.iter().sum();

    println!("\n========================================");
    println!("Top-5 Predictions:");
    println!("========================================");
    for (i, (class_idx, logit)) in indexed.iter().take(5).enumerate() {
        let prob = (output_data[*class_idx] - max_logit).exp() / exp_sum;
        let name = get_class_name(*class_idx);
        println!(
            "  {}. {:25} {:5.2}%  (logit: {:.2})",
            i + 1,
            name,
            prob * 100.0,
            logit
        );
    }
    println!("========================================");

    Ok(())
}

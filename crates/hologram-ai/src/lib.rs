//! Unified AI model compiler for hologram.
//!
//! This crate provides a unified interface for compiling AI models from various
//! formats to hologram's .holo format. Supported formats:
//!
//! - **ONNX**: Open Neural Network Exchange format (feature: `onnx`)
//! - **GGUF**: llama.cpp quantized format (feature: `gguf`)
//! - **SafeTensors**: HuggingFace model format (feature: `safetensors`)
//!
//! ## Feature Flags
//!
//! - `onnx` - Enable ONNX format support (default)
//! - `gguf` - Enable GGUF format support (default)
//! - `safetensors` - Enable SafeTensors format support (default)
//! - `image-output` - Enable image output handlers
//! - `audio-output` - Enable audio output handlers
//! - `text-output` - Enable text output handlers
//!
//! ## Example
//!
//! ```ignore
//! use hologram_ai::{compile_model, ModelFormat};
//!
//! // Auto-detect format from path
//! let (holo, weights) = compile_model("model.onnx", None)?;
//!
//! // Or specify format explicitly
//! let (holo, weights) = compile_model("model.bin", Some(ModelFormat::Gguf))?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod cli;
pub mod config;
pub mod runtime;
pub mod tokenizers;

// Re-export common types
pub use hologram_ai_common::{CommonError, WeightDtype, WeightMap, WeightTensor};

// Conditionally re-export format-specific crates
#[cfg(feature = "onnx")]
pub use hologram_ai_onnx as onnx;

#[cfg(feature = "gguf")]
pub use hologram_ai_gguf as gguf;

#[cfg(feature = "safetensors")]
pub use hologram_ai_safetensors as safetensors;

/// Model format variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// ONNX format (.onnx files)
    #[cfg(feature = "onnx")]
    Onnx,
    /// GGUF format (.gguf files)
    #[cfg(feature = "gguf")]
    Gguf,
    /// SafeTensors format (directory with .safetensors files)
    #[cfg(feature = "safetensors")]
    SafeTensors,
}

impl ModelFormat {
    /// Detect format from file path.
    pub fn detect(path: &str) -> Option<Self> {
        use std::path::Path;
        let path = Path::new(path);

        // Check if directory (SafeTensors)
        if path.is_dir() {
            #[cfg(feature = "safetensors")]
            {
                // Check for config.json and .safetensors files
                let config_exists = path.join("config.json").exists();
                let has_safetensors =
                    std::fs::read_dir(path)
                        .ok()?
                        .filter_map(|e| e.ok())
                        .any(|e| {
                            e.path()
                                .extension()
                                .map(|ext| ext == "safetensors")
                                .unwrap_or(false)
                        });
                if config_exists && has_safetensors {
                    return Some(Self::SafeTensors);
                }
            }
            return None;
        }

        // Check extension for files
        match path.extension()?.to_str()? {
            #[cfg(feature = "onnx")]
            "onnx" => Some(Self::Onnx),
            #[cfg(feature = "gguf")]
            "gguf" => Some(Self::Gguf),
            _ => None,
        }
    }
}

/// Compile a model from any supported format.
///
/// # Arguments
/// * `path` - Path to the model file or directory
/// * `format` - Optional explicit format (auto-detected if None)
///
/// # Returns
/// Tuple of (holo_bytes, weight_bytes)
pub fn compile_model(
    path: &str,
    format: Option<ModelFormat>,
) -> Result<(Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let format = format
        .or_else(|| ModelFormat::detect(path))
        .ok_or("Could not detect model format from path")?;

    match format {
        #[cfg(feature = "onnx")]
        ModelFormat::Onnx => {
            let bytes = std::fs::read(path)?;
            let holb_bytes = hologram_ai_onnx::compile_onnx(&bytes)?;
            // ONNX compilation produces a single .holb file with embedded weights
            Ok((holb_bytes, Vec::new()))
        }
        #[cfg(feature = "gguf")]
        ModelFormat::Gguf => {
            // TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
            let _ = hologram_ai_gguf::GgufCompiler::new();
            Err("GGUF compilation is temporarily disabled".into())
        }
        #[cfg(feature = "safetensors")]
        ModelFormat::SafeTensors => {
            // TEMPORARILY DISABLED: transformer module is disabled in hologram-ai-common
            let _ = hologram_ai_safetensors::SafeTensorsCompiler::new();
            Err("SafeTensors compilation is temporarily disabled".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "onnx")]
    fn test_format_detect_onnx() {
        assert_eq!(ModelFormat::detect("model.onnx"), Some(ModelFormat::Onnx));
    }

    #[test]
    #[cfg(feature = "gguf")]
    fn test_format_detect_gguf() {
        assert_eq!(ModelFormat::detect("model.gguf"), Some(ModelFormat::Gguf));
    }

    #[test]
    fn test_format_detect_unknown() {
        assert_eq!(ModelFormat::detect("model.txt"), None);
    }
}

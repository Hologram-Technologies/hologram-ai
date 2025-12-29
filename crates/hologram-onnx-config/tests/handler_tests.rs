//! Integration tests for hologram-onnx-config output handlers.
//!
//! These tests verify the full pipeline: config → handler → process → save.

use std::collections::HashMap;
use std::fs;
use tempfile::TempDir;

use hologram_onnx_config::{
    ConfigError, OutputHandler, OutputHandlerConfig, OutputHandlerRegistry, PipelineConfig,
    ProcessedOutput, TensorData,
};

// ============================================================================
// Test Fixtures and Helpers
// ============================================================================

/// Create a minimal pipeline config for testing.
fn minimal_config() -> PipelineConfig {
    let toml = r#"
        [pipeline]
        name = "test-pipeline"
        version = "1.0"
        description = "Integration test pipeline"

        [pipeline.execution]
        inputs = ["input"]
        outputs = ["output"]
    "#;
    PipelineConfig::from_str(toml).unwrap()
}

/// Create mock image tensor data (NCHW format, [0,1] range).
///
/// Returns a 4D tensor [batch=1, channels=3, height, width] with gradient pattern.
fn mock_image_tensor_nchw(height: usize, width: usize) -> TensorData {
    let channels = 3;
    let mut data = Vec::with_capacity(channels * height * width);

    // Create RGB gradient pattern
    for c in 0..channels {
        for h in 0..height {
            for w in 0..width {
                // Each channel gets a different gradient
                let value = match c {
                    0 => h as f32 / height as f32, // R: vertical gradient
                    1 => w as f32 / width as f32,  // G: horizontal gradient
                    2 => 0.5,                      // B: constant mid-gray
                    _ => 0.0,
                };
                data.push(value);
            }
        }
    }

    TensorData::new(data, vec![1, channels, height, width])
}

/// Create mock image tensor data (NHWC format, [0,1] range).
fn mock_image_tensor_nhwc(height: usize, width: usize) -> TensorData {
    let channels = 3;
    let mut data = Vec::with_capacity(height * width * channels);

    for h in 0..height {
        for w in 0..width {
            // RGB interleaved
            data.push(h as f32 / height as f32); // R
            data.push(w as f32 / width as f32); // G
            data.push(0.5); // B
        }
    }

    TensorData::new(data, vec![1, height, width, channels])
}

/// Create mock grayscale image tensor (NCHW).
fn mock_grayscale_tensor(height: usize, width: usize) -> TensorData {
    let mut data = Vec::with_capacity(height * width);

    for h in 0..height {
        for _w in 0..width {
            data.push(h as f32 / height as f32);
        }
    }

    TensorData::new(data, vec![1, 1, height, width])
}

/// Create mock audio tensor (mono, 1 second at given sample rate).
fn mock_audio_tensor_mono(sample_rate: usize) -> TensorData {
    let mut data = Vec::with_capacity(sample_rate);

    // Generate 440Hz sine wave (A4 note)
    for i in 0..sample_rate {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        data.push(sample);
    }

    TensorData::new(data, vec![sample_rate])
}

/// Create mock audio tensor (stereo, 1 second).
fn mock_audio_tensor_stereo(sample_rate: usize) -> TensorData {
    let mut data = Vec::with_capacity(sample_rate * 2);

    // Generate stereo sine wave (440Hz left, 880Hz right)
    for i in 0..sample_rate {
        let t = i as f32 / sample_rate as f32;
        // Left channel: 440Hz
        let left = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5;
        // Right channel: 880Hz
        let right = (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.5;
        data.push(left);
        data.push(right);
    }

    TensorData::new(data, vec![2, sample_rate])
}

/// Create mock token ID tensor for text output.
#[allow(dead_code)]
fn mock_token_ids() -> TensorData {
    // Common token IDs (these would decode to something like "Hello world")
    let data: Vec<f32> = vec![15496.0, 995.0, 0.0]; // "Hello world" + padding in GPT-2
    TensorData::new(data, vec![3])
}

// ============================================================================
// Config Loading Integration Tests
// ============================================================================

#[test]
fn test_config_from_file_roundtrip() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("pipeline.toml");

    // Create config programmatically
    let original = minimal_config();

    // Save to file
    original.to_file(&config_path).unwrap();

    // Verify file exists
    assert!(config_path.exists());

    // Load from file
    let loaded = PipelineConfig::from_file(&config_path).unwrap();

    // Verify roundtrip
    assert_eq!(original.pipeline.name, loaded.pipeline.name);
    assert_eq!(original.pipeline.version, loaded.pipeline.version);
}

#[test]
fn test_config_with_all_handler_types() {
    let toml = r#"
        [pipeline]
        name = "multi-modal"
        version = "1.0"

        [pipeline.execution]
        inputs = ["text", "seed"]
        outputs = ["image", "audio", "tokens"]

        [pipeline.execution.output_handlers.image_out]
        type = "image"
        output = "image"
        format = "rgb"
        layout = "NCHW"
        value_range = "zero_one"

        [pipeline.execution.output_handlers.audio_out]
        type = "audio"
        output = "audio"
        sample_rate = 44100
        channels = 2
        sample_format = "float32"

        [pipeline.execution.output_handlers.text_out]
        type = "text"
        output = "tokens"
        tokenizer_path = "tokenizer.json"
        skip_special_tokens = true
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();
    let handlers = config.output_handlers();

    assert_eq!(handlers.len(), 3);
    assert!(handlers.contains_key("image_out"));
    assert!(handlers.contains_key("audio_out"));
    assert!(handlers.contains_key("text_out"));

    // Verify image handler config
    let img = handlers.get("image_out").unwrap();
    assert_eq!(img.handler_type, "image");
    assert_eq!(img.get_string("format"), Some("rgb"));
    assert_eq!(img.get_string("layout"), Some("NCHW"));

    // Verify audio handler config
    let audio = handlers.get("audio_out").unwrap();
    assert_eq!(audio.handler_type, "audio");
    assert_eq!(audio.get_int("sample_rate"), Some(44100));
    assert_eq!(audio.get_int("channels"), Some(2));

    // Verify text handler config
    let text = handlers.get("text_out").unwrap();
    assert_eq!(text.handler_type, "text");
    assert_eq!(text.get_bool("skip_special_tokens"), Some(true));
}

#[test]
fn test_config_with_multi_stage_pipeline() {
    let toml = r#"
        [pipeline]
        name = "stable-diffusion"
        version = "1.0"
        description = "Text-to-image pipeline"

        [pipeline.execution]
        inputs = ["prompt", "negative_prompt", "seed"]
        outputs = ["image"]

        [[pipeline.execution.stages]]
        name = "text_encoder"
        model = "models/text_encoder.holo"
        inputs = { prompt = "input_ids", negative_prompt = "neg_input_ids" }
        outputs = ["text_embeddings", "negative_embeddings"]

        [[pipeline.execution.stages]]
        name = "unet"
        model = "models/unet.holo"
        inputs = { text_embeddings = "encoder_hidden_states", seed = "latent_seed" }
        outputs = ["latents"]

        [[pipeline.execution.stages]]
        name = "vae_decoder"
        model = "models/vae_decoder.holo"
        inputs = { latents = "latent_sample" }
        outputs = ["image"]

        [pipeline.execution.output_handlers.image]
        type = "image"
        output = "image"
        format = "rgb"
        layout = "NCHW"
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();
    let exec = config.execution().unwrap();

    // Verify stages
    let stages = exec.stages.as_ref().unwrap();
    assert_eq!(stages.len(), 3);

    // Stage 1: text encoder
    assert_eq!(stages[0].name, "text_encoder");
    assert_eq!(stages[0].model, "models/text_encoder.holo");
    assert_eq!(
        stages[0].inputs.get("prompt"),
        Some(&"input_ids".to_string())
    );

    // Stage 2: unet
    assert_eq!(stages[1].name, "unet");
    assert_eq!(stages[1].outputs, vec!["latents"]);

    // Stage 3: vae decoder
    assert_eq!(stages[2].name, "vae_decoder");
    assert_eq!(stages[2].outputs, vec!["image"]);
}

#[test]
fn test_config_validation_errors() {
    // Empty inputs should fail
    let toml = r#"
        [pipeline]
        name = "invalid"
        version = "1.0"

        [pipeline.execution]
        inputs = []
        outputs = ["out"]
    "#;

    let result = PipelineConfig::from_str(toml);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("inputs"));

    // Empty outputs should fail
    let toml = r#"
        [pipeline]
        name = "invalid"
        version = "1.0"

        [pipeline.execution]
        inputs = ["in"]
        outputs = []
    "#;

    let result = PipelineConfig::from_str(toml);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("outputs"));
}

#[test]
fn test_config_missing_file_error() {
    let result = PipelineConfig::from_file("/nonexistent/path/config.toml");
    assert!(result.is_err());
}

#[test]
fn test_config_malformed_toml_error() {
    let toml = r#"
        [pipeline
        name = "broken"
    "#;

    let result = PipelineConfig::from_str(toml);
    assert!(result.is_err());
}

// ============================================================================
// Handler Registry Integration Tests
// ============================================================================

#[test]
fn test_registry_creation() {
    let _registry = OutputHandlerRegistry::new();
    // Registry should be created successfully
    // Available handlers depend on enabled features
    assert!(true); // Registry created without panic
}

#[test]
fn test_registry_unknown_handler_error() {
    let registry = OutputHandlerRegistry::new();

    let config = OutputHandlerConfig {
        handler_type: "unknown_type".to_string(),
        output: "tensor".to_string(),
        config: HashMap::new(),
    };

    let result = registry.create_handler(&config);
    assert!(result.is_err());

    let err = result.err().unwrap();
    assert!(err.to_string().contains("unknown") || err.to_string().contains("Unknown"));
}

#[test]
fn test_registry_create_multiple_handlers() {
    let registry = OutputHandlerRegistry::new();

    let mut configs = HashMap::new();

    // Add image handler config (if feature enabled, will succeed)
    #[cfg(feature = "image-output")]
    {
        let mut image_config = HashMap::new();
        image_config.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        configs.insert(
            "image".to_string(),
            OutputHandlerConfig {
                handler_type: "image".to_string(),
                output: "sample".to_string(),
                config: image_config,
            },
        );
    }

    // Add audio handler config (if feature enabled)
    #[cfg(feature = "audio-output")]
    {
        let mut audio_config = HashMap::new();
        audio_config.insert("sample_rate".to_string(), toml::Value::Integer(44100));
        audio_config.insert("channels".to_string(), toml::Value::Integer(2));

        configs.insert(
            "audio".to_string(),
            OutputHandlerConfig {
                handler_type: "audio".to_string(),
                output: "waveform".to_string(),
                config: audio_config,
            },
        );
    }

    if !configs.is_empty() {
        let handlers = registry.create_handlers(&configs).unwrap();
        assert_eq!(handlers.len(), configs.len());
    }
}

// ============================================================================
// Image Handler Integration Tests (Feature-gated)
// ============================================================================

#[cfg(feature = "image-output")]
mod image_handler_tests {
    use super::*;
    use hologram_onnx_config::ImageHandler;

    #[test]
    fn test_image_handler_rgb_nchw_full_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("output.png");

        // Create handler from config
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
        config_map.insert(
            "layout".to_string(),
            toml::Value::String("NCHW".to_string()),
        );
        config_map.insert(
            "value_range".to_string(),
            toml::Value::String("zero_one".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        // Create mock tensor data
        let tensor = mock_image_tensor_nchw(64, 64);

        // Create outputs map
        let mut outputs = HashMap::new();
        outputs.insert("sample".to_string(), tensor);

        // Process
        let result = handler.process(&outputs).unwrap();

        // Verify processed output
        match &result {
            ProcessedOutput::Image(img) => {
                assert_eq!(img.width, 64);
                assert_eq!(img.height, 64);
                assert_eq!(img.channels, 3);
                assert_eq!(img.data.len(), 64 * 64 * 3);
            }
            _ => panic!("Expected Image output"),
        }

        // Save to file
        handler.save(&result, &output_path).unwrap();

        // Verify file was created
        assert!(output_path.exists());
        let file_size = fs::metadata(&output_path).unwrap().len();
        assert!(file_size > 0);
    }

    #[test]
    fn test_image_handler_rgb_nhwc_full_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("output.png");

        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
        config_map.insert(
            "layout".to_string(),
            toml::Value::String("NHWC".to_string()),
        );
        config_map.insert(
            "value_range".to_string(),
            toml::Value::String("zero_one".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let tensor = mock_image_tensor_nhwc(128, 128);

        let mut outputs = HashMap::new();
        outputs.insert("sample".to_string(), tensor);

        let result = handler.process(&outputs).unwrap();

        match &result {
            ProcessedOutput::Image(img) => {
                assert_eq!(img.width, 128);
                assert_eq!(img.height, 128);
                assert_eq!(img.channels, 3);
            }
            _ => panic!("Expected Image output"),
        }

        handler.save(&result, &output_path).unwrap();
        assert!(output_path.exists());
    }

    #[test]
    fn test_image_handler_grayscale_full_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("output.png");

        let mut config_map = HashMap::new();
        config_map.insert(
            "format".to_string(),
            toml::Value::String("grayscale".to_string()),
        );
        config_map.insert(
            "layout".to_string(),
            toml::Value::String("NCHW".to_string()),
        );
        config_map.insert(
            "value_range".to_string(),
            toml::Value::String("zero_one".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let tensor = mock_grayscale_tensor(32, 32);

        let mut outputs = HashMap::new();
        outputs.insert("sample".to_string(), tensor);

        let result = handler.process(&outputs).unwrap();

        match &result {
            ProcessedOutput::Image(img) => {
                assert_eq!(img.width, 32);
                assert_eq!(img.height, 32);
                assert_eq!(img.channels, 1);
            }
            _ => panic!("Expected Image output"),
        }

        handler.save(&result, &output_path).unwrap();
        assert!(output_path.exists());
    }

    #[test]
    fn test_image_handler_missing_tensor_error() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "missing_tensor".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let outputs = HashMap::new(); // Empty - no tensors

        let result = handler.process(&outputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_image_handler_large_image() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("large.png");

        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
        config_map.insert(
            "layout".to_string(),
            toml::Value::String("NCHW".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        // Create 512x512 image
        let tensor = mock_image_tensor_nchw(512, 512);

        let mut outputs = HashMap::new();
        outputs.insert("sample".to_string(), tensor);

        let result = handler.process(&outputs).unwrap();
        handler.save(&result, &output_path).unwrap();

        assert!(output_path.exists());
        let file_size = fs::metadata(&output_path).unwrap().len();
        assert!(file_size > 0);
    }

    #[test]
    fn test_image_handler_via_registry() {
        let registry = OutputHandlerRegistry::new();

        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = registry.create_handler(&config).unwrap();
        assert_eq!(handler.handler_type(), "image");
    }
}

// ============================================================================
// Audio Handler Integration Tests (Feature-gated)
// ============================================================================

#[cfg(feature = "audio-output")]
mod audio_handler_tests {
    use super::*;
    use hologram_onnx_config::AudioHandler;

    #[test]
    fn test_audio_handler_mono_full_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("output.wav");

        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(44100));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));
        config_map.insert(
            "sample_format".to_string(),
            toml::Value::String("int16".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let tensor = mock_audio_tensor_mono(44100);

        let mut outputs = HashMap::new();
        outputs.insert("waveform".to_string(), tensor);

        let result = handler.process(&outputs).unwrap();

        match &result {
            ProcessedOutput::Audio(audio) => {
                assert_eq!(audio.sample_rate, 44100);
                assert_eq!(audio.channels, 1);
                assert_eq!(audio.samples.len(), 44100);
            }
            _ => panic!("Expected Audio output"),
        }

        handler.save(&result, &output_path).unwrap();

        assert!(output_path.exists());
        let file_size = fs::metadata(&output_path).unwrap().len();
        // WAV header + samples: should be > 44 bytes (header) + samples
        assert!(file_size > 44);
    }

    #[test]
    fn test_audio_handler_stereo_full_pipeline() {
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("stereo.wav");

        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(48000));
        config_map.insert("channels".to_string(), toml::Value::Integer(2));
        config_map.insert(
            "sample_format".to_string(),
            toml::Value::String("float32".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let tensor = mock_audio_tensor_stereo(48000);

        let mut outputs = HashMap::new();
        outputs.insert("waveform".to_string(), tensor);

        let result = handler.process(&outputs).unwrap();

        match &result {
            ProcessedOutput::Audio(audio) => {
                assert_eq!(audio.sample_rate, 48000);
                assert_eq!(audio.channels, 2);
            }
            _ => panic!("Expected Audio output"),
        }

        handler.save(&result, &output_path).unwrap();
        assert!(output_path.exists());
    }

    #[test]
    fn test_audio_handler_missing_tensor_error() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(44100));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "missing".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let outputs = HashMap::new();

        let result = handler.process(&outputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_handler_via_registry() {
        let registry = OutputHandlerRegistry::new();

        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(44100));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let handler = registry.create_handler(&config).unwrap();
        assert_eq!(handler.handler_type(), "audio");
    }

    #[test]
    fn test_audio_handler_invalid_sample_rate() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(0));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let result = AudioHandler::from_config(&config);
        assert!(result.is_err());
    }
}

// ============================================================================
// Multi-Handler Coordination Tests
// ============================================================================

#[test]
fn test_multi_handler_creation_from_config() {
    let toml = r#"
        [pipeline]
        name = "multi-modal"
        version = "1.0"

        [pipeline.execution]
        inputs = ["input"]
        outputs = ["image", "audio"]

        [pipeline.execution.output_handlers.image]
        type = "image"
        output = "image_tensor"
        format = "rgb"

        [pipeline.execution.output_handlers.audio]
        type = "audio"
        output = "audio_tensor"
        sample_rate = 44100
        channels = 1
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();
    let handlers_config = config.output_handlers();

    assert_eq!(handlers_config.len(), 2);

    // Create registry and handlers
    let registry = OutputHandlerRegistry::new();

    // Convert to owned HashMap
    let owned_configs: HashMap<String, OutputHandlerConfig> = handlers_config
        .iter()
        .map(|(k, v)| (k.clone(), (*v).clone()))
        .collect();

    // This will only succeed if both features are enabled
    #[cfg(all(feature = "image-output", feature = "audio-output"))]
    {
        let handlers = registry.create_handlers(&owned_configs).unwrap();
        assert_eq!(handlers.len(), 2);
        assert!(handlers.contains_key("image"));
        assert!(handlers.contains_key("audio"));
    }
}

#[cfg(all(feature = "image-output", feature = "audio-output"))]
#[test]
fn test_multi_handler_processing() {
    use hologram_onnx_config::{AudioHandler, ImageHandler};

    let temp_dir = TempDir::new().unwrap();

    // Create image handler
    let mut img_config = HashMap::new();
    img_config.insert("format".to_string(), toml::Value::String("rgb".to_string()));
    img_config.insert(
        "layout".to_string(),
        toml::Value::String("NCHW".to_string()),
    );

    let image_handler = ImageHandler::from_config(&OutputHandlerConfig {
        handler_type: "image".to_string(),
        output: "image".to_string(),
        config: img_config,
    })
    .unwrap();

    // Create audio handler
    let mut audio_config = HashMap::new();
    audio_config.insert("sample_rate".to_string(), toml::Value::Integer(16000));
    audio_config.insert("channels".to_string(), toml::Value::Integer(1));

    let audio_handler = AudioHandler::from_config(&OutputHandlerConfig {
        handler_type: "audio".to_string(),
        output: "audio".to_string(),
        config: audio_config,
    })
    .unwrap();

    // Create mock outputs (simulating model output)
    let mut outputs = HashMap::new();
    outputs.insert("image".to_string(), mock_image_tensor_nchw(64, 64));
    outputs.insert("audio".to_string(), mock_audio_tensor_mono(16000));

    // Process both
    let image_result = image_handler.process(&outputs).unwrap();
    let audio_result = audio_handler.process(&outputs).unwrap();

    // Save both
    let image_path = temp_dir.path().join("output.png");
    let audio_path = temp_dir.path().join("output.wav");

    image_handler.save(&image_result, &image_path).unwrap();
    audio_handler.save(&audio_result, &audio_path).unwrap();

    // Verify both files exist
    assert!(image_path.exists());
    assert!(audio_path.exists());

    // Verify both have content
    assert!(fs::metadata(&image_path).unwrap().len() > 0);
    assert!(fs::metadata(&audio_path).unwrap().len() > 0);
}

// ============================================================================
// TensorData Tests
// ============================================================================

#[test]
fn test_tensor_data_creation_and_access() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let shape = vec![2, 3];
    let tensor = TensorData::new(data.clone(), shape.clone());

    assert_eq!(tensor.len(), 6);
    assert_eq!(tensor.ndim(), 2);
    assert!(!tensor.is_empty());
    assert_eq!(tensor.shape, vec![2, 3]);
    assert_eq!(tensor.data, data);
}

#[test]
fn test_tensor_data_empty() {
    let tensor = TensorData::new(vec![], vec![0]);
    assert!(tensor.is_empty());
    assert_eq!(tensor.len(), 0);
}

#[test]
fn test_tensor_data_4d() {
    // Simulate NCHW tensor
    let tensor = TensorData::new(vec![0.0; 1 * 3 * 64 * 64], vec![1, 3, 64, 64]);

    assert_eq!(tensor.ndim(), 4);
    assert_eq!(tensor.len(), 1 * 3 * 64 * 64);
    assert_eq!(tensor.shape[0], 1); // batch
    assert_eq!(tensor.shape[1], 3); // channels
    assert_eq!(tensor.shape[2], 64); // height
    assert_eq!(tensor.shape[3], 64); // width
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_config_error_display() {
    let err = ConfigError::missing_field("test_field");
    let msg = err.to_string();
    assert!(msg.contains("test_field"));
}

#[test]
fn test_feature_not_enabled_error_message() {
    let err = ConfigError::feature_not_enabled("image", "image-output");
    let msg = err.to_string();
    assert!(msg.contains("image"));
    assert!(msg.contains("image-output"));
}

#[test]
fn test_unknown_handler_type_error() {
    let err = ConfigError::unknown_handler_type("custom_handler");
    let msg = err.to_string();
    assert!(msg.contains("custom_handler") || msg.contains("Unknown"));
}

// ============================================================================
// File I/O Error Tests
// ============================================================================

#[test]
fn test_config_save_to_readonly_dir() {
    // This test may not work on all systems, but demonstrates the pattern
    let result = PipelineConfig::from_str(
        r#"
        [pipeline]
        name = "test"
        version = "1.0"
        [pipeline.execution]
        inputs = ["in"]
        outputs = ["out"]
    "#,
    )
    .unwrap();

    // Try to save to a non-existent path
    let save_result = result.to_file("/nonexistent/deeply/nested/path/config.toml");
    assert!(save_result.is_err());
}

// ============================================================================
// Real-World Config Examples
// ============================================================================

#[test]
fn test_stable_diffusion_config() {
    let toml = r#"
        [pipeline]
        name = "stable-diffusion-v1.5"
        version = "1.5"
        description = "Text-to-image generation with Stable Diffusion"

        [pipeline.execution]
        inputs = ["prompt", "negative_prompt", "seed", "num_inference_steps"]
        outputs = ["image"]

        [[pipeline.execution.stages]]
        name = "text_encoder"
        model = "models/clip_text_encoder.holo"
        inputs = { prompt = "input_ids" }
        outputs = ["text_embeddings"]

        [[pipeline.execution.stages]]
        name = "unet"
        model = "models/unet.holo"
        inputs = { text_embeddings = "encoder_hidden_states", seed = "latent" }
        outputs = ["denoised_latents"]

        [[pipeline.execution.stages]]
        name = "vae_decoder"
        model = "models/vae_decoder.holo"
        inputs = { denoised_latents = "latent_sample" }
        outputs = ["image"]

        [pipeline.execution.output_handlers.image]
        type = "image"
        output = "image"
        format = "rgb"
        layout = "NCHW"
        value_range = "neg_one_one"
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();

    assert_eq!(config.pipeline.name, "stable-diffusion-v1.5");
    assert_eq!(config.pipeline.version, "1.5");

    let exec = config.execution().unwrap();
    assert_eq!(exec.inputs.len(), 4);
    assert!(exec.inputs.contains(&"prompt".to_string()));
    assert!(exec.inputs.contains(&"seed".to_string()));

    let stages = exec.stages.as_ref().unwrap();
    assert_eq!(stages.len(), 3);

    let handlers = config.output_handlers();
    assert_eq!(handlers.len(), 1);
    let img = handlers.get("image").unwrap();
    assert_eq!(img.get_string("value_range"), Some("neg_one_one"));
}

#[test]
fn test_whisper_config() {
    let toml = r#"
        [pipeline]
        name = "whisper-large-v3"
        version = "3.0"
        description = "Speech-to-text with Whisper"

        [pipeline.execution]
        inputs = ["audio"]
        outputs = ["transcription"]

        [[pipeline.execution.stages]]
        name = "encoder"
        model = "models/whisper_encoder.holo"
        inputs = { audio = "input_features" }
        outputs = ["encoder_hidden_states"]

        [[pipeline.execution.stages]]
        name = "decoder"
        model = "models/whisper_decoder.holo"
        inputs = { encoder_hidden_states = "encoder_hidden_states" }
        outputs = ["token_ids"]

        [pipeline.execution.output_handlers.text]
        type = "text"
        output = "token_ids"
        tokenizer_path = "tokenizers/whisper.json"
        skip_special_tokens = true
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();

    assert_eq!(config.pipeline.name, "whisper-large-v3");

    let stages = config.execution().unwrap().stages.as_ref().unwrap();
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0].name, "encoder");
    assert_eq!(stages[1].name, "decoder");

    let handlers = config.output_handlers();
    let text = handlers.get("text").unwrap();
    assert_eq!(
        text.get_string("tokenizer_path"),
        Some("tokenizers/whisper.json")
    );
}

#[test]
fn test_musicgen_config() {
    let toml = r#"
        [pipeline]
        name = "musicgen-melody"
        version = "1.0"
        description = "Music generation from text and melody"

        [pipeline.execution]
        inputs = ["prompt", "melody"]
        outputs = ["audio"]

        [[pipeline.execution.stages]]
        name = "text_encoder"
        model = "models/t5_encoder.holo"
        inputs = { prompt = "input_ids" }
        outputs = ["text_embeddings"]

        [[pipeline.execution.stages]]
        name = "audio_encoder"
        model = "models/encodec_encoder.holo"
        inputs = { melody = "audio" }
        outputs = ["melody_codes"]

        [[pipeline.execution.stages]]
        name = "decoder"
        model = "models/musicgen_decoder.holo"
        inputs = { text_embeddings = "encoder_hidden_states", melody_codes = "melody" }
        outputs = ["audio_codes"]

        [[pipeline.execution.stages]]
        name = "audio_decoder"
        model = "models/encodec_decoder.holo"
        inputs = { audio_codes = "codes" }
        outputs = ["audio"]

        [pipeline.execution.output_handlers.audio]
        type = "audio"
        output = "audio"
        sample_rate = 32000
        channels = 1
        sample_format = "float32"
    "#;

    let config = PipelineConfig::from_str(toml).unwrap();

    assert_eq!(config.pipeline.name, "musicgen-melody");

    let stages = config.execution().unwrap().stages.as_ref().unwrap();
    assert_eq!(stages.len(), 4);

    let handlers = config.output_handlers();
    let audio = handlers.get("audio").unwrap();
    assert_eq!(audio.get_int("sample_rate"), Some(32000));
}

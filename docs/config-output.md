# Config-Driven Output Handlers

This document describes the implementation plan for config-driven output handlers that allow model authors to specify output types directly in their pipeline configuration files.

## Overview

Building on the existing execution config system, we extend pipeline configs to include:
1. **Output type declarations** - What kind of output the model produces (image, audio, text, etc.)
2. **Output handler configuration** - How to process raw tensor outputs
3. **Built-in operations** - Common post-processing operations as builtins

## Config Extension

### Current Config Structure

```toml
[pipeline.execution]
inputs = ["prompt", "seed", "height", "width"]
outputs = ["image"]

[[pipeline.execution.stages]]
name = "vae_decode"
model = "vae_decoder"
inputs = { latent_sample = "latents * 0.18215" }
outputs = ["vae_output"]
```

### Extended Config with Output Handlers

```toml
[pipeline.execution]
inputs = ["prompt", "seed", "height", "width"]
outputs = ["image"]

# NEW: Output type specification
[pipeline.execution.output_handlers.image]
type = "image"                    # Handler type: image, audio, text, tensor, json
format = "rgb"                    # Pixel format: rgb, rgba, grayscale
layout = "NCHW"                   # Tensor layout: NCHW, NHWC
value_range = "neg_one_one"       # Input range: neg_one_one, zero_one, byte
output_format = "png"             # File format: png, jpg, webp

# For audio models
[pipeline.execution.output_handlers.audio_out]
type = "audio"
sample_rate = 44100
channels = 2
format = "wav"

# For text/LLM models
[pipeline.execution.output_handlers.text_out]
type = "text"
tokenizer = "gpt2"               # Tokenizer to use for decoding
decoding = "greedy"              # greedy, beam_search, top_k, top_p

# For raw tensor output
[pipeline.execution.output_handlers.features]
type = "tensor"
format = "npy"                   # npy, safetensors, json
```

## Implementation Plan

### Phase 1: Output Handler Trait and Registry

**File: `crates/compiler/src/execution/output_handlers/mod.rs`**

```rust
use crate::error::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Output handler trait for converting raw model outputs to typed results
pub trait OutputHandler: Send + Sync {
    /// Name of this handler type (e.g., "image", "audio", "text")
    fn handler_type(&self) -> &'static str;

    /// Process raw tensor outputs and return processed value
    fn process(&self, outputs: &HashMap<String, Arc<Vec<f32>>>) -> Result<ProcessedOutput>;

    /// Save processed output to file
    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<()>;

    /// Validate configuration parameters
    fn validate_config(&self) -> Result<()>;
}

/// Processed output container
pub enum ProcessedOutput {
    Image(ImageOutput),
    Audio(AudioOutput),
    Text(String),
    Tensor(TensorOutput),
    Json(serde_json::Value),
}

/// Image output data
pub struct ImageOutput {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub format: ImageFormat,
}

/// Audio output data
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Raw tensor output
pub struct TensorOutput {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Handler registry for dynamic dispatch
pub struct OutputHandlerRegistry {
    handlers: HashMap<String, Box<dyn Fn(&toml::Value) -> Result<Box<dyn OutputHandler>>>>,
}

impl OutputHandlerRegistry {
    pub fn new() -> Self {
        let mut registry = Self { handlers: HashMap::new() };

        // Register built-in handlers
        registry.register("image", |config| {
            Ok(Box::new(ImageHandler::from_config(config)?))
        });
        registry.register("audio", |config| {
            Ok(Box::new(AudioHandler::from_config(config)?))
        });
        registry.register("text", |config| {
            Ok(Box::new(TextHandler::from_config(config)?))
        });
        registry.register("tensor", |config| {
            Ok(Box::new(TensorHandler::from_config(config)?))
        });

        registry
    }

    pub fn register<F>(&mut self, type_name: &str, factory: F)
    where
        F: Fn(&toml::Value) -> Result<Box<dyn OutputHandler>> + 'static,
    {
        self.handlers.insert(type_name.to_string(), Box::new(factory));
    }

    pub fn create(&self, type_name: &str, config: &toml::Value) -> Result<Box<dyn OutputHandler>> {
        let factory = self.handlers.get(type_name)
            .ok_or_else(|| CompilerError::InvalidModel(
                format!("Unknown output handler type: {}", type_name)
            ))?;
        factory(config)
    }
}
```

### Phase 2: Image Handler Implementation

**File: `crates/compiler/src/execution/output_handlers/image.rs`**

```rust
use super::*;

#[derive(Debug, Clone)]
pub enum ImageFormat {
    Png,
    Jpeg { quality: u8 },
    WebP { quality: u8 },
}

#[derive(Debug, Clone)]
pub enum PixelFormat {
    Rgb,
    Rgba,
    Grayscale,
}

#[derive(Debug, Clone)]
pub enum TensorLayout {
    NCHW,  // [batch, channels, height, width]
    NHWC,  // [batch, height, width, channels]
}

#[derive(Debug, Clone)]
pub enum ValueRange {
    NegOneOne,  // [-1, 1]
    ZeroOne,    // [0, 1]
    Byte,       // [0, 255]
}

pub struct ImageHandler {
    /// Output tensor name to process
    pub output_name: String,
    /// Expected image dimensions
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Pixel format
    pub pixel_format: PixelFormat,
    /// Tensor layout
    pub layout: TensorLayout,
    /// Input value range
    pub value_range: ValueRange,
    /// Output file format
    pub output_format: ImageFormat,
}

impl ImageHandler {
    pub fn from_config(config: &toml::Value) -> Result<Self> {
        let output_name = config.get("output")
            .and_then(|v| v.as_str())
            .unwrap_or("sample")
            .to_string();

        let pixel_format = match config.get("format").and_then(|v| v.as_str()) {
            Some("rgba") => PixelFormat::Rgba,
            Some("grayscale") => PixelFormat::Grayscale,
            _ => PixelFormat::Rgb,
        };

        let layout = match config.get("layout").and_then(|v| v.as_str()) {
            Some("NHWC") => TensorLayout::NHWC,
            _ => TensorLayout::NCHW,
        };

        let value_range = match config.get("value_range").and_then(|v| v.as_str()) {
            Some("zero_one") => ValueRange::ZeroOne,
            Some("byte") => ValueRange::Byte,
            _ => ValueRange::NegOneOne,
        };

        let output_format = match config.get("output_format").and_then(|v| v.as_str()) {
            Some("jpg") | Some("jpeg") => {
                let quality = config.get("quality")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(90) as u8;
                ImageFormat::Jpeg { quality }
            }
            Some("webp") => {
                let quality = config.get("quality")
                    .and_then(|v| v.as_integer())
                    .unwrap_or(90) as u8;
                ImageFormat::WebP { quality }
            }
            _ => ImageFormat::Png,
        };

        Ok(Self {
            output_name,
            width: config.get("width").and_then(|v| v.as_integer()).map(|v| v as usize),
            height: config.get("height").and_then(|v| v.as_integer()).map(|v| v as usize),
            pixel_format,
            layout,
            value_range,
            output_format,
        })
    }

    fn normalize_value(&self, v: f32) -> u8 {
        let normalized = match self.value_range {
            ValueRange::NegOneOne => (v + 1.0) / 2.0,
            ValueRange::ZeroOne => v,
            ValueRange::Byte => v / 255.0,
        };
        (normalized.clamp(0.0, 1.0) * 255.0) as u8
    }
}

impl OutputHandler for ImageHandler {
    fn handler_type(&self) -> &'static str {
        "image"
    }

    fn process(&self, outputs: &HashMap<String, Arc<Vec<f32>>>) -> Result<ProcessedOutput> {
        let data = outputs.get(&self.output_name)
            .ok_or_else(|| CompilerError::InvalidModel(
                format!("Output '{}' not found", self.output_name)
            ))?;

        // Infer dimensions from data size if not specified
        let total = data.len();
        let channels = match self.pixel_format {
            PixelFormat::Rgb => 3,
            PixelFormat::Rgba => 4,
            PixelFormat::Grayscale => 1,
        };

        let (width, height) = if let (Some(w), Some(h)) = (self.width, self.height) {
            (w, h)
        } else {
            // Infer square dimensions
            let pixels = total / channels;
            let side = (pixels as f64).sqrt() as usize;
            (side, side)
        };

        // Convert tensor to image bytes
        let mut pixels = Vec::with_capacity(width * height * channels);

        match self.layout {
            TensorLayout::NCHW => {
                // [batch, channels, height, width] -> [height, width, channels]
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let idx = c * height * width + y * width + x;
                            let value = data.get(idx).copied().unwrap_or(0.0);
                            pixels.push(self.normalize_value(value));
                        }
                    }
                }
            }
            TensorLayout::NHWC => {
                // Already in [batch, height, width, channels]
                for &v in data.iter().take(width * height * channels) {
                    pixels.push(self.normalize_value(v));
                }
            }
        }

        Ok(ProcessedOutput::Image(ImageOutput {
            data: pixels,
            width: width as u32,
            height: height as u32,
            channels: channels as u8,
            format: self.output_format.clone(),
        }))
    }

    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<()> {
        if let ProcessedOutput::Image(img) = output {
            match img.channels {
                1 => {
                    let image = image::GrayImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| CompilerError::InvalidModel("Invalid image dimensions".into()))?;
                    image.save(path)?;
                }
                3 => {
                    let image = image::RgbImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| CompilerError::InvalidModel("Invalid image dimensions".into()))?;
                    image.save(path)?;
                }
                4 => {
                    let image = image::RgbaImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| CompilerError::InvalidModel("Invalid image dimensions".into()))?;
                    image.save(path)?;
                }
                _ => return Err(CompilerError::InvalidModel(
                    format!("Unsupported channel count: {}", img.channels)
                ).into()),
            }
        }
        Ok(())
    }

    fn validate_config(&self) -> Result<()> {
        Ok(())
    }
}
```

### Phase 3: Config Integration

**Extend `crates/compiler/src/execution/config.rs`**

```rust
/// Output handler configuration from TOML
#[derive(Debug, Clone, Deserialize)]
pub struct OutputHandlerConfig {
    /// Handler type: "image", "audio", "text", "tensor"
    #[serde(rename = "type")]
    pub handler_type: String,

    /// All other config options passed to handler
    #[serde(flatten)]
    pub options: toml::Value,
}

/// Extended execution config with output handlers
#[derive(Debug, Clone, Deserialize)]
pub struct ExecutionConfig {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub stages: Vec<StageConfig>,

    /// NEW: Output handler configurations keyed by output name
    #[serde(default)]
    pub output_handlers: HashMap<String, OutputHandlerConfig>,
}

impl ExecutionConfig {
    /// Create output handlers from config
    pub fn create_output_handlers(&self) -> Result<HashMap<String, Box<dyn OutputHandler>>> {
        let registry = OutputHandlerRegistry::new();
        let mut handlers = HashMap::new();

        for (name, config) in &self.output_handlers {
            let handler = registry.create(&config.handler_type, &config.options)?;
            handlers.insert(name.clone(), handler);
        }

        Ok(handlers)
    }
}
```

### Phase 4: Pipeline Executor Integration

**Extend `crates/compiler/src/execution/mod.rs`**

```rust
impl PipelineExecutor {
    /// Execute pipeline and process outputs with handlers
    pub fn execute_with_handlers(
        &self,
        inputs: HashMap<String, Value>,
        bridge: &mut StreamingPipelineBridge,
    ) -> Result<HashMap<String, ProcessedOutput>> {
        // Execute pipeline
        let raw_outputs = self.execute_streaming(inputs, bridge)?;

        // Create output handlers
        let handlers = self.config.create_output_handlers()?;

        // Convert Value outputs to Arc<Vec<f32>> for handlers
        let tensor_outputs: HashMap<String, Arc<Vec<f32>>> = raw_outputs.into_iter()
            .filter_map(|(name, value)| {
                if let Value::Tensor(data) = value {
                    Some((name, data))
                } else {
                    None
                }
            })
            .collect();

        // Process each output with its handler
        let mut processed = HashMap::new();
        for (name, handler) in &handlers {
            let output = handler.process(&tensor_outputs)?;
            processed.insert(name.clone(), output);
        }

        Ok(processed)
    }

    /// Execute and save outputs to files
    pub fn execute_and_save(
        &self,
        inputs: HashMap<String, Value>,
        bridge: &mut StreamingPipelineBridge,
        output_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        let handlers = self.config.create_output_handlers()?;
        let outputs = self.execute_with_handlers(inputs, bridge)?;

        let mut saved_files = Vec::new();
        for (name, output) in &outputs {
            if let Some(handler) = handlers.get(name) {
                let extension = match output {
                    ProcessedOutput::Image(img) => match img.format {
                        ImageFormat::Png => "png",
                        ImageFormat::Jpeg { .. } => "jpg",
                        ImageFormat::WebP { .. } => "webp",
                    },
                    ProcessedOutput::Audio(_) => "wav",
                    ProcessedOutput::Text(_) => "txt",
                    ProcessedOutput::Tensor(_) => "npy",
                    ProcessedOutput::Json(_) => "json",
                };

                let path = output_dir.join(format!("{}.{}", name, extension));
                handler.save(output, &path)?;
                saved_files.push(path);
            }
        }

        Ok(saved_files)
    }
}
```

## Complete Config Example: SD-Turbo

```toml
# SD-Turbo Pipeline Configuration with Output Handlers

[pipeline]
name = "stable-diffusion-turbo"
description = "SD-Turbo - Fast text-to-image"
version = "1.0"

[compiler]
output = "./sd-turbo.holo"
memory_budget = 2000
verbose = false
parallel = false

[[stages]]
name = "text_encoder"
path = "public/models/onnx/sd-turbo/text_encoder/model.onnx"
iterations = 1

[[stages]]
name = "unet"
path = "public/models/onnx/sd-turbo/unet/model.onnx"
iterations = 4

[[stages]]
name = "vae_decoder"
path = "public/models/onnx/sd-turbo/vae_decoder/model.onnx"
iterations = 1

# ===== Execution Recipe =====
[pipeline.execution]
inputs = ["prompt", "seed", "height", "width"]
outputs = ["image"]

# ===== Output Handler Configuration =====
[pipeline.execution.output_handlers.image]
type = "image"
output = "vae_output"           # Which tensor to process
format = "rgb"                  # rgb, rgba, grayscale
layout = "NCHW"                 # NCHW (PyTorch) or NHWC (TensorFlow)
value_range = "neg_one_one"     # Input tensor range
output_format = "png"           # png, jpg, webp
# Optional: explicit dimensions (auto-inferred if not specified)
# width = 512
# height = 512

# ===== Execution Stages =====
[[pipeline.execution.stages]]
name = "tokenize"
type = "builtin.tokenize"
inputs = { text = "prompt" }
outputs = ["input_ids"]
max_length = 77
padding = true

[[pipeline.execution.stages]]
name = "text_encoder"
model = "text_encoder"
inputs = { input_ids = "input_ids" }
outputs = ["embeddings"]

[[pipeline.execution.stages]]
name = "noise_init"
type = "builtin.randn"
outputs = ["latents"]
shape = [1, 4, "height/8", "width/8"]
seed = "seed"

[[pipeline.execution.stages]]
name = "timesteps_init"
type = "builtin.timesteps"
outputs = ["timesteps"]
num_inference_steps = 4
num_train_timesteps = 1000

[[pipeline.execution.stages]]
name = "diffusion_loop"

[pipeline.execution.stages.loop]
count = 4
var = "step"

[[pipeline.execution.stages.stages]]
name = "unet"
model = "unet"
outputs = ["noise_pred"]

[pipeline.execution.stages.stages.inputs]
sample = "latents"
timestep = "timesteps[step]"
encoder_hidden_states = "embeddings"

[[pipeline.execution.stages.stages]]
name = "scheduler_step"
type = "builtin.euler_discrete"
inputs = { latents = "latents", noise_pred = "noise_pred" }
outputs = ["latents"]
num_inference_steps = 4

[[pipeline.execution.stages]]
name = "vae_decode"
model = "vae_decoder"
inputs = { latent_sample = "latents * 0.18215" }
outputs = ["vae_output"]

# No need for builtin.image_decode - the output handler does it!
```

## Config Example: Audio Model (Whisper)

```toml
[pipeline]
name = "whisper-tiny"
description = "Whisper speech recognition"

[pipeline.execution]
inputs = ["audio_path"]
outputs = ["transcription"]

# Audio input handler
[pipeline.execution.input_handlers.audio]
type = "audio"
sample_rate = 16000
mono = true
normalize = true

# Text output handler
[pipeline.execution.output_handlers.transcription]
type = "text"
tokenizer = "whisper-tiny"
decoding = "greedy"

[[pipeline.execution.stages]]
name = "mel_spectrogram"
type = "builtin.mel_spectrogram"
inputs = { audio = "audio" }
outputs = ["mel"]
n_mels = 80
n_fft = 400
hop_length = 160

[[pipeline.execution.stages]]
name = "encoder"
model = "encoder"
inputs = { mel = "mel" }
outputs = ["encoder_hidden_states"]

[[pipeline.execution.stages]]
name = "decoder"
model = "decoder"
inputs = { encoder_hidden_states = "encoder_hidden_states" }
outputs = ["logits"]
```

## Config Example: LLM (Text Generation)

```toml
[pipeline]
name = "phi-2"
description = "Microsoft Phi-2 text generation"

[pipeline.execution]
inputs = ["prompt", "max_tokens", "temperature"]
outputs = ["generated_text"]

[pipeline.execution.output_handlers.generated_text]
type = "text"
tokenizer = "microsoft/phi-2"
decoding = "top_p"
top_p = 0.95
temperature = "temperature"    # Reference to input variable

[[pipeline.execution.stages]]
name = "tokenize"
type = "builtin.tokenize"
inputs = { text = "prompt" }
outputs = ["input_ids", "attention_mask"]
tokenizer = "microsoft/phi-2"

[[pipeline.execution.stages]]
name = "generate"

[pipeline.execution.stages.loop]
count = "max_tokens"
var = "step"
break_on = "eos_token"         # Stop when EOS generated

[[pipeline.execution.stages.stages]]
name = "forward"
model = "phi2"
inputs = { input_ids = "input_ids", attention_mask = "attention_mask" }
outputs = ["logits"]

[[pipeline.execution.stages.stages]]
name = "sample"
type = "builtin.sample_token"
inputs = { logits = "logits", temperature = "temperature" }
outputs = ["next_token"]
top_p = 0.95

[[pipeline.execution.stages.stages]]
name = "append"
type = "builtin.append_token"
inputs = { input_ids = "input_ids", token = "next_token" }
outputs = ["input_ids"]
```

## Built-in Operations for Common Model Types

### Image Models
| Operation | Description |
|-----------|-------------|
| `builtin.randn` | Generate random noise tensor |
| `builtin.timesteps` | Generate diffusion timesteps |
| `builtin.euler_discrete` | Euler discrete scheduler step |
| `builtin.ddpm` | DDPM scheduler step |
| `builtin.resize` | Resize tensor (bilinear/nearest) |
| `builtin.normalize` | Normalize tensor values |

### Audio Models
| Operation | Description |
|-----------|-------------|
| `builtin.load_audio` | Load and resample audio file |
| `builtin.mel_spectrogram` | Compute mel spectrogram |
| `builtin.stft` | Short-time Fourier transform |
| `builtin.istft` | Inverse STFT |

### Text/LLM Models
| Operation | Description |
|-----------|-------------|
| `builtin.tokenize` | Tokenize text input |
| `builtin.decode_tokens` | Decode token IDs to text |
| `builtin.sample_token` | Sample next token from logits |
| `builtin.append_token` | Append token to sequence |
| `builtin.kv_cache` | Key-value cache management |

## Implementation Priority

| Phase | Component | Effort | Status |
|-------|-----------|--------|--------|
| 1 | OutputHandler trait + registry | Low | 🔲 |
| 2 | ImageHandler implementation | Medium | 🔲 |
| 3 | Config parsing for handlers | Low | 🔲 |
| 4 | PipelineExecutor integration | Medium | 🔲 |
| 5 | AudioHandler implementation | Medium | 🔲 |
| 6 | TextHandler implementation | High | 🔲 |
| 7 | Additional builtins | Medium | 🔲 |

## Benefits

1. **Declarative** - Model authors specify output format in config, not code
2. **Extensible** - Custom handlers can be registered at runtime
3. **Zero-overhead** - Handlers only instantiated when configured
4. **Type-safe** - Each handler validates its configuration
5. **Composable** - Multiple outputs can have different handlers
6. **Portable** - Same config works across all deployment targets

## Files to Create/Modify

| File | Action |
|------|--------|
| `crates/compiler/src/execution/output_handlers/mod.rs` | Create - Handler trait + registry |
| `crates/compiler/src/execution/output_handlers/image.rs` | Create - Image handler |
| `crates/compiler/src/execution/output_handlers/audio.rs` | Create - Audio handler |
| `crates/compiler/src/execution/output_handlers/text.rs` | Create - Text handler |
| `crates/compiler/src/execution/config.rs` | Modify - Add output_handlers field |
| `crates/compiler/src/execution/mod.rs` | Modify - Add execute_with_handlers |
| `crates/compiler/src/lib.rs` | Modify - Export new types |
| `src/main.rs` | Modify - Use handlers for output saving |

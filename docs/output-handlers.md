# Output Handlers Design

This document describes the recommended approach for handling different output types (images, audio, text) from compiled models.

## Design Philosophy

We recommend a **hybrid approach**:

1. **Trait-based API** for extensibility - model authors can implement custom handlers
2. **Feature-gated built-in handlers** - optional convenience helpers for common formats
3. **Zero-overhead** when handlers aren't used - no dependencies added unless opted-in

## Core Trait Definition

```rust
/// Trait for converting raw model outputs to typed results
pub trait OutputHandler {
    /// The output type produced by this handler
    type Output;

    /// Convert raw float outputs to the final type
    fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output>;

    /// Expected output names this handler processes
    fn expected_outputs(&self) -> &[&str];
}
```

### Example: Custom Image Handler

```rust
use hologram_onnx_compiler::{OutputHandler, Result};
use std::collections::HashMap;

pub struct RgbImageHandler {
    pub width: usize,
    pub height: usize,
}

impl OutputHandler for RgbImageHandler {
    type Output = RgbImage;

    fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output> {
        let data = outputs.get("sample")
            .ok_or_else(|| anyhow::anyhow!("Missing 'sample' output"))?;

        // Denormalize from [-1, 1] to [0, 255]
        let pixels: Vec<u8> = data.iter()
            .map(|&v| ((v.clamp(-1.0, 1.0) + 1.0) * 127.5) as u8)
            .collect();

        Ok(RgbImage::from_raw(self.width as u32, self.height as u32, pixels)
            .expect("Invalid image dimensions"))
    }

    fn expected_outputs(&self) -> &[&str] {
        &["sample"]
    }
}
```

## Built-in Handlers (Feature-Gated)

### Feature Flags

```toml
[features]
default = []
image-output = ["image"]
audio-output = ["hound"]
text-output = ["tokenizers"]
all-outputs = ["image-output", "audio-output", "text-output"]
```

### Image Handler (`image-output` feature)

```rust
#[cfg(feature = "image-output")]
pub mod image_output {
    use image::{DynamicImage, RgbImage, RgbaImage, GrayImage};

    /// Image output configuration
    pub struct ImageOutput {
        /// Output tensor name (default: "sample")
        pub output_name: String,
        /// Image dimensions [C, H, W] or [H, W, C]
        pub layout: ImageLayout,
        /// Value range in the tensor
        pub value_range: ValueRange,
        /// Output pixel format
        pub format: PixelFormat,
    }

    pub enum ImageLayout {
        /// Channel-first: [batch, channels, height, width]
        NCHW,
        /// Channel-last: [batch, height, width, channels]
        NHWC,
    }

    pub enum ValueRange {
        /// Values in [0, 1]
        ZeroOne,
        /// Values in [-1, 1]
        NegOneOne,
        /// Values in [0, 255]
        ByteRange,
    }

    pub enum PixelFormat {
        Rgb,
        Rgba,
        Grayscale,
    }

    impl OutputHandler for ImageOutput {
        type Output = DynamicImage;

        fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output> {
            let data = outputs.get(&self.output_name)
                .ok_or_else(|| anyhow::anyhow!("Missing output: {}", self.output_name))?;

            // Normalize values to [0, 255] based on value_range
            let normalized = match self.value_range {
                ValueRange::ZeroOne => data.iter().map(|&v| (v * 255.0) as u8).collect(),
                ValueRange::NegOneOne => data.iter()
                    .map(|&v| ((v + 1.0) * 127.5) as u8).collect(),
                ValueRange::ByteRange => data.iter().map(|&v| v as u8).collect(),
            };

            // Convert layout and create image
            // ... implementation details
        }
    }

    impl Default for ImageOutput {
        fn default() -> Self {
            Self {
                output_name: "sample".to_string(),
                layout: ImageLayout::NCHW,
                value_range: ValueRange::NegOneOne,
                format: PixelFormat::Rgb,
            }
        }
    }
}
```

### Audio Handler (`audio-output` feature)

```rust
#[cfg(feature = "audio-output")]
pub mod audio_output {
    use hound::{WavSpec, WavWriter};

    /// Audio output configuration
    pub struct AudioOutput {
        /// Output tensor name
        pub output_name: String,
        /// Sample rate in Hz
        pub sample_rate: u32,
        /// Number of channels (1 = mono, 2 = stereo)
        pub channels: u16,
        /// Value range in tensor
        pub value_range: AudioValueRange,
    }

    pub enum AudioValueRange {
        /// Float values in [-1.0, 1.0]
        Normalized,
        /// Integer values in [-32768, 32767]
        Int16,
    }

    impl OutputHandler for AudioOutput {
        type Output = AudioBuffer;

        fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output> {
            let data = outputs.get(&self.output_name)
                .ok_or_else(|| anyhow::anyhow!("Missing output: {}", self.output_name))?;

            Ok(AudioBuffer {
                samples: data.clone(),
                sample_rate: self.sample_rate,
                channels: self.channels,
            })
        }
    }

    pub struct AudioBuffer {
        pub samples: Vec<f32>,
        pub sample_rate: u32,
        pub channels: u16,
    }

    impl AudioBuffer {
        /// Save as WAV file
        pub fn save_wav(&self, path: &Path) -> Result<()> {
            let spec = WavSpec {
                channels: self.channels,
                sample_rate: self.sample_rate,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut writer = WavWriter::create(path, spec)?;
            for &sample in &self.samples {
                let int_sample = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer.write_sample(int_sample)?;
            }
            writer.finalize()?;
            Ok(())
        }
    }
}
```

### Text Handler (`text-output` feature)

```rust
#[cfg(feature = "text-output")]
pub mod text_output {
    use tokenizers::Tokenizer;

    /// Text output configuration for language models
    pub struct TextOutput {
        /// Output tensor name (typically "logits")
        pub output_name: String,
        /// Tokenizer for decoding
        tokenizer: Tokenizer,
        /// Decoding strategy
        pub strategy: DecodingStrategy,
    }

    pub enum DecodingStrategy {
        /// Take the highest probability token
        Greedy,
        /// Sample from top-k most likely tokens
        TopK(usize),
        /// Sample from tokens with cumulative probability p
        TopP(f32),
    }

    impl OutputHandler for TextOutput {
        type Output = String;

        fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output> {
            let logits = outputs.get(&self.output_name)
                .ok_or_else(|| anyhow::anyhow!("Missing output: {}", self.output_name))?;

            // Decode based on strategy
            let token_ids = match self.strategy {
                DecodingStrategy::Greedy => self.greedy_decode(logits),
                DecodingStrategy::TopK(k) => self.top_k_decode(logits, k),
                DecodingStrategy::TopP(p) => self.top_p_decode(logits, p),
            };

            self.tokenizer.decode(&token_ids, true)
                .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
        }
    }
}
```

## Usage Examples

### With Built-in Handlers

```rust
use hologram_onnx_compiler::{Model, ModelBuilder};

#[cfg(feature = "image-output")]
use hologram_onnx_compiler::image_output::{ImageOutput, ImageLayout, ValueRange};

fn generate_image() -> Result<()> {
    let model = ModelBuilder::new()
        .load_holo("sd-pipeline.holo")?
        .build()?;

    let outputs = model.run(inputs)?;

    #[cfg(feature = "image-output")]
    {
        let handler = ImageOutput {
            output_name: "sample".to_string(),
            layout: ImageLayout::NCHW,
            value_range: ValueRange::NegOneOne,
            format: PixelFormat::Rgb,
        };
        let image = handler.convert(outputs)?;
        image.save("output.png")?;
    }

    Ok(())
}
```

### With Custom Handler

```rust
use hologram_onnx_compiler::{Model, ModelBuilder, OutputHandler};

// Custom handler for a specific model's output format
struct MyModelOutput {
    threshold: f32,
}

impl OutputHandler for MyModelOutput {
    type Output = Vec<Detection>;

    fn convert(&self, outputs: HashMap<String, Vec<f32>>) -> Result<Self::Output> {
        let boxes = outputs.get("boxes").ok_or(anyhow!("Missing boxes"))?;
        let scores = outputs.get("scores").ok_or(anyhow!("Missing scores"))?;

        // Custom detection processing
        let detections = boxes.chunks(4)
            .zip(scores.iter())
            .filter(|(_, &score)| score > self.threshold)
            .map(|(box_coords, &score)| Detection {
                x1: box_coords[0],
                y1: box_coords[1],
                x2: box_coords[2],
                y2: box_coords[3],
                score,
            })
            .collect();

        Ok(detections)
    }

    fn expected_outputs(&self) -> &[&str] {
        &["boxes", "scores"]
    }
}
```

## Composable Handlers

Handlers can be composed for multi-output models:

```rust
/// Compose multiple handlers for models with multiple output types
pub struct ComposedOutputHandler<A, B> {
    handler_a: A,
    handler_b: B,
}

impl<A, B> ComposedOutputHandler<A, B>
where
    A: OutputHandler,
    B: OutputHandler,
{
    pub fn convert_both(
        &self,
        outputs: HashMap<String, Vec<f32>>
    ) -> Result<(A::Output, B::Output)> {
        let a = self.handler_a.convert(outputs.clone())?;
        let b = self.handler_b.convert(outputs)?;
        Ok((a, b))
    }
}

// Example: Model that outputs both image and confidence scores
let handler = ComposedOutputHandler {
    handler_a: ImageOutput::default(),
    handler_b: ConfidenceHandler { output_name: "confidence".to_string() },
};
let (image, confidence) = handler.convert_both(outputs)?;
```

## Implementation Priority

| Phase | Component | Effort |
|-------|-----------|--------|
| 1 | `OutputHandler` trait definition | Low |
| 2 | Image output handler (`image-output`) | Medium |
| 3 | Audio output handler (`audio-output`) | Medium |
| 4 | Text/LLM output handler (`text-output`) | High |
| 5 | Composed handlers | Low |

## Benefits of This Approach

1. **Extensibility**: Model authors can implement custom handlers for any output format
2. **Zero-cost**: Built-in handlers are feature-gated, no bloat in final binary
3. **Type safety**: Each handler specifies its output type at compile time
4. **Composability**: Handlers can be combined for multi-output models
5. **Separation of concerns**: Model execution is separate from output processing

## Related Files

- [api.rs](../crates/compiler/src/api.rs) - Core public API where handlers integrate
- [MEMORY_OPTIMIZATION.md](./MEMORY_OPTIMIZATION.md) - Memory optimization strategies

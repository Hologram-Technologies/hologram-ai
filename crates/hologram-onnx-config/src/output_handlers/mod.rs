//! Output handler trait and registry for multi-modal outputs.
//!
//! # Performance
//!
//! - **Zero-copy**: Tensor data referenced directly where possible
//! - **Lazy loading**: Handlers only created when features enabled
//! - **O(1) dispatch**: Handler registry uses HashMap lookup

use crate::config::OutputHandlerConfig;
use crate::error::ConfigError;
use ahash::AHashMap;
use std::collections::HashMap;
use std::path::Path;
#[allow(unused_imports)]
use tracing::{debug, trace};

// Feature-gated handler modules
#[cfg(feature = "image-output")]
pub mod image;

#[cfg(feature = "audio-output")]
pub mod audio;

#[cfg(feature = "text-output")]
pub mod text;

/// Output handler trait for processing model outputs.
///
/// Implementors process raw tensor data into domain-specific formats
/// (images, audio, text, etc.).
pub trait OutputHandler: Send + Sync {
    /// Handler type name (e.g., "image", "audio", "text").
    fn handler_type(&self) -> &'static str;

    /// Process raw tensor outputs into domain-specific format.
    ///
    /// # Performance: O(n) where n = tensor size
    ///
    /// Implementations should use SIMD and zero-copy where possible.
    fn process(&self, outputs: &HashMap<String, TensorData>) -> Result<ProcessedOutput, ConfigError>;

    /// Save processed output to file.
    ///
    /// # Performance: O(n) where n = output size
    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<(), ConfigError>;
}

/// Raw tensor data from model execution.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// Tensor values (row-major order)
    pub data: Vec<f32>,

    /// Tensor shape
    pub shape: Vec<usize>,
}

impl TensorData {
    /// Create new tensor data.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Get total element count.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Processed output from handler.
#[derive(Debug, Clone)]
pub enum ProcessedOutput {
    /// Image output (RGB/RGBA/Grayscale)
    Image(ImageOutput),

    /// Audio output (WAV)
    Audio(AudioOutput),

    /// Text output (string)
    Text(String),

    /// Raw tensor output
    Tensor(TensorOutput),
}

/// Image output data.
#[derive(Debug, Clone)]
pub struct ImageOutput {
    /// Image data (row-major, interleaved channels)
    pub data: Vec<u8>,

    /// Image width
    pub width: u32,

    /// Image height
    pub height: u32,

    /// Number of channels (1=grayscale, 3=RGB, 4=RGBA)
    pub channels: u8,
}

impl ImageOutput {
    /// Create new image output.
    pub fn new(data: Vec<u8>, width: u32, height: u32, channels: u8) -> Self {
        assert!(channels == 1 || channels == 3 || channels == 4,
                "Invalid channel count: {}", channels);
        assert_eq!(data.len(), (width * height * channels as u32) as usize,
                   "Data size mismatch");
        Self { data, width, height, channels }
    }
}

/// Audio output data.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    /// Audio samples
    pub samples: Vec<f32>,

    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of channels (1=mono, 2=stereo)
    pub channels: u16,
}

impl AudioOutput {
    /// Create new audio output.
    pub fn new(samples: Vec<f32>, sample_rate: u32, channels: u16) -> Self {
        assert!(channels > 0, "Invalid channel count: {}", channels);
        Self { samples, sample_rate, channels }
    }
}

/// Tensor output data.
#[derive(Debug, Clone)]
pub struct TensorOutput {
    /// Tensor data
    pub data: Vec<f32>,

    /// Tensor shape
    pub shape: Vec<usize>,
}

impl TensorOutput {
    /// Create new tensor output.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

/// Factory function type for creating output handlers.
pub type HandlerFactory = Box<dyn Fn(&OutputHandlerConfig) -> Result<Box<dyn OutputHandler>, ConfigError> + Send + Sync>;

/// Registry for output handler factories.
///
/// # Performance
///
/// - **O(1) lookup**: Uses AHashMap for handler type → factory
/// - **Lazy creation**: Handlers only instantiated when needed
pub struct OutputHandlerRegistry {
    factories: AHashMap<String, HandlerFactory>,
}

impl OutputHandlerRegistry {
    /// Create new registry with all available handlers.
    ///
    /// Handlers are registered based on enabled features.
    pub fn new() -> Self {
        let registry = Self {
            factories: AHashMap::new(),
        };

        // Register feature-gated handlers
        #[cfg(feature = "image-output")]
        {
            debug!("Registering image output handler");
            registry.register("image", Box::new(|config| {
                image::ImageHandler::from_config(config)
                    .map(|h| Box::new(h) as Box<dyn OutputHandler>)
            }));
        }

        #[cfg(feature = "audio-output")]
        {
            debug!("Registering audio output handler");
            registry.register("audio", Box::new(|config| {
                audio::AudioHandler::from_config(config)
                    .map(|h| Box::new(h) as Box<dyn OutputHandler>)
            }));
        }

        #[cfg(feature = "text-output")]
        {
            debug!("Registering text output handler");
            registry.register("text", Box::new(|config| {
                text::TextHandler::from_config(config)
                    .map(|h| Box::new(h) as Box<dyn OutputHandler>)
            }));
        }

        registry
    }

    /// Register a handler factory.
    ///
    /// # Performance: O(1)
    pub fn register(&mut self, handler_type: impl Into<String>, factory: HandlerFactory) {
        let handler_type = handler_type.into();
        trace!("Registering handler: {}", handler_type);
        self.factories.insert(handler_type, factory);
    }

    /// Create handler from config.
    ///
    /// # Performance: O(1) lookup + handler creation cost
    pub fn create_handler(
        &self,
        config: &OutputHandlerConfig,
    ) -> Result<Box<dyn OutputHandler>, ConfigError> {
        let factory = self.factories.get(&config.handler_type)
            .ok_or_else(|| {
                // Check if feature is disabled
                let feature_map = [
                    ("image", "image-output"),
                    ("audio", "audio-output"),
                    ("text", "text-output"),
                ];

                for (handler_type, feature) in &feature_map {
                    if &config.handler_type == handler_type {
                        return ConfigError::feature_not_enabled(
                            *handler_type,
                            *feature,
                        );
                    }
                }

                ConfigError::unknown_handler_type(&config.handler_type)
            })?;

        trace!("Creating handler: {}", config.handler_type);
        factory(config)
    }

    /// Create multiple handlers from config map.
    ///
    /// # Performance: O(n) where n = number of handlers
    pub fn create_handlers(
        &self,
        configs: &HashMap<String, OutputHandlerConfig>,
    ) -> Result<HashMap<String, Box<dyn OutputHandler>>, ConfigError> {
        let mut handlers = HashMap::new();

        for (name, config) in configs {
            let handler = self.create_handler(config)?;
            handlers.insert(name.clone(), handler);
        }

        Ok(handlers)
    }
}

impl Default for OutputHandlerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = TensorData::new(data.clone(), shape.clone());

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
        assert_eq!(tensor.len(), 4);
        assert_eq!(tensor.ndim(), 2);
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_data_empty() {
        let tensor = TensorData::new(vec![], vec![0]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
    }

    #[test]
    fn test_image_output_creation() {
        let data = vec![255u8; 512 * 512 * 3];
        let img = ImageOutput::new(data.clone(), 512, 512, 3);

        assert_eq!(img.width, 512);
        assert_eq!(img.height, 512);
        assert_eq!(img.channels, 3);
        assert_eq!(img.data.len(), 512 * 512 * 3);
    }

    #[test]
    #[should_panic(expected = "Invalid channel count")]
    fn test_image_output_invalid_channels() {
        let data = vec![255u8; 100];
        ImageOutput::new(data, 10, 10, 5);
    }

    #[test]
    #[should_panic(expected = "Data size mismatch")]
    fn test_image_output_size_mismatch() {
        let data = vec![255u8; 100];
        ImageOutput::new(data, 10, 10, 3);
    }

    #[test]
    fn test_audio_output_creation() {
        let samples = vec![0.0f32; 44100];
        let audio = AudioOutput::new(samples.clone(), 44100, 2);

        assert_eq!(audio.sample_rate, 44100);
        assert_eq!(audio.channels, 2);
        assert_eq!(audio.samples.len(), 44100);
    }

    #[test]
    #[should_panic(expected = "Invalid channel count")]
    fn test_audio_output_invalid_channels() {
        let samples = vec![0.0f32; 100];
        AudioOutput::new(samples, 44100, 0);
    }

    #[test]
    fn test_tensor_output_creation() {
        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![1, 3];
        let tensor = TensorOutput::new(data.clone(), shape.clone());

        assert_eq!(tensor.data, data);
        assert_eq!(tensor.shape, shape);
    }

    #[test]
    fn test_registry_creation() {
        let registry = OutputHandlerRegistry::new();
        // Registry should be created successfully
        assert!(true);
    }

    #[test]
    fn test_unknown_handler_type() {
        let registry = OutputHandlerRegistry::new();
        let config = OutputHandlerConfig {
            handler_type: "unknown".to_string(),
            output: "tensor".to_string(),
            config: HashMap::new(),
        };

        let result = registry.create_handler(&config);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("unknown"));
    }

    #[test]
    fn test_feature_not_enabled_error() {
        let registry = OutputHandlerRegistry::new();

        // Test for handlers that require features
        #[cfg(not(feature = "image-output"))]
        {
            let config = OutputHandlerConfig {
                handler_type: "image".to_string(),
                output: "tensor".to_string(),
                config: HashMap::new(),
            };

            let result = registry.create_handler(&config);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.to_string().contains("image-output"));
        }

        #[cfg(not(feature = "audio-output"))]
        {
            let config = OutputHandlerConfig {
                handler_type: "audio".to_string(),
                output: "tensor".to_string(),
                config: HashMap::new(),
            };

            let result = registry.create_handler(&config);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.to_string().contains("audio-output"));
        }

        #[cfg(not(feature = "text-output"))]
        {
            let config = OutputHandlerConfig {
                handler_type: "text".to_string(),
                output: "tensor".to_string(),
                config: HashMap::new(),
            };

            let result = registry.create_handler(&config);
            assert!(result.is_err());
            let err = result.err().unwrap();
            assert!(err.to_string().contains("text-output"));
        }
    }
}

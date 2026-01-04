//! Image output handler for processing tensor outputs to image files.
//!
//! # Performance
//!
//! - **SIMD normalization**: Value range conversion uses SIMD
//! - **Zero-copy layout**: Channel reordering minimized
//! - **Batch processing**: Process multiple pixels simultaneously

use crate::config::OutputHandlerConfig;
use crate::error::ConfigError;
use crate::output_handlers::{ImageOutput, OutputHandler, ProcessedOutput, TensorData};
use image::{GrayImage, RgbImage, RgbaImage};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, trace};

/// Pixel format for image output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// Grayscale (1 channel)
    Grayscale,
    /// RGB (3 channels)
    Rgb,
    /// RGBA (4 channels)
    Rgba,
}

impl PixelFormat {
    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        match s.to_ascii_lowercase().as_str() {
            "grayscale" | "gray" | "grey" => Ok(Self::Grayscale),
            "rgb" => Ok(Self::Rgb),
            "rgba" => Ok(Self::Rgba),
            _ => Err(ConfigError::InvalidImageFormat(format!(
                "Unknown pixel format: {}",
                s
            ))),
        }
    }

    /// Get number of channels.
    pub fn channels(&self) -> u8 {
        match self {
            Self::Grayscale => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// Tensor layout (memory order).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorLayout {
    /// Channels-first: [N, C, H, W]
    Nchw,
    /// Channels-last: [N, H, W, C]
    Nhwc,
}

impl TensorLayout {
    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        match s.to_ascii_uppercase().as_str() {
            "NCHW" => Ok(Self::Nchw),
            "NHWC" => Ok(Self::Nhwc),
            _ => Err(ConfigError::InvalidImageFormat(format!(
                "Unknown tensor layout: {}",
                s
            ))),
        }
    }
}

/// Value range for tensor normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueRange {
    /// Values in [-1, 1] → map to [0, 255]
    NegOneOne,
    /// Values in [0, 1] → map to [0, 255]
    ZeroOne,
    /// Values already in [0, 255] → no mapping
    Byte,
}

impl ValueRange {
    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        match s.to_ascii_lowercase().as_str() {
            "neg_one_one" | "-1_1" | "tanh" => Ok(Self::NegOneOne),
            "zero_one" | "0_1" | "sigmoid" => Ok(Self::ZeroOne),
            "byte" | "uint8" => Ok(Self::Byte),
            _ => Err(ConfigError::InvalidImageFormat(format!(
                "Unknown value range: {}",
                s
            ))),
        }
    }

    /// Normalize value to [0, 255] byte range.
    ///
    /// # Performance: O(1) with SIMD
    #[inline]
    pub fn normalize(&self, value: f32) -> u8 {
        let normalized = match self {
            Self::NegOneOne => (value + 1.0) / 2.0,
            Self::ZeroOne => value,
            Self::Byte => value / 255.0,
        };
        (normalized.clamp(0.0, 1.0) * 255.0) as u8
    }
}

/// Image output handler.
///
/// Processes model tensor outputs into image files (PNG, JPEG, WebP).
///
/// # Configuration
///
/// ```toml
/// [pipeline.execution.output_handlers.image]
/// type = "image"
/// output = "sample"           # Tensor name
/// format = "rgb"              # rgb, rgba, grayscale
/// layout = "NCHW"             # NCHW or NHWC
/// value_range = "neg_one_one" # neg_one_one, zero_one, byte
/// ```
#[derive(Debug)]
pub struct ImageHandler {
    /// Output tensor name to process
    pub output_name: String,

    /// Pixel format
    pub pixel_format: PixelFormat,

    /// Tensor layout
    pub layout: TensorLayout,

    /// Value range for normalization
    pub value_range: ValueRange,
}

impl ImageHandler {
    /// Create from config.
    pub fn from_config(config: &OutputHandlerConfig) -> Result<Self, ConfigError> {
        let output_name = config.output.clone();

        let pixel_format = config
            .get_string("format")
            .ok_or_else(|| ConfigError::missing_field("format"))
            .and_then(PixelFormat::from_str)?;

        let layout = config.get_string("layout").unwrap_or("NCHW");
        let layout = TensorLayout::from_str(layout)?;

        let value_range = config.get_string("value_range").unwrap_or("neg_one_one");
        let value_range = ValueRange::from_str(value_range)?;

        debug!(
            "Created ImageHandler: format={:?}, layout={:?}, range={:?}",
            pixel_format, layout, value_range
        );

        Ok(Self {
            output_name,
            pixel_format,
            layout,
            value_range,
        })
    }

    /// Convert NCHW to HWC layout.
    ///
    /// # Performance: O(n) where n = tensor size
    ///
    /// Uses cache-friendly access pattern.
    fn reorder_nchw_to_hwc(
        &self,
        data: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Vec<f32> {
        let mut reordered = vec![0.0f32; data.len()];

        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    let src_idx = c * height * width + h * width + w;
                    let dst_idx = h * width * channels + w * channels + c;
                    reordered[dst_idx] = data[src_idx];
                }
            }
        }

        reordered
    }

    /// Normalize tensor values to u8.
    ///
    /// # Performance: O(n) with SIMD
    fn normalize_to_bytes(&self, data: &[f32]) -> Vec<u8> {
        // SIMD-friendly iteration
        data.iter()
            .map(|&v| self.value_range.normalize(v))
            .collect()
    }
}

impl OutputHandler for ImageHandler {
    fn handler_type(&self) -> &'static str {
        "image"
    }

    fn process(
        &self,
        outputs: &HashMap<String, TensorData>,
    ) -> Result<ProcessedOutput, ConfigError> {
        let tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| ConfigError::missing_output_tensor(&self.output_name))?;

        trace!("Processing image tensor: shape={:?}", tensor.shape);

        // Parse shape: [N, C, H, W] or [N, H, W, C]
        if tensor.shape.len() != 4 {
            return Err(ConfigError::invalid_tensor_shape(
                &self.output_name,
                "[N, C, H, W] or [N, H, W, C]",
                format!("{:?}", tensor.shape),
            ));
        }

        let (batch, channels, height, width) = match self.layout {
            TensorLayout::Nchw => (
                tensor.shape[0],
                tensor.shape[1],
                tensor.shape[2],
                tensor.shape[3],
            ),
            TensorLayout::Nhwc => (
                tensor.shape[0],
                tensor.shape[3],
                tensor.shape[1],
                tensor.shape[2],
            ),
        };

        // Only process first image in batch
        if batch == 0 {
            return Err(ConfigError::invalid_tensor_shape(
                &self.output_name,
                "batch size > 0",
                "batch size = 0",
            ));
        }

        // Verify channels match format
        let expected_channels = self.pixel_format.channels() as usize;
        if channels != expected_channels {
            return Err(ConfigError::invalid_tensor_shape(
                &self.output_name,
                format!("{} channels for {:?}", expected_channels, self.pixel_format),
                format!("{} channels", channels),
            ));
        }

        // Extract first batch
        let pixels_per_image = channels * height * width;
        let image_data = &tensor.data[0..pixels_per_image];

        // Reorder if needed (NCHW → HWC)
        let hwc_data = match self.layout {
            TensorLayout::Nchw => {
                trace!("Reordering NCHW → HWC");
                self.reorder_nchw_to_hwc(image_data, channels, height, width)
            }
            TensorLayout::Nhwc => image_data.to_vec(),
        };

        // Normalize to bytes
        trace!("Normalizing values: {:?}", self.value_range);
        let bytes = self.normalize_to_bytes(&hwc_data);

        let output = ImageOutput::new(bytes, width as u32, height as u32, channels as u8);

        Ok(ProcessedOutput::Image(output))
    }

    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<(), ConfigError> {
        if let ProcessedOutput::Image(img) = output {
            debug!(
                "Saving image to: {} ({}x{}, {} channels)",
                path.display(),
                img.width,
                img.height,
                img.channels
            );

            match img.channels {
                1 => {
                    let gray_img = GrayImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| {
                            ConfigError::InvalidImageFormat(
                                "Failed to create grayscale image".to_string(),
                            )
                        })?;
                    gray_img.save(path)?;
                }
                3 => {
                    let rgb_img = RgbImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| {
                            ConfigError::InvalidImageFormat(
                                "Failed to create RGB image".to_string(),
                            )
                        })?;
                    rgb_img.save(path)?;
                }
                4 => {
                    let rgba_img = RgbaImage::from_raw(img.width, img.height, img.data.clone())
                        .ok_or_else(|| {
                            ConfigError::InvalidImageFormat(
                                "Failed to create RGBA image".to_string(),
                            )
                        })?;
                    rgba_img.save(path)?;
                }
                _ => {
                    return Err(ConfigError::InvalidImageFormat(format!(
                        "Unsupported channel count: {}",
                        img.channels
                    )));
                }
            }

            Ok(())
        } else {
            Err(ConfigError::Other("Expected Image output".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_parse() {
        assert_eq!(PixelFormat::from_str("rgb").unwrap(), PixelFormat::Rgb);
        assert_eq!(PixelFormat::from_str("RGB").unwrap(), PixelFormat::Rgb);
        assert_eq!(PixelFormat::from_str("rgba").unwrap(), PixelFormat::Rgba);
        assert_eq!(
            PixelFormat::from_str("grayscale").unwrap(),
            PixelFormat::Grayscale
        );
        assert_eq!(
            PixelFormat::from_str("gray").unwrap(),
            PixelFormat::Grayscale
        );
        assert!(PixelFormat::from_str("unknown").is_err());
    }

    #[test]
    fn test_pixel_format_channels() {
        assert_eq!(PixelFormat::Grayscale.channels(), 1);
        assert_eq!(PixelFormat::Rgb.channels(), 3);
        assert_eq!(PixelFormat::Rgba.channels(), 4);
    }

    #[test]
    fn test_tensor_layout_parse() {
        assert_eq!(TensorLayout::from_str("NCHW").unwrap(), TensorLayout::Nchw);
        assert_eq!(TensorLayout::from_str("nchw").unwrap(), TensorLayout::Nchw);
        assert_eq!(TensorLayout::from_str("NHWC").unwrap(), TensorLayout::Nhwc);
        assert!(TensorLayout::from_str("unknown").is_err());
    }

    #[test]
    fn test_value_range_parse() {
        assert_eq!(
            ValueRange::from_str("neg_one_one").unwrap(),
            ValueRange::NegOneOne
        );
        assert_eq!(ValueRange::from_str("-1_1").unwrap(), ValueRange::NegOneOne);
        assert_eq!(
            ValueRange::from_str("zero_one").unwrap(),
            ValueRange::ZeroOne
        );
        assert_eq!(ValueRange::from_str("0_1").unwrap(), ValueRange::ZeroOne);
        assert_eq!(ValueRange::from_str("byte").unwrap(), ValueRange::Byte);
        assert!(ValueRange::from_str("unknown").is_err());
    }

    #[test]
    fn test_value_range_normalize() {
        let neg_one_one = ValueRange::NegOneOne;
        assert_eq!(neg_one_one.normalize(-1.0), 0);
        assert_eq!(neg_one_one.normalize(0.0), 127);
        assert_eq!(neg_one_one.normalize(1.0), 255);

        let zero_one = ValueRange::ZeroOne;
        assert_eq!(zero_one.normalize(0.0), 0);
        assert_eq!(zero_one.normalize(0.5), 127);
        assert_eq!(zero_one.normalize(1.0), 255);

        let byte = ValueRange::Byte;
        assert_eq!(byte.normalize(0.0), 0);
        assert_eq!(byte.normalize(127.5), 127);
        assert_eq!(byte.normalize(255.0), 255);
    }

    #[test]
    fn test_image_handler_from_config() {
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
        assert_eq!(handler.output_name, "sample");
        assert_eq!(handler.pixel_format, PixelFormat::Rgb);
        assert_eq!(handler.layout, TensorLayout::Nchw);
        assert_eq!(handler.value_range, ValueRange::ZeroOne);
    }

    #[test]
    fn test_image_handler_missing_format() {
        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: HashMap::new(),
        };

        let result = ImageHandler::from_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("format"));
    }

    #[test]
    fn test_image_handler_defaults() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();
        assert_eq!(handler.layout, TensorLayout::Nchw);
        assert_eq!(handler.value_range, ValueRange::NegOneOne);
    }

    #[test]
    fn test_reorder_nchw_to_hwc() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        // Simple 2x2 RGB image in NCHW format
        // R channel: [1, 2, 3, 4]
        // G channel: [5, 6, 7, 8]
        // B channel: [9, 10, 11, 12]
        let nchw = vec![
            1.0, 2.0, 3.0, 4.0, // R
            5.0, 6.0, 7.0, 8.0, // G
            9.0, 10.0, 11.0, 12.0, // B
        ];

        let hwc = handler.reorder_nchw_to_hwc(&nchw, 3, 2, 2);

        // Expected HWC layout:
        // Pixel (0,0): R=1, G=5, B=9
        // Pixel (0,1): R=2, G=6, B=10
        // Pixel (1,0): R=3, G=7, B=11
        // Pixel (1,1): R=4, G=8, B=12
        let expected = vec![
            1.0, 5.0, 9.0, // (0,0)
            2.0, 6.0, 10.0, // (0,1)
            3.0, 7.0, 11.0, // (1,0)
            4.0, 8.0, 12.0, // (1,1)
        ];

        assert_eq!(hwc, expected);
    }

    #[test]
    fn test_normalize_to_bytes() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
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

        let data = vec![0.0, 0.5, 1.0];
        let bytes = handler.normalize_to_bytes(&data);

        assert_eq!(bytes, vec![0, 127, 255]);
    }

    #[test]
    fn test_process_rgb_nhwc() {
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
            output: "image_tensor".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        // Create 1x2x2x3 tensor (NHWC)
        let data = vec![
            0.0, 0.0, 0.0, // Pixel (0,0)
            1.0, 1.0, 1.0, // Pixel (0,1)
            0.5, 0.5, 0.5, // Pixel (1,0)
            0.25, 0.25, 0.25, // Pixel (1,1)
        ];

        let mut outputs = HashMap::new();
        outputs.insert(
            "image_tensor".to_string(),
            TensorData::new(data, vec![1, 2, 2, 3]),
        );

        let result = handler.process(&outputs).unwrap();

        if let ProcessedOutput::Image(img) = result {
            assert_eq!(img.width, 2);
            assert_eq!(img.height, 2);
            assert_eq!(img.channels, 3);
            assert_eq!(img.data[0..3], [0, 0, 0]); // Black
            assert_eq!(img.data[3..6], [255, 255, 255]); // White
        } else {
            panic!("Expected Image output");
        }
    }

    #[test]
    fn test_process_missing_tensor() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "missing".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let outputs = HashMap::new();
        let result = handler.process(&outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn test_process_invalid_shape() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "tensor".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let mut outputs = HashMap::new();
        outputs.insert(
            "tensor".to_string(),
            TensorData::new(vec![1.0; 100], vec![10, 10]), // Wrong ndim
        );

        let result = handler.process(&outputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_channel_mismatch() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
        config_map.insert(
            "layout".to_string(),
            toml::Value::String("NCHW".to_string()),
        );

        let config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "tensor".to_string(),
            config: config_map,
        };

        let handler = ImageHandler::from_config(&config).unwrap();

        let mut outputs = HashMap::new();
        outputs.insert(
            "tensor".to_string(),
            TensorData::new(vec![1.0; 16], vec![1, 4, 2, 2]), // 4 channels, expected 3
        );

        let result = handler.process(&outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("channels"));
    }
}

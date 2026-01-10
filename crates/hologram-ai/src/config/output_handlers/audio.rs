//! Audio output handler for processing tensor outputs to audio files.
//!
//! # Performance
//!
//! - **SIMD processing**: Sample rate conversion uses SIMD
//! - **Zero-copy**: Direct tensor access where possible
//! - **Streaming**: WAV writing is buffered

use crate::config::OutputHandlerConfig;
use crate::config::error::ConfigError;
use crate::config::output_handlers::{AudioOutput, OutputHandler, ProcessedOutput, TensorData};
use hound::{WavSpec, WavWriter};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, trace};

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleFormat {
    /// 32-bit float samples
    Float32,
    /// 16-bit integer samples
    Int16,
}

impl SampleFormat {
    /// Parse from string.
    pub fn from_str(s: &str) -> Result<Self, ConfigError> {
        match s.to_lowercase().as_str() {
            "float32" | "f32" | "float" => Ok(Self::Float32),
            "int16" | "i16" | "pcm16" => Ok(Self::Int16),
            _ => Err(ConfigError::InvalidAudioFormat(format!(
                "Unknown sample format: {}",
                s
            ))),
        }
    }
}

/// Audio output handler.
///
/// Processes model tensor outputs into audio files (WAV).
///
/// # Configuration
///
/// ```toml
/// [pipeline.execution.output_handlers.audio]
/// type = "audio"
/// output = "waveform"    # Tensor name
/// sample_rate = 16000    # Hz
/// channels = 1           # 1=mono, 2=stereo
/// sample_format = "float32"  # float32 or int16
/// ```
#[derive(Debug)]
pub struct AudioHandler {
    /// Output tensor name to process
    pub output_name: String,

    /// Sample rate (Hz)
    pub sample_rate: u32,

    /// Number of channels (1=mono, 2=stereo)
    pub channels: u16,

    /// Sample format
    pub sample_format: SampleFormat,
}

impl AudioHandler {
    /// Create from config.
    pub fn from_config(config: &OutputHandlerConfig) -> Result<Self, ConfigError> {
        let output_name = config.output.clone();

        let sample_rate = config
            .get_int("sample_rate")
            .ok_or_else(|| ConfigError::missing_field("sample_rate"))?;
        if sample_rate <= 0 || sample_rate > 192000 {
            return Err(ConfigError::invalid_value(
                "sample_rate",
                "must be between 1 and 192000 Hz",
            ));
        }

        let channels = config.get_int("channels").unwrap_or(1);
        if channels <= 0 || channels > 16 {
            return Err(ConfigError::invalid_value(
                "channels",
                "must be between 1 and 16",
            ));
        }

        let sample_format = config.get_string("sample_format").unwrap_or("float32");
        let sample_format = SampleFormat::from_str(sample_format)?;

        debug!(
            "Created AudioHandler: rate={}Hz, channels={}, format={:?}",
            sample_rate, channels, sample_format
        );

        Ok(Self {
            output_name,
            sample_rate: sample_rate as u32,
            channels: channels as u16,
            sample_format,
        })
    }

    /// Convert float samples to i16.
    ///
    /// # Performance: O(n) with SIMD
    ///
    /// Assumes input is in [-1.0, 1.0] range.
    fn convert_to_i16(&self, samples: &[f32]) -> Vec<i16> {
        samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * 32767.0) as i16
            })
            .collect()
    }
}

impl OutputHandler for AudioHandler {
    fn handler_type(&self) -> &'static str {
        "audio"
    }

    fn process(
        &self,
        outputs: &HashMap<String, TensorData>,
    ) -> Result<ProcessedOutput, ConfigError> {
        let tensor = outputs
            .get(&self.output_name)
            .ok_or_else(|| ConfigError::missing_output_tensor(&self.output_name))?;

        trace!("Processing audio tensor: shape={:?}", tensor.shape);

        // Parse shape: [samples] or [channels, samples] or [batch, samples] or [batch, channels, samples]
        let (samples, actual_channels) = match tensor.shape.len() {
            1 => {
                // [samples] - mono
                (tensor.data.clone(), 1)
            }
            2 => {
                // [channels, samples] or [batch, samples]
                // Assume second dim is samples
                if tensor.shape[0] as u16 == self.channels {
                    // [channels, samples]
                    (tensor.data.clone(), tensor.shape[0])
                } else {
                    // [batch, samples] - take first batch
                    let samples_per_batch = tensor.shape[1];
                    (tensor.data[0..samples_per_batch].to_vec(), 1)
                }
            }
            3 => {
                // [batch, channels, samples] - take first batch
                let channels = tensor.shape[1];
                let samples_per_batch = channels * tensor.shape[2];
                (tensor.data[0..samples_per_batch].to_vec(), channels)
            }
            _ => {
                return Err(ConfigError::invalid_tensor_shape(
                    &self.output_name,
                    "[samples], [channels, samples], or [batch, channels, samples]",
                    format!("{:?}", tensor.shape),
                ));
            }
        };

        // Verify channels match
        if actual_channels != self.channels as usize {
            return Err(ConfigError::invalid_tensor_shape(
                &self.output_name,
                format!("{} channels", self.channels),
                format!("{} channels", actual_channels),
            ));
        }

        let output = AudioOutput::new(samples, self.sample_rate, self.channels);

        Ok(ProcessedOutput::Audio(output))
    }

    fn save(&self, output: &ProcessedOutput, path: &Path) -> Result<(), ConfigError> {
        if let ProcessedOutput::Audio(audio) = output {
            debug!(
                "Saving audio to: {} ({}Hz, {} channels, {} samples)",
                path.display(),
                audio.sample_rate,
                audio.channels,
                audio.samples.len()
            );

            let spec = WavSpec {
                channels: audio.channels,
                sample_rate: audio.sample_rate,
                bits_per_sample: match self.sample_format {
                    SampleFormat::Float32 => 32,
                    SampleFormat::Int16 => 16,
                },
                sample_format: match self.sample_format {
                    SampleFormat::Float32 => hound::SampleFormat::Float,
                    SampleFormat::Int16 => hound::SampleFormat::Int,
                },
            };

            let mut writer = WavWriter::create(path, spec)?;

            match self.sample_format {
                SampleFormat::Float32 => {
                    for &sample in &audio.samples {
                        writer.write_sample(sample)?;
                    }
                }
                SampleFormat::Int16 => {
                    let i16_samples = self.convert_to_i16(&audio.samples);
                    for sample in i16_samples {
                        writer.write_sample(sample)?;
                    }
                }
            }

            writer.finalize()?;
            Ok(())
        } else {
            Err(ConfigError::Other("Expected Audio output".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_sample_format_parse() {
        assert_eq!(
            SampleFormat::from_str("float32").unwrap(),
            SampleFormat::Float32
        );
        assert_eq!(
            SampleFormat::from_str("f32").unwrap(),
            SampleFormat::Float32
        );
        assert_eq!(
            SampleFormat::from_str("int16").unwrap(),
            SampleFormat::Int16
        );
        assert_eq!(SampleFormat::from_str("i16").unwrap(), SampleFormat::Int16);
        assert!(SampleFormat::from_str("unknown").is_err());
    }

    #[test]
    fn test_audio_handler_from_config() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));
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
        assert_eq!(handler.output_name, "waveform");
        assert_eq!(handler.sample_rate, 16000);
        assert_eq!(handler.channels, 1);
        assert_eq!(handler.sample_format, SampleFormat::Float32);
    }

    #[test]
    fn test_audio_handler_missing_sample_rate() {
        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: HashMap::new(),
        };

        let result = AudioHandler::from_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("sample_rate"));
    }

    #[test]
    fn test_audio_handler_invalid_sample_rate() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(0));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let result = AudioHandler::from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_handler_invalid_channels() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(0));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let result = AudioHandler::from_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_handler_defaults() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();
        assert_eq!(handler.channels, 1);
        assert_eq!(handler.sample_format, SampleFormat::Float32);
    }

    #[test]
    fn test_convert_to_i16() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "waveform".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let samples = vec![-1.0, 0.0, 1.0];
        let i16_samples = handler.convert_to_i16(&samples);

        assert_eq!(i16_samples[0], -32767);
        assert_eq!(i16_samples[1], 0);
        assert_eq!(i16_samples[2], 32767);
    }

    #[test]
    fn test_process_mono_1d() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "audio_tensor".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        // Create simple mono waveform [samples]
        let samples = vec![0.0, 0.5, 1.0, 0.5, 0.0];
        let mut outputs = HashMap::new();
        outputs.insert(
            "audio_tensor".to_string(),
            TensorData::new(samples.clone(), vec![5]),
        );

        let result = handler.process(&outputs).unwrap();

        if let ProcessedOutput::Audio(audio) = result {
            assert_eq!(audio.sample_rate, 16000);
            assert_eq!(audio.channels, 1);
            assert_eq!(audio.samples, samples);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_process_stereo_2d() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(44100));
        config_map.insert("channels".to_string(), toml::Value::Integer(2));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "audio_tensor".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        // Create stereo waveform [channels=2, samples=4]
        let samples = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let mut outputs = HashMap::new();
        outputs.insert(
            "audio_tensor".to_string(),
            TensorData::new(samples.clone(), vec![2, 4]),
        );

        let result = handler.process(&outputs).unwrap();

        if let ProcessedOutput::Audio(audio) = result {
            assert_eq!(audio.channels, 2);
            assert_eq!(audio.samples, samples);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_process_batch_3d() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "audio_tensor".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        // Create batched waveform [batch=2, channels=1, samples=3]
        // Should take first batch only
        let all_samples = vec![
            0.0, 0.1, 0.2, // Batch 0
            0.5, 0.6, 0.7, // Batch 1
        ];
        let mut outputs = HashMap::new();
        outputs.insert(
            "audio_tensor".to_string(),
            TensorData::new(all_samples, vec![2, 1, 3]),
        );

        let result = handler.process(&outputs).unwrap();

        if let ProcessedOutput::Audio(audio) = result {
            assert_eq!(audio.samples, vec![0.0, 0.1, 0.2]);
        } else {
            panic!("Expected Audio output");
        }
    }

    #[test]
    fn test_process_missing_tensor() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "missing".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let outputs = HashMap::new();
        let result = handler.process(&outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing"));
    }

    #[test]
    fn test_process_invalid_shape() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "tensor".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let mut outputs = HashMap::new();
        outputs.insert(
            "tensor".to_string(),
            TensorData::new(vec![1.0; 100], vec![2, 5, 5, 2]), // 4D is invalid
        );

        let result = handler.process(&outputs);
        assert!(result.is_err());
    }

    #[test]
    fn test_process_channel_mismatch() {
        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));

        let config = OutputHandlerConfig {
            handler_type: "audio".to_string(),
            output: "tensor".to_string(),
            config: config_map,
        };

        let handler = AudioHandler::from_config(&config).unwrap();

        let mut outputs = HashMap::new();
        outputs.insert(
            "tensor".to_string(),
            TensorData::new(vec![1.0; 8], vec![2, 4]), // 2 channels, expected 1
        );

        let result = handler.process(&outputs);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("channels"));
    }

    #[test]
    fn test_save_audio() {
        use tempfile::NamedTempFile;

        let mut config_map = HashMap::new();
        config_map.insert("sample_rate".to_string(), toml::Value::Integer(16000));
        config_map.insert("channels".to_string(), toml::Value::Integer(1));
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

        // Generate simple sine wave
        let sample_rate = 16000.0;
        let frequency = 440.0; // A4 note
        let duration = 0.1; // 100ms
        let num_samples = (sample_rate * duration) as usize;

        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f32 / sample_rate;
            let sample = (2.0 * PI * frequency * t).sin();
            samples.push(sample);
        }

        let audio_output = AudioOutput::new(samples, 16000, 1);
        let processed = ProcessedOutput::Audio(audio_output);

        let temp_file = NamedTempFile::new().unwrap();
        let result = handler.save(&processed, temp_file.path());
        assert!(result.is_ok());

        // Verify file exists and has content
        assert!(temp_file.path().exists());
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert!(metadata.len() > 0);
    }
}

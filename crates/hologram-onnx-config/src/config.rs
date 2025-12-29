//! TOML configuration parsing for ONNX pipelines.
//!
//! # Performance
//!
//! - **Parse once**: Configs loaded at startup, zero runtime overhead
//! - **Validation**: All errors caught at load time
//! - **Zero-copy**: Config strings reference original TOML data where possible

use crate::error::ConfigError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::{debug, trace};

/// Pipeline configuration loaded from TOML.
///
/// # Example
///
/// ```toml
/// [pipeline]
/// name = "stable-diffusion"
/// version = "1.0"
///
/// [pipeline.execution]
/// inputs = ["prompt", "seed"]
/// outputs = ["image"]
///
/// [pipeline.execution.output_handlers.image]
/// type = "image"
/// output = "sample"
/// format = "rgb"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineConfig {
    /// Pipeline metadata
    pub pipeline: PipelineMetadata,
}

impl PipelineConfig {
    /// Load pipeline config from TOML file.
    ///
    /// # Performance: O(n) where n = file size
    ///
    /// Parse happens once at startup.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        debug!("Loading pipeline config from: {}", path.display());

        let contents = fs::read_to_string(path)?;
        Self::from_str(&contents)
    }

    /// Load pipeline config from TOML string.
    ///
    /// # Performance: O(n) where n = string length
    pub fn from_str(toml: &str) -> Result<Self, ConfigError> {
        trace!("Parsing TOML config");
        let config: PipelineConfig = toml::from_str(toml)?;
        config.validate()?;
        Ok(config)
    }

    /// Save pipeline config to TOML file.
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let toml = toml::to_string_pretty(self)?;
        fs::write(path, toml)?;
        Ok(())
    }

    /// Validate config structure.
    ///
    /// # Performance: O(1)
    ///
    /// Only checks required fields exist.
    fn validate(&self) -> Result<(), ConfigError> {
        // Validate execution config
        if let Some(ref exec) = self.pipeline.execution {
            if exec.inputs.is_empty() {
                return Err(ConfigError::invalid_value(
                    "pipeline.execution.inputs",
                    "must not be empty"
                ));
            }
            if exec.outputs.is_empty() {
                return Err(ConfigError::invalid_value(
                    "pipeline.execution.outputs",
                    "must not be empty"
                ));
            }
        }

        Ok(())
    }

    /// Get execution config if present.
    pub fn execution(&self) -> Option<&ExecutionConfig> {
        self.pipeline.execution.as_ref()
    }

    /// Get output handler configs.
    pub fn output_handlers(&self) -> HashMap<String, &OutputHandlerConfig> {
        self.pipeline.execution.as_ref()
            .and_then(|e| e.output_handlers.as_ref())
            .map(|h| h.iter().map(|(k, v)| (k.clone(), v)).collect())
            .unwrap_or_default()
    }
}

/// Pipeline metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PipelineMetadata {
    /// Pipeline name
    pub name: String,

    /// Pipeline version
    pub version: String,

    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Execution configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionConfig>,
}

/// Execution configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionConfig {
    /// Input variable names
    pub inputs: Vec<String>,

    /// Output variable names
    pub outputs: Vec<String>,

    /// Output handlers by name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_handlers: Option<HashMap<String, OutputHandlerConfig>>,

    /// Pipeline stages (for multi-model pipelines)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stages: Option<Vec<StageConfig>>,
}

/// Pipeline stage configuration (for multi-model pipelines).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageConfig {
    /// Stage name
    pub name: String,

    /// Model file path (.holo)
    pub model: String,

    /// Input mappings (variable name → tensor name)
    pub inputs: HashMap<String, String>,

    /// Output tensor names
    pub outputs: Vec<String>,
}

/// Output handler configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutputHandlerConfig {
    /// Handler type ("image", "audio", "text", "tensor")
    #[serde(rename = "type")]
    pub handler_type: String,

    /// Output tensor name to process
    pub output: String,

    /// Handler-specific configuration (flatten rest)
    #[serde(flatten)]
    pub config: HashMap<String, toml::Value>,
}

impl OutputHandlerConfig {
    /// Get string config value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.config.get(key)
            .and_then(|v| v.as_str())
    }

    /// Get integer config value.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.config.get(key)
            .and_then(|v| v.as_integer())
    }

    /// Get float config value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.config.get(key)
            .and_then(|v| v.as_float())
    }

    /// Get boolean config value.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.config.get(key)
            .and_then(|v| v.as_bool())
    }

    /// Get array config value.
    pub fn get_array(&self, key: &str) -> Option<&Vec<toml::Value>> {
        self.config.get(key)
            .and_then(|v| v.as_array())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_minimal_config() -> PipelineConfig {
        PipelineConfig {
            pipeline: PipelineMetadata {
                name: "test".to_string(),
                version: "1.0".to_string(),
                description: None,
                execution: Some(ExecutionConfig {
                    inputs: vec!["input".to_string()],
                    outputs: vec!["output".to_string()],
                    output_handlers: None,
                    stages: None,
                }),
            },
        }
    }

    #[test]
    fn test_minimal_config_parse() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"

            [pipeline.execution]
            inputs = ["input"]
            outputs = ["output"]
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        assert_eq!(config.pipeline.name, "test");
        assert_eq!(config.pipeline.version, "1.0");
        assert_eq!(config.execution().unwrap().inputs, vec!["input"]);
        assert_eq!(config.execution().unwrap().outputs, vec!["output"]);
    }

    #[test]
    fn test_config_with_description() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"
            description = "Test pipeline"

            [pipeline.execution]
            inputs = ["input"]
            outputs = ["output"]
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        assert_eq!(config.pipeline.description, Some("Test pipeline".to_string()));
    }

    #[test]
    fn test_config_with_output_handler() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"

            [pipeline.execution]
            inputs = ["prompt"]
            outputs = ["image"]

            [pipeline.execution.output_handlers.image]
            type = "image"
            output = "sample"
            format = "rgb"
            layout = "NCHW"
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        let handlers = config.output_handlers();
        assert_eq!(handlers.len(), 1);

        let image_handler = handlers.get("image").unwrap();
        assert_eq!(image_handler.handler_type, "image");
        assert_eq!(image_handler.output, "sample");
        assert_eq!(image_handler.get_string("format"), Some("rgb"));
        assert_eq!(image_handler.get_string("layout"), Some("NCHW"));
    }

    #[test]
    fn test_config_with_stages() {
        let toml = r#"
            [pipeline]
            name = "multi-stage"
            version = "1.0"

            [pipeline.execution]
            inputs = ["text"]
            outputs = ["audio"]

            [[pipeline.execution.stages]]
            name = "encoder"
            model = "encoder.holo"
            inputs = { text = "input_ids" }
            outputs = ["embeddings"]

            [[pipeline.execution.stages]]
            name = "decoder"
            model = "decoder.holo"
            inputs = { embeddings = "encoder_hidden_states" }
            outputs = ["audio"]
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        let stages = config.execution().unwrap().stages.as_ref().unwrap();
        assert_eq!(stages.len(), 2);

        assert_eq!(stages[0].name, "encoder");
        assert_eq!(stages[0].model, "encoder.holo");
        assert_eq!(stages[0].outputs, vec!["embeddings"]);

        assert_eq!(stages[1].name, "decoder");
        assert_eq!(stages[1].model, "decoder.holo");
        assert_eq!(stages[1].outputs, vec!["audio"]);
    }

    #[test]
    fn test_validation_empty_inputs() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"

            [pipeline.execution]
            inputs = []
            outputs = ["output"]
        "#;

        let result = PipelineConfig::from_str(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("inputs"));
    }

    #[test]
    fn test_validation_empty_outputs() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"

            [pipeline.execution]
            inputs = ["input"]
            outputs = []
        "#;

        let result = PipelineConfig::from_str(toml);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outputs"));
    }

    #[test]
    fn test_config_round_trip() {
        let config = make_minimal_config();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed = PipelineConfig::from_str(&toml_str).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_output_handler_config_getters() {
        let mut config_map = HashMap::new();
        config_map.insert("format".to_string(), toml::Value::String("rgb".to_string()));
        config_map.insert("width".to_string(), toml::Value::Integer(512));
        config_map.insert("scale".to_string(), toml::Value::Float(2.5));
        config_map.insert("enabled".to_string(), toml::Value::Boolean(true));

        let handler_config = OutputHandlerConfig {
            handler_type: "image".to_string(),
            output: "sample".to_string(),
            config: config_map,
        };

        assert_eq!(handler_config.get_string("format"), Some("rgb"));
        assert_eq!(handler_config.get_int("width"), Some(512));
        assert_eq!(handler_config.get_float("scale"), Some(2.5));
        assert_eq!(handler_config.get_bool("enabled"), Some(true));
        assert_eq!(handler_config.get_string("missing"), None);
    }

    #[test]
    fn test_get_output_handlers() {
        let toml = r#"
            [pipeline]
            name = "test"
            version = "1.0"

            [pipeline.execution]
            inputs = ["input"]
            outputs = ["image", "audio"]

            [pipeline.execution.output_handlers.image]
            type = "image"
            output = "img_tensor"

            [pipeline.execution.output_handlers.audio]
            type = "audio"
            output = "audio_tensor"
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        let handlers = config.output_handlers();
        assert_eq!(handlers.len(), 2);
        assert!(handlers.contains_key("image"));
        assert!(handlers.contains_key("audio"));
    }

    #[test]
    fn test_config_no_execution() {
        let toml = r#"
            [pipeline]
            name = "simple"
            version = "1.0"
        "#;

        let config = PipelineConfig::from_str(toml).unwrap();
        assert!(config.execution().is_none());
        assert_eq!(config.output_handlers().len(), 0);
    }
}

//! Conversion adapters between UnifiedConfig and existing config types.
//!
//! This module provides conversions from the unified configuration format
//! to the specialized config types used by different parts of the system.
//!
//! NOTE: During API migration, many config fields are not used by the
//! simplified hologram-ai-onnx compile API.

use std::collections::HashMap;

#[cfg(feature = "onnx")]
use hologram_ai_onnx::compat::OnnxConfig;

use super::pipeline::{
    ExecutionConfig, OutputHandlerConfig, PipelineConfig, PipelineMetadata, StageConfig,
};
use super::unified::{
    CompilerConfig, ModelStage, OutputDef, OutputHandlerType, StageDef, UnifiedConfig,
};

// =============================================================================
// CompilerConfig → OnnxConfig
// =============================================================================

impl From<&CompilerConfig> for OnnxConfig {
    /// Convert unified CompilerConfig to core OnnxConfig.
    ///
    /// Note: Many CompilerConfig fields are not used by the simplified OnnxConfig.
    fn from(config: &CompilerConfig) -> Self {
        OnnxConfig {
            opset_version: None, // Auto-detect from model
            optimize: config.opt_level > 0,
            enable_partitioning: config.enable_partitioning,
            partition_size: config.partition_size,
        }
    }
}

impl From<CompilerConfig> for OnnxConfig {
    fn from(config: CompilerConfig) -> Self {
        OnnxConfig::from(&config)
    }
}

// =============================================================================
// OnnxConfig → CompilerConfig
// =============================================================================

impl From<&OnnxConfig> for CompilerConfig {
    /// Convert core OnnxConfig to unified CompilerConfig.
    fn from(config: &OnnxConfig) -> Self {
        CompilerConfig {
            weight_threshold: 4096, // Default
            enable_partitioning: config.enable_partitioning,
            partition_size: config.partition_size,
            decompose_conv2d: false,
            decompose_pooling: false,
            pack_weights: true,
            memory_budget: None,
            enable_resize_upscaling: false,
            backend: None,
            aggressive_fusion: false,
            opt_level: if config.optimize { 2 } else { 0 },
            auto_fuse: true,
            use_fp16: false,
            use_int8: false,
            quantization_mode: "none".to_string(),
        }
    }
}

impl From<OnnxConfig> for CompilerConfig {
    fn from(config: OnnxConfig) -> Self {
        CompilerConfig::from(&config)
    }
}

// =============================================================================
// UnifiedConfig → PipelineConfig
// =============================================================================

impl From<&UnifiedConfig> for PipelineConfig {
    fn from(config: &UnifiedConfig) -> Self {
        // Extract input names
        let inputs: Vec<String> = config.inputs.keys().cloned().collect();

        // Extract output names
        let outputs: Vec<String> = config.outputs.keys().cloned().collect();

        // Convert output handlers
        let output_handlers = convert_output_handlers(&config.outputs);

        // Convert stages (only model stages are supported in PipelineConfig)
        let stages = convert_stages(&config.stages, &config.models);

        // Build execution config
        let execution = Some(ExecutionConfig {
            inputs: if inputs.is_empty() {
                vec!["input".to_string()]
            } else {
                inputs
            },
            outputs: if outputs.is_empty() {
                vec!["output".to_string()]
            } else {
                outputs
            },
            output_handlers: if output_handlers.is_empty() {
                None
            } else {
                Some(output_handlers)
            },
            stages: if stages.is_empty() {
                None
            } else {
                Some(stages)
            },
        });

        PipelineConfig {
            pipeline: PipelineMetadata {
                name: config.name.clone().unwrap_or_else(|| {
                    config
                        .models
                        .values()
                        .next()
                        .map(|m| m.infer_name())
                        .unwrap_or_else(|| "pipeline".to_string())
                }),
                version: config.version.clone().unwrap_or_else(|| "1.0".to_string()),
                description: config.description.clone(),
                execution,
            },
        }
    }
}

impl From<UnifiedConfig> for PipelineConfig {
    fn from(config: UnifiedConfig) -> Self {
        PipelineConfig::from(&config)
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn convert_output_handlers(
    outputs: &HashMap<String, OutputDef>,
) -> HashMap<String, OutputHandlerConfig> {
    outputs
        .iter()
        .filter_map(|(name, output)| {
            let handler_type = output.handler_type();
            if handler_type == OutputHandlerType::Auto {
                return None;
            }

            let handler_config = OutputHandlerConfig {
                handler_type: handler_type_to_string(&handler_type),
                output: output.tensor().to_string(),
                config: convert_output_options(output),
            };

            Some((name.clone(), handler_config))
        })
        .collect()
}

fn handler_type_to_string(handler_type: &OutputHandlerType) -> String {
    match handler_type {
        OutputHandlerType::Image => "image".to_string(),
        OutputHandlerType::Audio => "audio".to_string(),
        OutputHandlerType::Text => "text".to_string(),
        OutputHandlerType::Json => "json".to_string(),
        OutputHandlerType::Binary => "binary".to_string(),
        OutputHandlerType::Auto => "auto".to_string(),
    }
}

fn convert_output_options(output: &OutputDef) -> HashMap<String, toml::Value> {
    match output {
        OutputDef::Simple(_) => HashMap::new(),
        OutputDef::Full(spec) => spec
            .options
            .iter()
            .filter_map(|(k, v)| json_to_toml(v).map(|toml_val| (k.clone(), toml_val)))
            .collect(),
    }
}

fn json_to_toml(json: &serde_json::Value) -> Option<toml::Value> {
    match json {
        serde_json::Value::Null => None,
        serde_json::Value::Bool(b) => Some(toml::Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(toml::Value::Integer(i))
            } else {
                n.as_f64().map(toml::Value::Float)
            }
        }
        serde_json::Value::String(s) => Some(toml::Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let toml_arr: Vec<toml::Value> = arr.iter().filter_map(json_to_toml).collect();
            Some(toml::Value::Array(toml_arr))
        }
        serde_json::Value::Object(obj) => {
            let toml_table: toml::map::Map<String, toml::Value> = obj
                .iter()
                .filter_map(|(k, v)| json_to_toml(v).map(|tv| (k.clone(), tv)))
                .collect();
            Some(toml::Value::Table(toml_table))
        }
    }
}

fn convert_stages(
    stages: &[StageDef],
    models: &HashMap<String, crate::config::unified::ModelDef>,
) -> Vec<StageConfig> {
    stages
        .iter()
        .filter_map(|stage| match stage {
            StageDef::Model(model_stage) => Some(convert_model_stage(model_stage, models)),
            _ => None,
        })
        .collect()
}

fn convert_model_stage(
    stage: &ModelStage,
    models: &HashMap<String, crate::config::unified::ModelDef>,
) -> StageConfig {
    let model_path = models
        .get(&stage.model)
        .map(|m| {
            let path = m.path();
            if let Some(stripped) = path.strip_suffix(".onnx") {
                format!("{}.holo", stripped)
            } else {
                format!("{}.holo", path)
            }
        })
        .unwrap_or_else(|| format!("{}.holo", stage.model));

    let inputs: HashMap<String, String> = stage
        .inputs
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect();

    StageConfig {
        name: stage.model.clone(),
        model: model_path,
        inputs,
        outputs: stage.outputs.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiler_config_to_onnx_config() {
        let compiler_config = CompilerConfig {
            enable_partitioning: true,
            partition_size: 1000,
            opt_level: 2,
            ..Default::default()
        };

        let onnx_config: OnnxConfig = (&compiler_config).into();

        assert!(onnx_config.enable_partitioning);
        assert_eq!(onnx_config.partition_size, 1000);
        assert!(onnx_config.optimize);
    }

    #[test]
    fn test_handler_type_to_string() {
        assert_eq!(handler_type_to_string(&OutputHandlerType::Image), "image");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Audio), "audio");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Text), "text");
    }

    #[test]
    fn test_json_to_toml_primitives() {
        assert_eq!(
            json_to_toml(&serde_json::Value::Bool(true)),
            Some(toml::Value::Boolean(true))
        );
        assert_eq!(
            json_to_toml(&serde_json::json!(42)),
            Some(toml::Value::Integer(42))
        );
        assert_eq!(json_to_toml(&serde_json::Value::Null), None);
    }
}

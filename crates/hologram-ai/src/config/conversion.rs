//! Conversion adapters between UnifiedConfig and existing config types.
//!
//! This module provides conversions from the unified configuration format
//! to the specialized config types used by different parts of the system:
//!
//! - `OnnxConfig`: Used by hologram-onnx-core for compilation settings
//! - `PipelineConfig`: Used for execution pipeline configuration
//!
//! # Example
//!
//! ```rust,ignore
//! use crate::config::{UnifiedConfig, conversion};
//! use hologram_onnx::core::OnnxConfig;
//!
//! let unified = UnifiedConfig::from_file("pipeline.toml")?;
//!
//! // Convert to compilation config
//! let onnx_config: OnnxConfig = (&unified.compiler).into();
//!
//! // Convert to execution config
//! let pipeline_config: PipelineConfig = (&unified).into();
//! ```

use std::collections::HashMap;

#[cfg(feature = "onnx")]
use hologram_ai_onnx::core::OnnxConfig;

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
    /// The `backend` field from CompilerConfig is not used in OnnxConfig
    /// as it's handled separately by the runtime.
    fn from(config: &CompilerConfig) -> Self {
        OnnxConfig {
            weight_threshold: config.weight_threshold,
            enable_partitioning: config.enable_partitioning,
            partition_size: config.partition_size,
            decompose_conv2d: config.decompose_conv2d,
            decompose_pooling: config.decompose_pooling,
            pack_weights: config.pack_weights,
            memory_budget: config.memory_budget,
            enable_resize_upscaling: config.enable_resize_upscaling,
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
    ///
    /// The `backend` field is set to None as it's not present in OnnxConfig.
    fn from(config: &OnnxConfig) -> Self {
        CompilerConfig {
            weight_threshold: config.weight_threshold,
            enable_partitioning: config.enable_partitioning,
            partition_size: config.partition_size,
            decompose_conv2d: config.decompose_conv2d,
            decompose_pooling: config.decompose_pooling,
            pack_weights: config.pack_weights,
            memory_budget: config.memory_budget,
            enable_resize_upscaling: config.enable_resize_upscaling,
            backend: None,
            aggressive_fusion: false,
            opt_level: 2,
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
    /// Convert UnifiedConfig to PipelineConfig for execution.
    ///
    /// This conversion:
    /// - Extracts pipeline metadata (name, version, description)
    /// - Converts inputs to input names
    /// - Converts outputs to output names
    /// - Converts output handlers to OutputHandlerConfig
    /// - Converts stages to StageConfig (model stages only)
    fn from(config: &UnifiedConfig) -> Self {
        // Extract input names
        let inputs: Vec<String> = config.inputs.keys().cloned().collect();

        // Extract output names
        let outputs: Vec<String> = config.outputs.keys().cloned().collect();

        // Convert output handlers
        let output_handlers = convert_output_handlers(&config.outputs);

        // Convert stages (only model stages are supported in PipelineConfig)
        let stages = convert_stages(&config.stages, &config.models);

        // Build execution config if we have inputs and outputs
        let execution = if !inputs.is_empty() || !outputs.is_empty() {
            Some(ExecutionConfig {
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
            })
        } else {
            // Create minimal execution config for single-model case
            Some(ExecutionConfig {
                inputs: vec!["input".to_string()],
                outputs: vec!["output".to_string()],
                output_handlers: None,
                stages: None,
            })
        };

        PipelineConfig {
            pipeline: PipelineMetadata {
                name: config.name.clone().unwrap_or_else(|| {
                    // Infer name from first model
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

/// Convert unified OutputDef map to OutputHandlerConfig map.
fn convert_output_handlers(
    outputs: &HashMap<String, OutputDef>,
) -> HashMap<String, OutputHandlerConfig> {
    outputs
        .iter()
        .filter_map(|(name, output)| {
            // Only create handler config for outputs with explicit handlers
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

/// Convert OutputHandlerType to string for PipelineConfig.
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

/// Convert output options from serde_json::Value to toml::Value.
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

/// Convert serde_json::Value to toml::Value.
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

/// Convert unified stages to PipelineConfig stages.
///
/// Only ModelStage types are converted; Loop and Conditional stages
/// are not supported by the basic PipelineConfig format.
fn convert_stages(
    stages: &[StageDef],
    models: &HashMap<String, crate::config::unified::ModelDef>,
) -> Vec<StageConfig> {
    stages
        .iter()
        .filter_map(|stage| {
            match stage {
                StageDef::Model(model_stage) => Some(convert_model_stage(model_stage, models)),
                // Builtin, Loop, and Conditional stages are not supported
                // in the basic PipelineConfig format
                _ => None,
            }
        })
        .collect()
}

/// Convert a ModelStage to StageConfig.
fn convert_model_stage(
    stage: &ModelStage,
    models: &HashMap<String, crate::config::unified::ModelDef>,
) -> StageConfig {
    // Get the model path, constructing .holo path
    let model_path = models
        .get(&stage.model)
        .map(|m| {
            // Convert .onnx to .holo
            let path = m.path();
            if let Some(stripped) = path.strip_suffix(".onnx") {
                format!("{}.holo", stripped)
            } else {
                format!("{}.holo", path)
            }
        })
        .unwrap_or_else(|| format!("{}.holo", stage.model));

    // Convert inputs: Expr → String (extract string value)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::unified::{InputDef, InputSpec, InputType, ModelDef, OutputSpec};

    #[test]
    fn test_compiler_config_to_onnx_config() {
        let compiler_config = CompilerConfig {
            weight_threshold: 8192,
            enable_partitioning: true,
            partition_size: 1000,
            decompose_conv2d: true,
            decompose_pooling: false,
            pack_weights: true,
            memory_budget: Some(8 * 1024),
            enable_resize_upscaling: false,
            backend: Some("cuda".to_string()),
            aggressive_fusion: false,
            opt_level: 2,
            auto_fuse: true,
            use_fp16: false,
            use_int8: false,
            quantization_mode: "none".to_string(),
        };

        let onnx_config: OnnxConfig = (&compiler_config).into();

        assert_eq!(onnx_config.weight_threshold, 8192);
        assert!(onnx_config.enable_partitioning);
        assert_eq!(onnx_config.partition_size, 1000);
        assert!(onnx_config.decompose_conv2d);
        assert!(!onnx_config.decompose_pooling);
        assert!(onnx_config.pack_weights);
        assert_eq!(onnx_config.memory_budget, Some(8 * 1024));
        assert!(!onnx_config.enable_resize_upscaling);
    }

    #[test]
    fn test_onnx_config_to_compiler_config() {
        let onnx_config = OnnxConfig {
            weight_threshold: 4096,
            enable_partitioning: false,
            partition_size: 500,
            decompose_conv2d: true,
            decompose_pooling: true,
            pack_weights: false,
            memory_budget: None,
            enable_resize_upscaling: true,
        };

        let compiler_config: CompilerConfig = onnx_config.into();

        assert_eq!(compiler_config.weight_threshold, 4096);
        assert!(!compiler_config.enable_partitioning);
        assert_eq!(compiler_config.partition_size, 500);
        assert!(compiler_config.decompose_conv2d);
        assert!(compiler_config.decompose_pooling);
        assert!(!compiler_config.pack_weights);
        assert_eq!(compiler_config.memory_budget, None);
        assert_eq!(compiler_config.backend, None);
        assert!(compiler_config.enable_resize_upscaling);
    }

    #[test]
    fn test_unified_to_pipeline_minimal() {
        let toml = r#"
            model = "mnist.onnx"
        "#;

        let unified = UnifiedConfig::from_str(toml).unwrap();
        let pipeline: PipelineConfig = unified.into();

        assert_eq!(pipeline.pipeline.name, "mnist");
        assert_eq!(pipeline.pipeline.version, "1.0");
        assert!(pipeline.pipeline.execution.is_some());
    }

    #[test]
    fn test_unified_to_pipeline_with_metadata() {
        let toml = r#"
            name = "my-pipeline"
            version = "2.0"
            description = "Test pipeline"
            model = "test.onnx"
        "#;

        let unified = UnifiedConfig::from_str(toml).unwrap();
        let pipeline: PipelineConfig = unified.into();

        assert_eq!(pipeline.pipeline.name, "my-pipeline");
        assert_eq!(pipeline.pipeline.version, "2.0");
        assert_eq!(
            pipeline.pipeline.description,
            Some("Test pipeline".to_string())
        );
    }

    #[test]
    fn test_unified_to_pipeline_with_inputs_outputs() {
        let mut unified = UnifiedConfig {
            name: Some("test".to_string()),
            ..Default::default()
        };

        unified.inputs.insert(
            "prompt".to_string(),
            InputDef::Full(InputSpec {
                dtype: InputType::Text,
                default: Some(serde_json::Value::String("hello".to_string())),
                shape: None,
            }),
        );

        unified.inputs.insert(
            "image".to_string(),
            InputDef::Full(InputSpec {
                dtype: InputType::Image,
                default: None,
                shape: None,
            }),
        );

        unified.outputs.insert(
            "result".to_string(),
            OutputDef::Full(OutputSpec {
                tensor: "output_tensor".to_string(),
                handler: OutputHandlerType::Image,
                options: HashMap::new(),
            }),
        );

        let pipeline: PipelineConfig = unified.into();
        let exec = pipeline.pipeline.execution.unwrap();

        assert_eq!(exec.inputs.len(), 2);
        assert!(exec.inputs.contains(&"prompt".to_string()));
        assert!(exec.inputs.contains(&"image".to_string()));
        assert_eq!(exec.outputs.len(), 1);
        assert!(exec.outputs.contains(&"result".to_string()));

        let handlers = exec.output_handlers.unwrap();
        assert_eq!(handlers.len(), 1);
        let handler = handlers.get("result").unwrap();
        assert_eq!(handler.handler_type, "image");
        assert_eq!(handler.output, "output_tensor");
    }

    #[test]
    fn test_unified_to_pipeline_with_stages() {
        let mut unified = UnifiedConfig {
            name: Some("multi-model".to_string()),
            ..Default::default()
        };

        unified.models.insert(
            "encoder".to_string(),
            ModelDef::Path("models/encoder.onnx".to_string()),
        );
        unified.models.insert(
            "decoder".to_string(),
            ModelDef::Path("models/decoder.onnx".to_string()),
        );

        unified.stages.push(StageDef::Model(ModelStage {
            model: "encoder".to_string(),
            inputs: HashMap::new(),
            outputs: vec!["embeddings".to_string()],
        }));

        unified.stages.push(StageDef::Model(ModelStage {
            model: "decoder".to_string(),
            inputs: HashMap::new(),
            outputs: vec!["output".to_string()],
        }));

        unified
            .inputs
            .insert("input".to_string(), InputDef::Simple("test".to_string()));
        unified.outputs.insert(
            "output".to_string(),
            OutputDef::Simple("result".to_string()),
        );

        let pipeline: PipelineConfig = unified.into();
        let exec = pipeline.pipeline.execution.unwrap();
        let stages = exec.stages.unwrap();

        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].name, "encoder");
        assert_eq!(stages[0].model, "models/encoder.holo");
        assert_eq!(stages[0].outputs, vec!["embeddings"]);

        assert_eq!(stages[1].name, "decoder");
        assert_eq!(stages[1].model, "models/decoder.holo");
    }

    #[test]
    fn test_handler_type_to_string() {
        assert_eq!(handler_type_to_string(&OutputHandlerType::Image), "image");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Audio), "audio");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Text), "text");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Json), "json");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Binary), "binary");
        assert_eq!(handler_type_to_string(&OutputHandlerType::Auto), "auto");
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
        assert_eq!(
            json_to_toml(&serde_json::json!(std::f64::consts::PI)),
            Some(toml::Value::Float(std::f64::consts::PI))
        );
        assert_eq!(
            json_to_toml(&serde_json::Value::String("hello".to_string())),
            Some(toml::Value::String("hello".to_string()))
        );
        assert_eq!(json_to_toml(&serde_json::Value::Null), None);
    }

    #[test]
    fn test_json_to_toml_array() {
        let json_arr = serde_json::json!([1, 2, 3]);
        let toml_val = json_to_toml(&json_arr).unwrap();

        if let toml::Value::Array(arr) = toml_val {
            assert_eq!(arr.len(), 3);
            assert_eq!(arr[0], toml::Value::Integer(1));
        } else {
            panic!("Expected array");
        }
    }

    #[test]
    fn test_output_options_conversion() {
        let mut options = HashMap::new();
        options.insert("format".to_string(), serde_json::json!("rgb"));
        options.insert("width".to_string(), serde_json::json!(512));

        let output = OutputDef::Full(OutputSpec {
            tensor: "image".to_string(),
            handler: OutputHandlerType::Image,
            options,
        });

        let converted = convert_output_options(&output);
        assert_eq!(
            converted.get("format"),
            Some(&toml::Value::String("rgb".to_string()))
        );
        assert_eq!(converted.get("width"), Some(&toml::Value::Integer(512)));
    }

    #[test]
    fn test_round_trip_compiler_config() {
        let original = CompilerConfig {
            weight_threshold: 8192,
            enable_partitioning: true,
            partition_size: 750,
            decompose_conv2d: false,
            decompose_pooling: true,
            pack_weights: true,
            memory_budget: Some(4096),
            enable_resize_upscaling: false,
            backend: None,
            aggressive_fusion: true,
            opt_level: 3,
            auto_fuse: false,
            use_fp16: true,
            use_int8: false,
            quantization_mode: "dynamic".to_string(),
        };

        let onnx: OnnxConfig = (&original).into();
        let back: CompilerConfig = onnx.into();

        // Core fields that round-trip perfectly
        assert_eq!(original.weight_threshold, back.weight_threshold);
        assert_eq!(original.enable_partitioning, back.enable_partitioning);
        assert_eq!(original.partition_size, back.partition_size);
        assert_eq!(original.decompose_conv2d, back.decompose_conv2d);
        assert_eq!(original.decompose_pooling, back.decompose_pooling);
        assert_eq!(original.pack_weights, back.pack_weights);
        assert_eq!(original.memory_budget, back.memory_budget);
        assert_eq!(original.enable_resize_upscaling, back.enable_resize_upscaling);

        // Note: Fields like backend, aggressive_fusion, opt_level, auto_fuse, use_fp16,
        // use_int8, and quantization_mode have default values in the conversion and
        // don't round-trip perfectly. This is expected behavior.
    }
}

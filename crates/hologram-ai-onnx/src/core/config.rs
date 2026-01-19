//! Compilation configuration for ONNX models.
//!
//! This module defines configuration options that control the ONNX compilation
//! pipeline, including memory management, graph partitioning, ISA optimizations,
//! and embedded sections for single-file model distribution.

use std::path::PathBuf;

/// Type of section to embed in the bundle.
///
/// This enum specifies how a file should be interpreted when embedding
/// it as a section in a `.holo` bundle. Each type corresponds to a
/// specific section implementation with appropriate parsing and validation.
///
/// # Examples
///
/// ```
/// use hologram_ai_onnx::core::SectionType;
///
/// // WordPiece vocabulary (vocab.txt)
/// let vocab_type = SectionType::Vocabulary;
///
/// // JSON vocabulary (vocab.json)
/// let vocab_json = SectionType::VocabularyJson;
///
/// // Arbitrary file with custom content type
/// let raw = SectionType::Raw {
///     content_type: "application/x-custom".to_string(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SectionType {
    /// WordPiece/BPE line-based vocabulary (vocab.txt format).
    /// Each line is a token, line number is the token ID.
    Vocabulary,

    /// JSON vocabulary (vocab.json format).
    /// Maps token strings to integer IDs.
    VocabularyJson,

    /// Tokenizer configuration (tokenizer_config.json).
    TokenizerConfig,

    /// Model configuration (config.json).
    ModelConfig,

    /// Special tokens map (special_tokens_map.json).
    SpecialTokensMap,

    /// Preprocessor configuration (preprocessor_config.json).
    PreprocessorConfig,

    /// SentencePiece model (*.model binary format).
    SentencePiece,

    /// Generation configuration (generation_config.json).
    GenerationConfig,

    /// Arbitrary file with custom content type.
    Raw {
        /// MIME content type for the file (e.g., "text/plain", "application/octet-stream").
        content_type: String,
    },
}

impl SectionType {
    /// Get the default section ID for this type.
    ///
    /// Returns the standard section ID used for each type. For `Raw` types,
    /// a custom ID should typically be provided in `EmbeddedFileConfig`.
    pub fn default_section_id(&self) -> &'static str {
        match self {
            Self::Vocabulary | Self::VocabularyJson => "vocabulary",
            Self::TokenizerConfig => "tokenizer_config",
            Self::ModelConfig => "model_config",
            Self::SpecialTokensMap => "special_tokens",
            Self::PreprocessorConfig => "preprocessor_config",
            Self::SentencePiece => "sentencepiece",
            Self::GenerationConfig => "generation_config",
            Self::Raw { .. } => "raw",
        }
    }
}

/// Configuration for embedding a file in the bundle.
///
/// This structure specifies a file to be embedded as a section in the
/// compiled `.holo` bundle. The file will be loaded, validated, and
/// stored with the appropriate section type and ID.
///
/// # Examples
///
/// ```
/// use hologram_ai_onnx::core::{EmbeddedFileConfig, SectionType};
/// use std::path::PathBuf;
///
/// // Embed a vocabulary file
/// let vocab_config = EmbeddedFileConfig {
///     path: PathBuf::from("models/bert/vocab.txt"),
///     section_type: SectionType::Vocabulary,
///     custom_id: None, // Use default ID "vocabulary"
/// };
///
/// // Embed a custom file with custom ID
/// let custom_config = EmbeddedFileConfig {
///     path: PathBuf::from("models/bert/custom_data.bin"),
///     section_type: SectionType::Raw {
///         content_type: "application/octet-stream".to_string(),
///     },
///     custom_id: Some("custom_data".to_string()),
/// };
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddedFileConfig {
    /// Path to the file to embed.
    ///
    /// Can be relative (resolved from the ONNX model directory) or absolute.
    pub path: PathBuf,

    /// Type of section to create from this file.
    pub section_type: SectionType,

    /// Optional custom section ID.
    ///
    /// If `None`, the default ID for the section type is used.
    /// If `Some`, this ID overrides the default.
    pub custom_id: Option<String>,
}

impl EmbeddedFileConfig {
    /// Create a new embedded file configuration.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to embed
    /// * `section_type` - Type of section to create
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::{EmbeddedFileConfig, SectionType};
    /// use std::path::PathBuf;
    ///
    /// let config = EmbeddedFileConfig::new(
    ///     PathBuf::from("vocab.txt"),
    ///     SectionType::Vocabulary,
    /// );
    /// ```
    pub fn new(path: PathBuf, section_type: SectionType) -> Self {
        Self {
            path,
            section_type,
            custom_id: None,
        }
    }

    /// Create an embedded file config with a custom section ID.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to embed
    /// * `section_type` - Type of section to create
    /// * `custom_id` - Custom section ID to use
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::{EmbeddedFileConfig, SectionType};
    /// use std::path::PathBuf;
    ///
    /// let config = EmbeddedFileConfig::with_id(
    ///     PathBuf::from("custom.bin"),
    ///     SectionType::Raw { content_type: "application/octet-stream".to_string() },
    ///     "my_custom_section",
    /// );
    /// assert_eq!(config.section_id(), "my_custom_section");
    /// ```
    pub fn with_id(path: PathBuf, section_type: SectionType, custom_id: &str) -> Self {
        Self {
            path,
            section_type,
            custom_id: Some(custom_id.to_string()),
        }
    }

    /// Get the section ID to use for this embedded file.
    ///
    /// Returns the custom ID if set, otherwise the default ID for the section type.
    pub fn section_id(&self) -> &str {
        self.custom_id
            .as_deref()
            .unwrap_or_else(|| self.section_type.default_section_id())
    }

    /// Create a vocabulary file configuration.
    ///
    /// Convenience method for creating a line-based vocabulary section.
    pub fn vocabulary(path: impl Into<PathBuf>) -> Self {
        Self::new(path.into(), SectionType::Vocabulary)
    }

    /// Create a JSON vocabulary file configuration.
    ///
    /// Convenience method for creating a JSON vocabulary section.
    pub fn vocabulary_json(path: impl Into<PathBuf>) -> Self {
        Self::new(path.into(), SectionType::VocabularyJson)
    }

    /// Create a tokenizer config file configuration.
    ///
    /// Convenience method for creating a tokenizer config section.
    pub fn tokenizer_config(path: impl Into<PathBuf>) -> Self {
        Self::new(path.into(), SectionType::TokenizerConfig)
    }

    /// Create a model config file configuration.
    ///
    /// Convenience method for creating a model config section.
    pub fn model_config(path: impl Into<PathBuf>) -> Self {
        Self::new(path.into(), SectionType::ModelConfig)
    }

    /// Create a SentencePiece model file configuration.
    ///
    /// Convenience method for creating a SentencePiece section.
    pub fn sentencepiece(path: impl Into<PathBuf>) -> Self {
        Self::new(path.into(), SectionType::SentencePiece)
    }

    /// Create a raw file configuration with custom content type.
    ///
    /// Convenience method for creating a raw section.
    pub fn raw(path: impl Into<PathBuf>, content_type: &str) -> Self {
        Self::new(
            path.into(),
            SectionType::Raw {
                content_type: content_type.to_string(),
            },
        )
    }
}

/// Configuration for ONNX compilation.
///
/// This structure controls various aspects of the compilation pipeline:
/// - Weight storage strategy (inline vs external file)
/// - Graph partitioning for large models
/// - ISA optimization passes (decomposition)
/// - Memory budget constraints
/// - Embedded sections for single-file distribution
///
/// # Examples
///
/// ```
/// use hologram_ai_onnx::core::OnnxConfig;
///
/// // Default configuration
/// let config = OnnxConfig::default();
/// assert_eq!(config.weight_threshold, 4096);
/// assert!(!config.enable_partitioning);
///
/// // Custom configuration for large models
/// let config = OnnxConfig {
///     weight_threshold: 8192,
///     enable_partitioning: true,
///     partition_size: 1000,
///     decompose_conv2d: true,
///     decompose_pooling: true,
///     pack_weights: true,
///     memory_budget: Some(16 * 1024), // 16 GB
///     enable_resize_upscaling: true,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    /// Threshold (in bytes) for storing weights externally.
    ///
    /// Weights larger than this threshold are stored in the external
    /// `.weights` file. Smaller weights are inlined in the `.holo` file.
    ///
    /// Default: 4096 bytes (4 KB)
    ///
    /// # Rationale
    ///
    /// Small weights benefit from being inlined (fewer file I/O operations),
    /// while large weights should be external to avoid bloating the .holo file
    /// and to enable memory-mapped loading at runtime.
    pub weight_threshold: usize,

    /// Enable graph partitioning for large models.
    ///
    /// When enabled, models with more than `partition_size` nodes are
    /// automatically partitioned into smaller chunks to avoid OOM errors
    /// during compilation.
    ///
    /// Default: false
    ///
    /// # When to Enable
    ///
    /// Enable this for:
    /// - Models with >500 nodes
    /// - Systems with limited RAM (<16 GB)
    /// - Large models like Stable Diffusion UNet (3052 nodes)
    pub enable_partitioning: bool,

    /// Number of nodes per partition.
    ///
    /// Only used when `enable_partitioning` is true. Models with more
    /// than this many nodes are split into chunks of this size.
    ///
    /// Default: 500
    ///
    /// # Memory Impact
    ///
    /// Smaller partition sizes reduce peak memory usage but increase
    /// compilation time due to partition merging overhead.
    ///
    /// Recommended values:
    /// - 500: Default, works well for most models
    /// - 300: For systems with very limited RAM (<8 GB)
    /// - 1000: For systems with plenty of RAM (>32 GB)
    pub partition_size: usize,

    /// Enable Conv2D → Im2col+GEMM decomposition.
    ///
    /// When enabled, Conv2D operations are decomposed into Im2col
    /// (image-to-column transformation) followed by GEMM (matrix multiplication).
    ///
    /// Default: true
    ///
    /// # ISA Optimization
    ///
    /// This decomposition enables hologram's LOOP instructions and SIMD
    /// vectorization for Conv2D operations, providing significant speedup.
    ///
    /// **CRITICAL**: Should always be true for maximum performance.
    pub decompose_conv2d: bool,

    /// Enable pooling operation decomposition.
    ///
    /// When enabled, pooling operations (MaxPool, AveragePool) are
    /// decomposed into primitive operations that leverage hologram's ISA.
    ///
    /// Default: true
    ///
    /// # ISA Optimization
    ///
    /// Enables PhiCoordinate addressing and LOOP instructions for pooling.
    pub decompose_pooling: bool,

    /// Enable serialization of packed weights for faster runtime execution.
    ///
    /// When enabled, the compiler pre-packs Conv2D/MatMul weights into
    /// layouts that minimize runtime overhead.
    ///
    /// Default: true
    pub pack_weights: bool,

    /// Memory budget in megabytes (MB).
    ///
    /// If set, compilation will fail with [`OnnxError::MemoryBudgetExceeded`]
    /// if peak memory usage exceeds this limit.
    ///
    /// Default: None (unlimited)
    ///
    /// # Usage
    ///
    /// Set this to prevent OOM kills on systems with limited RAM:
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxConfig;
    ///
    /// // Limit compilation to 8 GB
    /// let config = OnnxConfig {
    ///     memory_budget: Some(8 * 1024),
    ///     ..Default::default()
    /// };
    /// ```
    pub memory_budget: Option<usize>,

    /// Enable Resize upscaling operations.
    ///
    /// When true (default), Resize ops extract scale factors from ONNX
    /// constants and upscale tensors appropriately (e.g., 64x64 → 512x512).
    ///
    /// When false, Resize ops pass through without upscaling, saving memory
    /// at the cost of lower resolution outputs.
    ///
    /// Default: true
    ///
    /// # Memory Impact
    ///
    /// Full upscaling (512x512) requires ~8GB RAM for VAE decoders.
    /// Disable this option for systems with limited memory (<8GB).
    pub enable_resize_upscaling: bool,

    /// Files to embed as sections in the bundle.
    ///
    /// When this vector is non-empty, the compiler produces a V2 bundle
    /// with the specified files embedded as sections. This enables
    /// single-file model distribution including vocabulary, tokenizer
    /// config, and other auxiliary data.
    ///
    /// Default: empty (no embedded files, produces V1 bundle)
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::{OnnxConfig, EmbeddedFileConfig};
    ///
    /// let config = OnnxConfig {
    ///     embedded_files: vec![
    ///         EmbeddedFileConfig::vocabulary("vocab.txt"),
    ///         EmbeddedFileConfig::tokenizer_config("tokenizer_config.json"),
    ///     ],
    ///     ..Default::default()
    /// };
    /// ```
    pub embedded_files: Vec<EmbeddedFileConfig>,

    /// Enable parallel execution groups and activation fusion.
    ///
    /// When true (default), the compiler detects attention patterns and
    /// activation chains to enable:
    /// - Parallel Q/K/V projection execution (2-3x speedup)
    /// - Activation chain fusion with view composition (3x speedup)
    ///
    /// When false, uses standard sequential translation without pattern detection.
    ///
    /// Default: true
    ///
    /// # Performance Impact
    ///
    /// Provides significant speedup for transformer models with multi-head attention.
    /// No impact on non-transformer models (pattern detection is lightweight).
    pub enable_parallel_execution: bool,
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            weight_threshold: 4096, // 4 KB
            enable_partitioning: false,
            partition_size: 500,
            decompose_conv2d: true,
            decompose_pooling: true,
            pack_weights: true,
            memory_budget: None,
            enable_resize_upscaling: true,
            embedded_files: Vec::new(),
            enable_parallel_execution: true,
        }
    }
}

impl OnnxConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to [`OnnxConfig::default()`].
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxConfig;
    ///
    /// let config = OnnxConfig::new();
    /// assert_eq!(config.weight_threshold, 4096);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for large models.
    ///
    /// This configuration enables:
    /// - Graph partitioning (500 nodes per partition)
    /// - Memory budget (8 GB)
    /// - All ISA optimizations
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxConfig;
    ///
    /// let config = OnnxConfig::for_large_model();
    /// assert!(config.enable_partitioning);
    /// assert_eq!(config.memory_budget, Some(8 * 1024));
    /// ```
    pub fn for_large_model() -> Self {
        Self {
            weight_threshold: 4096,
            enable_partitioning: true,
            partition_size: 500,
            decompose_conv2d: true,
            decompose_pooling: true,
            pack_weights: true,
            memory_budget: Some(8 * 1024), // 8 GB
            enable_resize_upscaling: true,
            embedded_files: Vec::new(),
            enable_parallel_execution: true,
        }
    }

    /// Create a configuration optimized for small models.
    ///
    /// This configuration disables:
    /// - Graph partitioning (not needed for small models)
    /// - Memory budget (small models don't need it)
    ///
    /// Enables:
    /// - All ISA optimizations
    /// - Higher weight threshold (16 KB) for more inlining
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxConfig;
    ///
    /// let config = OnnxConfig::for_small_model();
    /// assert!(!config.enable_partitioning);
    /// assert_eq!(config.weight_threshold, 16384);
    /// ```
    pub fn for_small_model() -> Self {
        Self {
            weight_threshold: 16384, // 16 KB (more inlining)
            enable_partitioning: false,
            partition_size: 500,
            decompose_conv2d: true,
            decompose_pooling: true,
            pack_weights: true,
            memory_budget: None,
            enable_resize_upscaling: true,
            embedded_files: Vec::new(),
            enable_parallel_execution: false, // Not needed for small models
        }
    }

    /// Validate configuration settings.
    ///
    /// Checks that:
    /// - Weight threshold is reasonable (>0, <1 GB)
    /// - Partition size is reasonable (>10, <10000)
    /// - Memory budget is reasonable if set (>100 MB)
    ///
    /// # Errors
    ///
    /// Returns error message if validation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::OnnxConfig;
    ///
    /// let config = OnnxConfig::default();
    /// assert!(config.validate().is_ok());
    ///
    /// let bad_config = OnnxConfig {
    ///     weight_threshold: 0, // Invalid!
    ///     ..Default::default()
    /// };
    /// assert!(bad_config.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.weight_threshold == 0 {
            return Err("weight_threshold must be greater than 0".into());
        }

        if self.weight_threshold > 1024 * 1024 * 1024 {
            return Err("weight_threshold too large (>1 GB)".into());
        }

        if self.partition_size < 10 {
            return Err("partition_size too small (must be >= 10)".into());
        }

        if self.partition_size > 10000 {
            return Err("partition_size too large (must be <= 10000)".into());
        }

        if let Some(budget) = self.memory_budget
            && budget < 100
        {
            return Err("memory_budget too small (must be >= 100 MB)".into());
        }

        Ok(())
    }

    /// Add an embedded file to the configuration.
    ///
    /// This is a builder method that returns `self` for chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use hologram_ai_onnx::core::{OnnxConfig, EmbeddedFileConfig};
    ///
    /// let config = OnnxConfig::new()
    ///     .with_embedded_file(EmbeddedFileConfig::vocabulary("vocab.txt"))
    ///     .with_embedded_file(EmbeddedFileConfig::tokenizer_config("tokenizer_config.json"));
    ///
    /// assert_eq!(config.embedded_files.len(), 2);
    /// ```
    pub fn with_embedded_file(mut self, file: EmbeddedFileConfig) -> Self {
        self.embedded_files.push(file);
        self
    }

    /// Check if the configuration will produce a V2 bundle.
    ///
    /// Returns `true` if there are any embedded files, which triggers
    /// V2 bundle format output.
    pub fn produces_v2_bundle(&self) -> bool {
        !self.embedded_files.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = OnnxConfig::default();
        assert_eq!(config.weight_threshold, 4096);
        assert!(!config.enable_partitioning);
        assert_eq!(config.partition_size, 500);
        assert!(config.decompose_conv2d);
        assert!(config.decompose_pooling);
        assert!(config.pack_weights);
        assert_eq!(config.memory_budget, None);
        assert!(config.embedded_files.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_new_config() {
        let config = OnnxConfig::new();
        assert_eq!(config.weight_threshold, 4096);
    }

    #[test]
    fn test_large_model_config() {
        let config = OnnxConfig::for_large_model();
        assert!(config.enable_partitioning);
        assert_eq!(config.partition_size, 500);
        assert_eq!(config.memory_budget, Some(8 * 1024));
        assert!(config.embedded_files.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_model_config() {
        let config = OnnxConfig::for_small_model();
        assert!(!config.enable_partitioning);
        assert_eq!(config.weight_threshold, 16384);
        assert_eq!(config.memory_budget, None);
        assert!(config.embedded_files.is_empty());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        // Valid config
        let config = OnnxConfig::default();
        assert!(config.validate().is_ok());

        // Invalid: weight_threshold = 0
        let config = OnnxConfig {
            weight_threshold: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: weight_threshold too large
        let config = OnnxConfig {
            weight_threshold: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: partition_size too small
        let config = OnnxConfig {
            partition_size: 5,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: partition_size too large
        let config = OnnxConfig {
            partition_size: 20000,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Invalid: memory_budget too small
        let config = OnnxConfig {
            memory_budget: Some(50),
            ..Default::default()
        };
        assert!(config.validate().is_err());

        // Valid: custom config
        let config = OnnxConfig {
            weight_threshold: 8192,
            enable_partitioning: true,
            partition_size: 1000,
            decompose_conv2d: true,
            decompose_pooling: false,
            pack_weights: true,
            memory_budget: Some(16 * 1024),
            enable_resize_upscaling: false,
            embedded_files: Vec::new(),
            enable_parallel_execution: true,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_clone() {
        let config1 = OnnxConfig::for_large_model();
        let config2 = config1.clone();
        assert_eq!(config1.weight_threshold, config2.weight_threshold);
        assert_eq!(config1.enable_partitioning, config2.enable_partitioning);
    }

    #[test]
    fn test_section_type_default_ids() {
        assert_eq!(SectionType::Vocabulary.default_section_id(), "vocabulary");
        assert_eq!(
            SectionType::VocabularyJson.default_section_id(),
            "vocabulary"
        );
        assert_eq!(
            SectionType::TokenizerConfig.default_section_id(),
            "tokenizer_config"
        );
        assert_eq!(
            SectionType::ModelConfig.default_section_id(),
            "model_config"
        );
        assert_eq!(
            SectionType::SpecialTokensMap.default_section_id(),
            "special_tokens"
        );
        assert_eq!(
            SectionType::PreprocessorConfig.default_section_id(),
            "preprocessor_config"
        );
        assert_eq!(
            SectionType::SentencePiece.default_section_id(),
            "sentencepiece"
        );
        assert_eq!(
            SectionType::GenerationConfig.default_section_id(),
            "generation_config"
        );
        assert_eq!(
            SectionType::Raw {
                content_type: "text/plain".to_string()
            }
            .default_section_id(),
            "raw"
        );
    }

    #[test]
    fn test_embedded_file_config() {
        let config = EmbeddedFileConfig::new(PathBuf::from("vocab.txt"), SectionType::Vocabulary);
        assert_eq!(config.section_id(), "vocabulary");
        assert!(config.custom_id.is_none());

        let config = EmbeddedFileConfig::with_id(
            PathBuf::from("custom.bin"),
            SectionType::Raw {
                content_type: "application/octet-stream".to_string(),
            },
            "my_section",
        );
        assert_eq!(config.section_id(), "my_section");
    }

    #[test]
    fn test_embedded_file_convenience_methods() {
        let vocab = EmbeddedFileConfig::vocabulary("vocab.txt");
        assert_eq!(vocab.section_type, SectionType::Vocabulary);

        let vocab_json = EmbeddedFileConfig::vocabulary_json("vocab.json");
        assert_eq!(vocab_json.section_type, SectionType::VocabularyJson);

        let tok_config = EmbeddedFileConfig::tokenizer_config("tokenizer_config.json");
        assert_eq!(tok_config.section_type, SectionType::TokenizerConfig);

        let model_config = EmbeddedFileConfig::model_config("config.json");
        assert_eq!(model_config.section_type, SectionType::ModelConfig);

        let sp = EmbeddedFileConfig::sentencepiece("model.model");
        assert_eq!(sp.section_type, SectionType::SentencePiece);

        let raw = EmbeddedFileConfig::raw("data.bin", "application/octet-stream");
        assert!(matches!(raw.section_type, SectionType::Raw { .. }));
    }

    #[test]
    fn test_with_embedded_file() {
        let config = OnnxConfig::new()
            .with_embedded_file(EmbeddedFileConfig::vocabulary("vocab.txt"))
            .with_embedded_file(EmbeddedFileConfig::tokenizer_config(
                "tokenizer_config.json",
            ));

        assert_eq!(config.embedded_files.len(), 2);
        assert!(config.produces_v2_bundle());
    }

    #[test]
    fn test_produces_v2_bundle() {
        let config = OnnxConfig::default();
        assert!(!config.produces_v2_bundle());

        let config =
            OnnxConfig::new().with_embedded_file(EmbeddedFileConfig::vocabulary("vocab.txt"));
        assert!(config.produces_v2_bundle());
    }
}

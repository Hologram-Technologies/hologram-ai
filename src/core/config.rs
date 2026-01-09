//! Compilation configuration for ONNX models.
//!
//! This module defines configuration options that control the ONNX compilation
//! pipeline, including memory management, graph partitioning, and ISA optimizations.

/// Configuration for ONNX compilation.
///
/// This structure controls various aspects of the compilation pipeline:
/// - Weight storage strategy (inline vs external file)
/// - Graph partitioning for large models
/// - ISA optimization passes (decomposition)
/// - Memory budget constraints
///
/// # Examples
///
/// ```
/// use hologram_onnx::core::OnnxConfig;
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
    /// use hologram_onnx::core::OnnxConfig;
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
    /// use hologram_onnx::core::OnnxConfig;
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
    /// use hologram_onnx::core::OnnxConfig;
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
    /// use hologram_onnx::core::OnnxConfig;
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
    /// use hologram_onnx::core::OnnxConfig;
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
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_small_model_config() {
        let config = OnnxConfig::for_small_model();
        assert!(!config.enable_partitioning);
        assert_eq!(config.weight_threshold, 16384);
        assert_eq!(config.memory_budget, None);
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
}

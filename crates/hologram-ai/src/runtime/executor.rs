//! Model executor for running compiled .holo models.
//!
//! This module provides a high-level executor that wraps the hologram Backend API
//! and provides tensor I/O interfaces.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::loader::{load_holo_auto, load_holo_auto_with_inputs, load_with_external_weights};
use super::metrics::PerformanceMetrics;
use super::tensors::Tensor;

use hologram::backend::{Backend, BackendPlan};

/// Type alias for layer cache used by CallLayer instruction support.
type LayerCache = Arc<std::sync::Mutex<HashMap<u64, Arc<BackendPlan>>>>;

/// Optimization capabilities detected in a BackendPlan.
///
/// This structure tracks which optimizations are available based on
/// the operations present in the compiled plan.
#[derive(Debug, Clone, Default)]
struct OptimizationCapabilities {
    /// Plan contains SIMD-accelerated activation kernels
    has_simd_activations: bool,
    /// Plan contains fused/composed view kernels
    has_composed_views: bool,
    /// Plan contains operations with parallel hints
    has_parallel_ops: bool,
    /// Plan contains large embedding constants (>1MB)
    has_large_embeddings: bool,
}

/// Analysis of parallel execution opportunities in a BackendPlan.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used for analysis/debugging
struct ParallelismAnalysis {
    /// Groups of operations that can execute in parallel
    parallel_groups: Vec<Vec<usize>>,
    /// Total number of operations that could be parallelized
    total_parallelizable_ops: usize,
    /// Total number of operations that must run sequentially
    total_sequential_ops: usize,
}

/// Buffer requirements from BackendPlan metadata.
pub struct BufferRequirements {
    /// Number of input buffers
    pub num_inputs: usize,
    /// Number of output buffers
    pub num_outputs: usize,
    /// Size of each input buffer in bytes
    pub input_sizes: Vec<usize>,
    /// Shape of each input buffer (0s indicate dynamic dims)
    pub input_shapes: Vec<[usize; 4]>,
    /// Size of each output buffer in bytes
    pub output_sizes: Vec<usize>,
    /// Shape of each output buffer (0s indicate dynamic dims)
    pub output_shapes: Vec<[usize; 4]>,
}

/// Optimization features detected in the compiled model.
///
/// Following Integration Guide Section 7 (Optimization Features).
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// SIMD-accelerated activation kernels detected (Guide Section 7.3)
    pub has_simd_activations: bool,
    /// Fused operation chains detected (Guide Section 7.1)
    pub has_epilogue_fusion: bool,
    /// Parallel execution groups detected (Guide Section 7.4)
    pub has_parallel_groups: bool,
    /// Number of parallel execution groups
    pub parallel_group_count: usize,
    /// Total parallelizable operations
    pub parallelizable_ops: usize,
    /// Large embeddings for cache pinning (>1MB)
    pub has_embedding_cache: bool,
    /// SIMD level available on this CPU
    pub simd_level: String,
    /// Dynamic shape support detected
    pub dynamic_shapes: bool,
}

/// Model executor for running compiled .holo models.
///
/// Wraps hologram's Backend trait and provides a high-level tensor I/O interface.
pub struct ModelExecutor {
    /// The compiled backend plan
    plan: Arc<BackendPlan>,
    /// Backend for execution
    backend: Box<dyn Backend>,
    /// Optional input order override (uses LayerHeader input ordering when available)
    input_order: Option<Vec<String>>,
    /// Detected optimization capabilities
    optimization_caps: OptimizationCapabilities,
    /// Optional performance metrics tracker
    metrics: Option<PerformanceMetrics>,
    /// Optional layer cache for CallLayer instruction support
    layer_cache: Option<LayerCache>,
}

impl ModelExecutor {
    /// Create a new executor from a .holo file path.
    ///
    /// This loads the .holo file and creates a ready-to-execute model executor.
    ///
    /// # Arguments
    /// * `path` - Path to compiled .holo file
    ///
    /// # Returns
    /// ModelExecutor ready for execution
    pub fn from_holo_file(path: &Path) -> Result<Self> {
        let (plan, backend, input_order) = load_holo_auto_with_inputs(path)?;
        Ok(Self {
            plan: Arc::new(plan),
            backend,
            input_order,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            layer_cache: None,
        })
    }

    /// Create a new executor from .holo and .weights files.
    ///
    /// This loads the .holo file and creates an executor that uses memory-mapped
    /// access to the external weights file. This enables lazy loading of large
    /// weights (GB-sized) without loading them all into memory.
    ///
    /// # Arguments
    /// * `holo_path` - Path to compiled .holo file
    /// * `weights_path` - Path to external .weights file (will be memory-mapped)
    ///
    /// # Returns
    /// ModelExecutor ready for execution with external weights
    pub fn from_holo_with_weights(holo_path: &Path, weights_path: &Path) -> Result<Self> {
        let (plan, backend) = load_with_external_weights(holo_path, weights_path)?;

        Ok(Self {
            plan: Arc::new(plan),
            backend,
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            layer_cache: None,
        })
    }

    /// Create a new executor from an existing BackendPlan and backend.
    ///
    /// This is useful when loading models from a pipeline bundle.
    ///
    /// # Arguments
    /// * `plan` - Pre-compiled BackendPlan
    /// * `backend` - Backend for execution
    ///
    /// # Returns
    /// ModelExecutor ready for execution
    pub fn from_plan(plan: BackendPlan, backend: Box<dyn Backend>) -> Self {
        Self {
            plan: Arc::new(plan),
            backend,
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            layer_cache: None,
        }
    }

    /// Create a new executor with an explicit input order override.
    ///
    /// This is useful when loading models from a pipeline bundle that embed
    /// a LayerHeader describing the expected input ordering.
    pub fn from_plan_with_inputs(
        plan: BackendPlan,
        backend: Box<dyn Backend>,
        input_order: Vec<String>,
    ) -> Self {
        Self {
            plan: Arc::new(plan),
            backend,
            input_order: Some(input_order),
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            layer_cache: None,
        }
    }

    /// Set the input order for named tensor execution.
    ///
    /// By default, inputs are sorted alphabetically. Use this to override
    /// with the model's expected input order.
    pub fn set_input_order(&mut self, order: Vec<String>) {
        self.input_order = Some(order);
    }

    /// Set layer cache for CallLayer instruction support.
    ///
    /// When models contain `CallLayer` instructions (for layer composition),
    /// this cache provides the sub-layer plans that will be executed.
    ///
    /// # Arguments
    /// * `cache` - Shared cache mapping layer IDs to their compiled plans
    pub fn with_layer_cache(
        mut self,
        cache: Arc<std::sync::Mutex<std::collections::HashMap<u64, Arc<BackendPlan>>>>,
    ) -> Self {
        self.layer_cache = Some(cache);
        self
    }

    /// Create a new executor from a .holo file with optimizations enabled.
    ///
    /// This constructor:
    /// 1. Loads the .holo file
    /// 2. Detects available optimizations (SIMD, parallel, cache)
    /// 3. Initializes performance metrics tracking
    ///
    /// # Arguments
    /// * `path` - Path to compiled .holo file
    ///
    /// # Returns
    /// ModelExecutor with optimizations enabled and metrics tracking
    pub fn from_holo_file_optimized(path: &Path) -> Result<Self> {
        let (plan, backend) = load_holo_auto(path)?;
        let plan = Arc::new(plan);

        // Detect optimization capabilities
        let optimization_caps = Self::detect_optimizations(&plan);

        Ok(Self {
            plan,
            backend,
            input_order: None,
            optimization_caps,
            metrics: Some(PerformanceMetrics::new()),
            layer_cache: None,
        })
    }

    /// Access performance metrics (if enabled).
    pub fn metrics(&self) -> Option<&PerformanceMetrics> {
        self.metrics.as_ref()
    }

    /// Access mutable performance metrics (if enabled).
    pub fn metrics_mut(&mut self) -> Option<&mut PerformanceMetrics> {
        self.metrics.as_mut()
    }

    /// Access the compiled backend plan.
    pub fn plan(&self) -> &BackendPlan {
        &self.plan
    }

    /// Get optimization report for the compiled model.
    ///
    /// Returns detailed information about optimizations detected in the
    /// BackendPlan. This follows Integration Guide Section 7 (Optimization Features).
    pub fn optimization_report(&self) -> OptimizationReport {
        let caps = &self.optimization_caps;
        let parallelism = Self::analyze_parallelism(&self.plan);

        OptimizationReport {
            has_simd_activations: caps.has_simd_activations,
            has_epilogue_fusion: caps.has_composed_views,
            has_parallel_groups: caps.has_parallel_ops,
            parallel_group_count: parallelism.parallel_groups.len(),
            parallelizable_ops: parallelism.total_parallelizable_ops,
            has_embedding_cache: caps.has_large_embeddings,
            simd_level: "Auto".to_string(),
            dynamic_shapes: self.has_dynamic_shapes(),
        }
    }

    /// Check if the plan has dynamic shapes.
    fn has_dynamic_shapes(&self) -> bool {
        // Note: workspace_layout not available in new API
        false
    }

    /// Detect optimization capabilities from a BackendPlan.
    fn detect_optimizations(plan: &BackendPlan) -> OptimizationCapabilities {
        let mut caps = OptimizationCapabilities::default();

        // Check for large embeddings in constants (>1MB)
        if plan.constants.len() > 1_000_000 {
            caps.has_large_embeddings = true;
        }

        // Note: Instruction-level optimization detection not available in new API
        caps
    }

    /// Analyze parallelism opportunities in the plan.
    fn analyze_parallelism(_plan: &BackendPlan) -> ParallelismAnalysis {
        // Note: parallel_group not available in new API
        ParallelismAnalysis {
            parallel_groups: vec![],
            total_parallelizable_ops: 0,
            total_sequential_ops: 0,
        }
    }

    /// Get buffer requirements from the plan.
    ///
    /// Note: In the new API, we infer buffer requirements from the buffers array.
    pub fn get_buffer_requirements(&self) -> BufferRequirements {
        use hologram::holo::types::BufferType;

        let mut num_inputs = 0;
        let mut num_outputs = 0;
        let mut input_sizes = vec![];
        let mut output_sizes = vec![];

        for buf in &self.plan.buffers {
            match buf.buffer_type {
                BufferType::Input => {
                    num_inputs += 1;
                    input_sizes.push(buf.size);
                }
                BufferType::Output => {
                    num_outputs += 1;
                    output_sizes.push(buf.size);
                }
                _ => {}
            }
        }

        BufferRequirements {
            num_inputs,
            num_outputs,
            input_sizes,
            input_shapes: vec![[0, 0, 0, 0]; num_inputs], // Shapes not available
            output_sizes,
            output_shapes: vec![[0, 0, 0, 0]; num_outputs], // Shapes not available
        }
    }

    /// Check if the plan requires layer executor (contains CallLayer instructions).
    ///
    /// Returns true only if:
    /// 1. Plan has CallLayer instructions, AND
    /// 2. Plan has dependencies listed
    ///
    /// Note: Some ONNX models compiled with older hologram versions have spurious
    /// CallLayer instructions without dependencies. We skip layer loading in those cases.
    pub fn requires_layer_executor(&self) -> bool {
        use hologram::holo::IsaInstruction;

        let has_call_layer = self
            .plan
            .instructions
            .iter()
            .any(|instr| matches!(instr, IsaInstruction::CallLayer { .. }));

        // Only require layer executor if we have both CallLayer AND dependencies
        has_call_layer && !self.plan.dependencies.is_empty()
    }

    /// Load layer cache from plan dependencies.
    ///
    /// This method inspects the plan's dependencies and loads all referenced
    /// sub-layers, either from embedded data or external files.
    ///
    /// # Arguments
    /// * `base_path` - Base directory for resolving external layer files
    ///
    /// # Returns
    /// Arc-wrapped Mutex-protected HashMap mapping layer IDs to BackendPlans
    pub fn load_layer_cache(&self, base_path: &Path) -> Result<LayerCache> {
        use hologram::holo::{IsaInstruction, LayerLocation};
        use std::collections::HashMap;

        let mut cache = HashMap::new();

        // If no CallLayer instructions, return empty cache
        if !self.requires_layer_executor() {
            return Ok(Arc::new(std::sync::Mutex::new(cache)));
        }

        // Collect all layer IDs from CallLayer instructions
        let mut layer_ids = std::collections::HashSet::new();
        for instr in &self.plan.instructions {
            if let IsaInstruction::CallLayer { layer_id, .. } = instr {
                layer_ids.insert(*layer_id);
            }
        }

        // Load each dependency
        for layer_ref in &self.plan.dependencies {
            // Convert full hash to u64 (first 8 bytes)
            let layer_id = u64::from_le_bytes([
                layer_ref.layer_id[0],
                layer_ref.layer_id[1],
                layer_ref.layer_id[2],
                layer_ref.layer_id[3],
                layer_ref.layer_id[4],
                layer_ref.layer_id[5],
                layer_ref.layer_id[6],
                layer_ref.layer_id[7],
            ]);

            // Only load if actually referenced by CallLayer
            if !layer_ids.contains(&layer_id) {
                continue;
            }

            let sublayer_plan = match &layer_ref.location {
                LayerLocation::Embedded { offset, size } => {
                    // Load from embedded data in the same archive
                    // Note: This requires access to the raw archive bytes, which isn't
                    // exposed in the current API. For now, we'll return an error.
                    anyhow::bail!(
                        "Embedded layers not yet supported. Layer ID: {:016x}, offset: {}, size: {}",
                        layer_id,
                        offset,
                        size
                    );
                }
                LayerLocation::External(path_str) => {
                    // Load from external file
                    let layer_path = if Path::new(path_str).is_absolute() {
                        PathBuf::from(path_str)
                    } else {
                        base_path.join(path_str)
                    };

                    if !layer_path.exists() {
                        anyhow::bail!(
                            "External layer file not found: {} (resolved from base: {})",
                            layer_path.display(),
                            base_path.display()
                        );
                    }

                    let (plan, _backend) = crate::runtime::loader::load_holo_auto(&layer_path)
                        .with_context(|| {
                            format!("Failed to load sub-layer from {}", layer_path.display())
                        })?;

                    Arc::new(plan)
                }
                LayerLocation::Registry {
                    registry_url,
                    version,
                } => {
                    anyhow::bail!(
                        "Registry layers not yet supported. URL: {}, version: {}",
                        registry_url,
                        version
                    );
                }
            };

            cache.insert(layer_id, sublayer_plan);
        }

        // Verify all required layers were loaded
        for required_id in &layer_ids {
            if !cache.contains_key(required_id) {
                anyhow::bail!(
                    "Required layer {:016x} not found in dependencies. Available: {:?}",
                    required_id,
                    self.plan
                        .dependencies
                        .iter()
                        .map(|d| {
                            let id = u64::from_le_bytes([
                                d.layer_id[0],
                                d.layer_id[1],
                                d.layer_id[2],
                                d.layer_id[3],
                                d.layer_id[4],
                                d.layer_id[5],
                                d.layer_id[6],
                                d.layer_id[7],
                            ]);
                            format!("{:016x}", id)
                        })
                        .collect::<Vec<_>>()
                );
            }
        }

        Ok(Arc::new(std::sync::Mutex::new(cache)))
    }

    /// Debug: dump all buffer info to tracing.
    pub fn debug_dump_buffers(&self) {
        use hologram::holo::types::BufferType;

        let mut constant_offset = 0usize;
        tracing::info!(
            "Buffer dump ({} total, constants blob: {} bytes):",
            self.plan.buffers.len(),
            self.plan.constants.len()
        );

        for (idx, buf) in self.plan.buffers.iter().enumerate() {
            let type_str = match buf.buffer_type {
                BufferType::Input => "Input",
                BufferType::Output => "Output",
                BufferType::Workspace => "Workspace",
                BufferType::Constant => "Constant",
            };

            if buf.buffer_type == BufferType::Constant {
                let end = constant_offset + buf.size;
                let avail = self.plan.constants.len().saturating_sub(constant_offset);
                tracing::info!(
                    "  [{:4}] {} size={} (const offset {}..{}, avail={})",
                    idx,
                    type_str,
                    buf.size,
                    constant_offset,
                    end,
                    avail
                );
                constant_offset += buf.size;
            } else {
                tracing::info!("  [{:4}] {} size={}", idx, type_str, buf.size);
            }
        }
        tracing::info!("  Total constant bytes expected: {}", constant_offset);
    }

    /// Execute the model with tensor inputs and outputs.
    ///
    /// This is the main execution entry point. It:
    /// 1. Converts input tensors to byte buffers
    /// 2. Allocates output buffers
    /// 3. Calls the backend's execute_plan method
    /// 4. Converts output buffers back to tensors
    ///
    /// # Arguments
    /// * `inputs` - HashMap of input name to Tensor
    ///
    /// # Returns
    /// HashMap of output name to Tensor
    pub fn execute(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let execution_start = std::time::Instant::now();

        // Get buffer requirements
        let requirements = self.get_buffer_requirements();

        tracing::debug!(
            plan_inputs = requirements.num_inputs,
            plan_outputs = requirements.num_outputs,
            input_sizes = ?requirements.input_sizes,
            "Buffer requirements"
        );

        // Resolve input order
        let input_names = self.resolve_input_order(&inputs, &requirements)?;

        // Convert inputs to byte buffers
        let input_bytes: Vec<Vec<u8>> = input_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                let tensor = inputs.get(name).unwrap();
                let bytes = tensor.to_bytes();
                // Debug: show first value as f32 to verify we're sending the right data
                let first_val = if bytes.len() >= 4 {
                    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    0.0
                };
                tracing::debug!(
                    "Input[{}] '{}': {} elements, dtype={:?}, {} bytes, first_f32={}",
                    idx,
                    name,
                    tensor.numel(),
                    tensor.dtype,
                    bytes.len(),
                    first_val
                );
                bytes
            })
            .collect();

        let input_refs: Vec<&[u8]> = input_bytes.iter().map(|v| v.as_slice()).collect();

        // Allocate output buffers based on plan requirements
        let mut output_bytes: Vec<Vec<u8>> = requirements
            .output_sizes
            .iter()
            .map(|&size| vec![0u8; size])
            .collect();

        let mut output_refs: Vec<&mut [u8]> =
            output_bytes.iter_mut().map(|v| v.as_mut_slice()).collect();

        // Execute the plan (with layer executor if cache is available)
        if let Some(ref layer_cache) = self.layer_cache {
            // Clone Arc for use in closure
            let cache = Arc::clone(layer_cache);
            let backend_clone = &self.backend;

            // Define layer executor closure
            let mut executor = |layer_id: u64, inputs: &[&[u8]], outputs: &mut [&mut [u8]]| {
                let cache_guard = cache.lock().unwrap();
                let sublayer = cache_guard.get(&layer_id).ok_or_else(|| {
                    hologram::backend::BackendError::LayerLoadError {
                        layer_id: format!("{layer_id:016x}"),
                        reason: "not found in cache".into(),
                    }
                })?;
                backend_clone.execute_plan(sublayer, inputs, outputs)
            };

            self.backend
                .execute_plan_with_layers(&self.plan, &input_refs, &mut output_refs, &mut executor)
                .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))?;
        } else {
            self.backend
                .execute_plan(&self.plan, &input_refs, &mut output_refs)
                .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))?;
        }

        // Convert outputs to tensors
        let mut outputs = HashMap::new();
        for (idx, data) in output_bytes.into_iter().enumerate() {
            let shape = requirements.output_shapes[idx];
            let mut shape_vec: Vec<usize> = shape.iter().copied().filter(|&d| d > 0).collect();

            // If shape information is not available (all zeros), infer flat shape from byte size.
            // This happens because hologram's BufferMetadata doesn't store shape information,
            // only size/alignment. We assume f32 dtype (4 bytes per element).
            if shape_vec.is_empty() {
                let num_elements = data.len() / 4; // 4 bytes per f32
                shape_vec = vec![num_elements];
            }

            let tensor = Tensor::from_bytes(&data, shape_vec)?;
            outputs.insert(format!("output_{}", idx), tensor);
        }

        // Update metrics if enabled
        if let Some(metrics) = &mut self.metrics {
            metrics.set_execution_time(execution_start.elapsed());
        }

        Ok(outputs)
    }

    /// Execute with raw byte buffers (lower-level API).
    ///
    /// This bypasses tensor conversion for performance-critical paths.
    pub fn execute_raw(&self, inputs: &[&[u8]], outputs: &mut [&mut [u8]]) -> Result<()> {
        if let Some(ref layer_cache) = self.layer_cache {
            let cache = Arc::clone(layer_cache);
            let backend_clone = &self.backend;

            let mut executor = |layer_id: u64, inputs: &[&[u8]], outputs: &mut [&mut [u8]]| {
                let cache_guard = cache.lock().unwrap();
                let sublayer = cache_guard.get(&layer_id).ok_or_else(|| {
                    hologram::backend::BackendError::LayerLoadError {
                        layer_id: format!("{layer_id:016x}"),
                        reason: "not found in cache".into(),
                    }
                })?;
                backend_clone.execute_plan(sublayer, inputs, outputs)
            };

            self.backend
                .execute_plan_with_layers(&self.plan, inputs, outputs, &mut executor)
                .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))
        } else {
            self.backend
                .execute_plan(&self.plan, inputs, outputs)
                .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))
        }
    }

    /// Resolve input order based on configuration or requirements.
    ///
    /// This uses size-based matching with semantic awareness:
    /// 1. For each expected buffer size, find matching input tensors
    /// 2. When multiple inputs have the same size, use semantic ordering:
    ///    - input_ids before attention_mask (transformer models)
    ///    - Examine tensor content: integer-like values (token IDs) before float masks
    /// 3. This handles cases where ONNX input order doesn't match lexicographic order
    fn resolve_input_order(
        &self,
        inputs: &HashMap<String, Tensor>,
        requirements: &BufferRequirements,
    ) -> Result<Vec<String>> {
        // Use explicit input order if available
        if let Some(ref order) = self.input_order {
            return Ok(order.clone());
        }

        // Validate count matches
        if inputs.len() != requirements.num_inputs {
            anyhow::bail!(
                "Input count mismatch: got {} inputs, plan expects {}",
                inputs.len(),
                requirements.num_inputs
            );
        }

        // Try size-based matching: match inputs to expected buffer sizes
        let mut result = Vec::with_capacity(requirements.num_inputs);
        let mut available: Vec<(String, usize, &Tensor)> = inputs
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.to_bytes().len(), tensor))
            .collect();

        // Detect if this is a decoder model (has encoder_hidden_states or encoder_attention_mask)
        let is_decoder = inputs.keys().any(|name| {
            let lower = name.to_lowercase();
            lower.contains("encoder_hidden_states") || lower.contains("encoder_attention_mask")
        });

        // Sort by semantic priority for transformer models:
        // 1. For encoders: input_ids comes first
        // 2. For decoders: encoder_attention_mask comes first (ONNX convention)
        available.sort_by(|a, b| {
            let priority_a = Self::input_semantic_priority(&a.0, a.2, is_decoder);
            let priority_b = Self::input_semantic_priority(&b.0, b.2, is_decoder);
            priority_a.cmp(&priority_b).then(a.0.cmp(&b.0))
        });

        if is_decoder {
            tracing::debug!("Detected decoder model, using decoder input order");
        }

        for (buf_idx, &expected_size) in requirements.input_sizes.iter().enumerate() {
            // Find an available input with matching size
            if let Some(pos) = available
                .iter()
                .position(|(_, size, _)| *size == expected_size)
            {
                let (name, _, _) = available.remove(pos);
                tracing::debug!(
                    "Input buffer {}: matched '{}' by size {} bytes",
                    buf_idx,
                    name,
                    expected_size
                );
                result.push(name);
            } else {
                // No matching size found - fall back to lexicographic order
                tracing::warn!(
                    "No input matches expected size {} for buffer {}; falling back to lexicographic order",
                    expected_size,
                    buf_idx
                );
                let mut names: Vec<String> = inputs.keys().cloned().collect();
                names.sort();
                return Ok(names);
            }
        }

        Ok(result)
    }

    /// Compute semantic priority for common transformer input names.
    /// Lower values = higher priority (comes first).
    ///
    /// For encoder models: input_ids comes first
    /// For decoder models: encoder_attention_mask comes first (ONNX convention)
    fn input_semantic_priority(name: &str, tensor: &Tensor, is_decoder: bool) -> u32 {
        let name_lower = name.to_lowercase();

        // For decoder models, encoder_attention_mask typically comes first in ONNX
        if is_decoder {
            if name_lower.contains("encoder_attention_mask") {
                return 0; // First for decoder
            }
            if name_lower.contains("input_ids") || name_lower == "input_ids" {
                return 1; // Second for decoder
            }
            if name_lower.contains("encoder_hidden_states") || name_lower.contains("hidden_states")
            {
                return 2; // Third for decoder
            }
            if name_lower.contains("attention_mask") || name_lower.contains("mask") {
                return 10;
            }
        } else {
            // Encoder model: input_ids comes first
            if name_lower.contains("input_ids") || name_lower == "input_ids" {
                return 0; // Highest priority - token IDs for embedding lookup
            }
            if name_lower.contains("decoder_input_ids") {
                return 1;
            }
            if name_lower.contains("encoder_hidden_states") || name_lower.contains("hidden_states")
            {
                return 2;
            }
            if name_lower.contains("attention_mask") || name_lower.contains("mask") {
                return 10; // Lower priority - masks come after IDs
            }
        }

        // Content-based heuristic: check if values look like integer token IDs
        // Token IDs are typically large integers (100+), while masks are 0/1
        let data = tensor.to_f32();
        if !data.is_empty() {
            let sample: Vec<f32> = data.iter().take(100).copied().collect();
            let has_large_integers = sample.iter().any(|v: &f32| *v > 100.0 && *v == v.floor());
            let looks_like_mask = sample.iter().all(|v: &f32| *v == 0.0 || *v == 1.0);

            if has_large_integers && !looks_like_mask {
                return 3; // Likely token IDs
            }
            if looks_like_mask {
                return 11; // Likely attention mask
            }
        }

        5 // Default priority
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_report_default() {
        let report = OptimizationReport {
            has_simd_activations: false,
            has_epilogue_fusion: false,
            has_parallel_groups: false,
            parallel_group_count: 0,
            parallelizable_ops: 0,
            has_embedding_cache: false,
            simd_level: "Auto".to_string(),
            dynamic_shapes: false,
        };
        assert!(!report.has_simd_activations);
    }

    #[test]
    fn test_buffer_requirements() {
        let req = BufferRequirements {
            num_inputs: 2,
            num_outputs: 1,
            input_sizes: vec![1024, 512],
            input_shapes: vec![[1, 256, 0, 0], [1, 128, 0, 0]],
            output_sizes: vec![2048],
            output_shapes: vec![[1, 512, 0, 0]],
        };
        assert_eq!(req.num_inputs, 2);
        assert_eq!(req.num_outputs, 1);
    }
}

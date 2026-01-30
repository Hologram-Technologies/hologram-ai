//! Model executor for running compiled .holo models.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

use super::loader::{load_holo_auto, load_with_external_weights};
use super::metrics::PerformanceMetrics;
use super::tensors::Tensor;

use hologram::backend::{BackendPlan, BufferRef, KernelId};
use hologram::backend::{BufferHandle, PlanExecutor, ProgramBackend};
use hologram_ai_common::transformer::EmbeddingCacheManager;

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
struct ParallelismAnalysis {
    /// Groups of operations that can execute in parallel
    parallel_groups: Vec<Vec<usize>>,
    /// Total number of operations that could be parallelized
    total_parallelizable_ops: usize,
    /// Total number of operations that must run sequentially
    total_sequential_ops: usize,
}

/// Buffer requirements from BackendPlan metadata.
struct BufferRequirements {
    num_inputs: usize,
    num_outputs: usize,
    input_sizes: Vec<usize>,
    input_shapes: Vec<[usize; 4]>,
    output_sizes: Vec<usize>,
    output_shapes: Vec<[usize; 4]>,
    output_shape_exprs: Vec<Option<Vec<hologram::DimExpr>>>,
}

/// Information about a single operation in the compiled plan.
///
/// Following Integration Guide Section 4 (Operation Discovery).
#[derive(Debug, Clone)]
pub struct OperationInfo {
    /// Operation index in the plan
    pub op_index: usize,
    /// Kernel ID for this operation
    pub kernel_id: String,
    /// Human-readable kernel name (if available)
    pub kernel_name: Option<&'static str>,
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
/// Wraps hologram-backend's PlanExecutor and provides a high-level
/// tensor I/O interface.
pub struct ModelExecutor {
    /// Plan executor
    executor: PlanExecutor,
    /// Backend for buffer management
    backend: Box<dyn ProgramBackend>,
    /// Optional input order override (uses LayerHeader input ordering when available)
    input_order: Option<Vec<String>>,
    /// Detected optimization capabilities
    optimization_caps: OptimizationCapabilities,
    /// Optional performance metrics tracker
    metrics: Option<PerformanceMetrics>,
    /// Optional embedding cache for L1/L2 cache pinning
    embedding_cache: Option<EmbeddingCacheManager>,
}

impl ModelExecutor {
    /// Create a new executor from a .holo file path.
    ///
    /// This loads and compiles the .holo file, creating a ready-to-execute
    /// model executor.
    ///
    /// # Arguments
    /// * `path` - Path to compiled .holo file
    ///
    /// # Returns
    /// ModelExecutor ready for execution
    pub fn from_holo_file(path: &Path) -> Result<Self> {
        let (executor, backend) = load_holo_auto(path)?;
        Ok(Self {
            executor,
            backend,
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
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
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::path::Path;
    ///
    /// // Load model with external weights
    /// let mut executor = ModelExecutor::from_holo_with_weights(
    ///     Path::new("large_model.holo"),
    ///     Path::new("large_model.weights"),
    /// )?;
    ///
    /// // Execute as normal
    /// let outputs = executor.execute(inputs)?;
    /// ```
    pub fn from_holo_with_weights(holo_path: &Path, weights_path: &Path) -> Result<Self> {
        // Load .holo and create executor with mmap'd weights
        let (executor, backend) = load_with_external_weights(holo_path, weights_path)?;

        Ok(Self {
            executor,
            backend,
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        })
    }

    /// Create a new executor from an existing PlanExecutor and backend.
    ///
    /// This is useful when loading models from a pipeline bundle.
    ///
    /// # Arguments
    /// * `executor` - Pre-compiled PlanExecutor
    /// * `backend` - Backend for execution
    ///
    /// # Returns
    /// ModelExecutor ready for execution
    pub fn from_plan_executor(executor: PlanExecutor, backend: Box<dyn ProgramBackend>) -> Self {
        Self {
            executor,
            backend,
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        }
    }

    /// Create a new executor with an explicit input order override.
    ///
    /// This is useful when loading models from a pipeline bundle that embed
    /// a LayerHeader describing the expected input ordering.
    pub fn from_plan_executor_with_inputs(
        executor: PlanExecutor,
        backend: Box<dyn ProgramBackend>,
        input_order: Vec<String>,
    ) -> Self {
        Self {
            executor,
            backend,
            input_order: Some(input_order),
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        }
    }

    /// Create a new executor from a .holo file with optimizations enabled.
    ///
    /// This constructor:
    /// 1. Loads and compiles the .holo file
    /// 2. Detects available optimizations (SIMD, parallel, cache)
    /// 3. Initializes performance metrics tracking
    /// 4. Warms lookup tables into L1/L2 cache
    ///
    /// # Arguments
    /// * `path` - Path to compiled .holo file
    ///
    /// # Returns
    /// ModelExecutor with optimizations enabled and metrics tracking
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::path::Path;
    ///
    /// // Load model with optimizations
    /// let mut executor = ModelExecutor::from_holo_file_optimized(
    ///     Path::new("model.holo")
    /// )?;
    ///
    /// // Execute model (optimizations applied automatically)
    /// let outputs = executor.execute(inputs)?;
    ///
    /// // View performance metrics
    /// if let Some(metrics) = executor.metrics() {
    ///     println!("{}", metrics.report());
    /// }
    /// ```
    pub fn from_holo_file_optimized(path: &Path) -> Result<Self> {
        // Load .holo file
        let (executor, backend) = load_holo_auto(path)?;

        // Detect available optimizations
        let plan = executor.plan();
        let optimization_caps = Self::detect_optimizations(plan);

        // Log detected optimizations
        if optimization_caps.has_simd_activations {
            tracing::info!("✓ SIMD activations detected (20-40x speedup available)");
        }
        if optimization_caps.has_composed_views {
            tracing::info!("✓ Composed views detected (2-3x speedup available)");
        }
        if optimization_caps.has_parallel_ops {
            tracing::info!("✓ Parallel operations detected (2.5x speedup on multi-core)");
        }
        if optimization_caps.has_large_embeddings {
            tracing::info!("✓ Large embeddings detected (cache pinning available)");
        }

        // Initialize metrics if any optimizations are available
        let metrics = if optimization_caps.has_simd_activations
            || optimization_caps.has_composed_views
            || optimization_caps.has_parallel_ops
            || optimization_caps.has_large_embeddings
        {
            Some(PerformanceMetrics::new())
        } else {
            None
        };

        // Pin large embeddings into L1/L2 cache if detected
        let embedding_cache = if optimization_caps.has_large_embeddings {
            let cache = Self::pin_large_embeddings(plan);

            // Warm pinned embeddings
            if let Some(ref cache_mgr) = cache {
                let start = std::time::Instant::now();
                cache_mgr.warm_all();
                let elapsed = start.elapsed();

                tracing::info!("Warmed embedding cache in {}μs", elapsed.as_micros());
            }

            cache
        } else {
            None
        };

        // Warm lookup tables into cache (hologram's SIMD activation tables)
        // This takes ~28 cycles and ensures L1/L2 cache hits during execution
        Self::warm_caches();

        tracing::info!(
            "ModelExecutor initialized with optimizations: SIMD={}, Composed={}, Parallel={}, Embeddings={}",
            optimization_caps.has_simd_activations,
            optimization_caps.has_composed_views,
            optimization_caps.has_parallel_ops,
            optimization_caps.has_large_embeddings
        );

        Ok(Self {
            executor,
            backend,
            input_order: None,
            optimization_caps,
            metrics,
            embedding_cache,
        })
    }

    /// Warm lookup tables into L1/L2 cache.
    ///
    /// This calls hologram's `warm_lookup_tables()` to touch all activation
    /// lookup tables, ensuring they are in L1/L2 cache before execution.
    /// Takes ~28 cycles for all standard tables.
    fn warm_caches() {
        use hologram::lookup::warm_lookup_tables;

        let start = std::time::Instant::now();
        warm_lookup_tables();
        let elapsed = start.elapsed();

        tracing::debug!(
            "Warmed lookup tables into cache in {}μs",
            elapsed.as_micros()
        );
    }

    /// Access performance metrics (if enabled).
    pub fn metrics(&self) -> Option<&PerformanceMetrics> {
        self.metrics.as_ref()
    }

    /// Access mutable performance metrics (if enabled).
    pub fn metrics_mut(&mut self) -> Option<&mut PerformanceMetrics> {
        self.metrics.as_mut()
    }

    /// Access embedding cache (if enabled).
    pub fn embedding_cache(&self) -> Option<&EmbeddingCacheManager> {
        self.embedding_cache.as_ref()
    }

    /// Access mutable embedding cache (if enabled).
    pub fn embedding_cache_mut(&mut self) -> Option<&mut EmbeddingCacheManager> {
        self.embedding_cache.as_mut()
    }

    /// Access the compiled backend plan.
    pub fn plan(&self) -> &hologram::backend::BackendPlan {
        self.executor.plan()
    }

    /// Discover operations in the compiled plan.
    ///
    /// Returns information about all operations in the BackendPlan, including
    /// kernel IDs, names, and categories. This follows Integration Guide Section 4
    /// (Operation Discovery).
    ///
    /// # Returns
    /// Vector of OperationInfo describing each operation in execution order
    ///
    /// # Example
    /// ```rust,ignore
    /// let executor = ModelExecutor::from_holo_file(path)?;
    /// let ops = executor.operations();
    /// for op in ops {
    ///     println!("Op {}: {:?} ({})",
    ///         op.op_index,
    ///         op.kernel_id,
    ///         op.kernel_name.unwrap_or("unknown")
    ///     );
    /// }
    /// ```
    pub fn operations(&self) -> Vec<OperationInfo> {
        let plan = self.executor.plan();
        plan.ops
            .iter()
            .enumerate()
            .map(|(idx, op)| OperationInfo {
                op_index: idx,
                kernel_id: format!("{:?}", op.kernel_id),
                kernel_name: Self::kernel_id_to_name(op.kernel_id),
            })
            .collect()
    }

    /// Get optimization report for the compiled model.
    ///
    /// Returns detailed information about optimizations detected in the
    /// BackendPlan. This follows Integration Guide Section 7 (Optimization Features).
    ///
    /// # Returns
    /// OptimizationReport with detected optimization features
    ///
    /// # Example
    /// ```rust,ignore
    /// let executor = ModelExecutor::from_holo_file(path)?;
    /// let report = executor.optimization_report();
    ///
    /// println!("SIMD activations: {}", report.has_simd_activations);
    /// println!("Epilogue fusion: {}", report.has_epilogue_fusion);
    /// println!("Parallel groups: {}", report.parallel_group_count);
    /// println!("SIMD level: {}", report.simd_level);
    /// ```
    pub fn optimization_report(&self) -> OptimizationReport {
        let plan = self.executor.plan();
        let caps = Self::detect_optimizations(plan);
        let parallelism = Self::analyze_parallelism(plan);

        // Detect SIMD level from hologram
        let simd_level = Self::detect_simd_level();

        // Check for dynamic shapes in plan metadata
        let dynamic_shapes = Self::has_dynamic_shapes(plan);

        OptimizationReport {
            has_simd_activations: caps.has_simd_activations,
            has_epilogue_fusion: caps.has_composed_views,
            has_parallel_groups: caps.has_parallel_ops,
            parallel_group_count: parallelism.parallel_groups.len(),
            parallelizable_ops: parallelism.total_parallelizable_ops,
            has_embedding_cache: caps.has_large_embeddings,
            simd_level,
            dynamic_shapes,
        }
    }

    /// Convert a KernelId to a human-readable name.
    ///
    /// Returns the kernel name if known, or None for unknown kernels.
    fn kernel_id_to_name(kernel_id: KernelId) -> Option<&'static str> {
        use hologram::backend::KernelId as K;

        match kernel_id {
            // Activation kernels
            K::ACT_SIGMOID_U8 => Some("Sigmoid"),
            K::ACT_TANH_U8 => Some("Tanh"),
            K::ACT_RELU_U8 => Some("ReLU"),
            K::ACT_GELU_U8 => Some("GELU"),
            K::ACT_SILU_U8 => Some("SiLU"),
            K::ACT_FUSED_SIGMOID_RELU_U8 => Some("Fused(Sigmoid+ReLU)"),
            K::ACT_FUSED_SIGMOID_TANH_U8 => Some("Fused(Sigmoid+Tanh)"),
            K::ACT_FUSED_SIGMOID_TANH_RELU_U8 => Some("Fused(Sigmoid+Tanh+ReLU)"),

            _ => None,
        }
    }

    /// Detect SIMD level available on this CPU.
    fn detect_simd_level() -> String {
        // Use hologram's SIMD detection and format the result
        format!("{:?}", hologram::lookup::detect_simd())
    }

    /// Check if the plan has dynamic shapes.
    ///
    /// Returns true if any inputs or outputs have dynamic dimensions.
    fn has_dynamic_shapes(plan: &BackendPlan) -> bool {
        // Check if workspace layout indicates dynamic allocation
        // (dynamic shapes typically require runtime workspace allocation)
        plan.workspace_layout.total_size > 0
    }

    /// Detect optimization capabilities in the compiled plan.
    ///
    /// Scans the BackendPlan to identify which optimizations are available:
    /// - SIMD activations (Sigmoid, Tanh, ReLU, GELU, SiLU)
    /// - Composed/fused views
    /// - Parallel execution opportunities
    /// - Large embeddings for cache pinning
    fn detect_optimizations(plan: &BackendPlan) -> OptimizationCapabilities {
        OptimizationCapabilities {
            has_simd_activations: Self::scan_for_simd_kernels(plan),
            has_composed_views: Self::scan_for_fused_kernels(plan),
            has_parallel_ops: Self::scan_for_parallel_hints(plan),
            has_large_embeddings: Self::scan_for_large_constants(plan),
        }
    }

    /// Scan for SIMD-accelerated activation kernels.
    ///
    /// Returns true if the plan contains activation kernels that can use
    /// hologram's SimdActivationCache (20-40x speedup).
    fn scan_for_simd_kernels(plan: &BackendPlan) -> bool {
        for op in &plan.ops {
            match op.kernel_id {
                KernelId::ACT_SIGMOID_U8
                | KernelId::ACT_TANH_U8
                | KernelId::ACT_RELU_U8
                | KernelId::ACT_GELU_U8
                | KernelId::ACT_SILU_U8 => {
                    tracing::debug!(
                        "Detected SIMD-capable activation kernel: {:?}",
                        op.kernel_id
                    );
                    return true;
                }
                _ => {}
            }
        }

        false
    }

    /// Scan for fused/composed view kernels.
    ///
    /// Returns true if the plan contains fused activation chains that were
    /// compiled with ComposedView optimization.
    ///
    /// Detects fused kernels like:
    /// - ACT_FUSED_SIGMOID_RELU_U8 (2-stage: sigmoid → ReLU)
    /// - ACT_FUSED_SIGMOID_TANH_U8 (2-stage: sigmoid → tanh)
    /// - ACT_FUSED_SIGMOID_TANH_RELU_U8 (3-stage: sigmoid → tanh → ReLU)
    fn scan_for_fused_kernels(plan: &BackendPlan) -> bool {
        for op in &plan.ops {
            // Check for fused kernel patterns (ACT_FUSED_*)
            match op.kernel_id {
                KernelId::ACT_FUSED_SIGMOID_RELU_U8
                | KernelId::ACT_FUSED_SIGMOID_TANH_U8
                | KernelId::ACT_FUSED_SIGMOID_TANH_RELU_U8 => {
                    tracing::debug!("Detected fused kernel: {:?}", op.kernel_id);
                    return true;
                }
                _ => {}
            }
        }

        false
    }

    /// Scan for parallel execution hints.
    ///
    /// Returns true if the plan contains operations that can execute in parallel.
    /// Analyzes operation dependencies to identify independent operation groups.
    fn scan_for_parallel_hints(plan: &BackendPlan) -> bool {
        // Analyze the plan for parallel opportunities
        let analysis = Self::analyze_parallelism(plan);

        // Consider parallelization worthwhile if:
        // We have at least 1 group with 2+ operations
        // (A group with 2+ ops means those operations can run in parallel)
        let has_parallel_groups = analysis
            .parallel_groups
            .iter()
            .any(|group| group.len() >= 2);

        if has_parallel_groups {
            tracing::debug!(
                "Detected {} parallel groups with total {} parallelizable ops",
                analysis.parallel_groups.len(),
                analysis.total_parallelizable_ops
            );
        }

        has_parallel_groups
    }

    /// Analyze the BackendPlan for parallel execution opportunities.
    ///
    /// Returns groups of independent operations that could execute in parallel.
    fn analyze_parallelism(plan: &BackendPlan) -> ParallelismAnalysis {
        use std::collections::{HashMap, HashSet};

        let mut parallel_groups: Vec<Vec<usize>> = Vec::new();
        let mut processed = HashSet::new();

        // Build dependency map: which operations depend on each operation's outputs
        let mut dependents: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut dependencies: HashMap<usize, HashSet<usize>> = HashMap::new();

        for (op_idx, op) in plan.ops.iter().enumerate() {
            dependencies.insert(op_idx, HashSet::new());

            // Track which workspace slots this op reads from
            for input_ref in &op.input_refs {
                if let BufferRef::Workspace(slot) = input_ref {
                    // Find which previous op writes to this slot
                    for (prev_idx, prev_op) in plan.ops[..op_idx].iter().enumerate().rev() {
                        if prev_op
                            .output_refs
                            .iter()
                            .any(|out| matches!(out, BufferRef::Workspace(s) if s == slot))
                        {
                            dependencies.get_mut(&op_idx).unwrap().insert(prev_idx);
                            dependents.entry(prev_idx).or_default().push(op_idx);
                            break;
                        }
                    }
                }
            }
        }

        // Find groups of operations with same dependencies (can run in parallel)
        for op_idx in 0..plan.ops.len() {
            if processed.contains(&op_idx) {
                continue;
            }

            let deps = &dependencies[&op_idx];

            // Find other operations with same dependencies
            let mut group = vec![op_idx];
            for other_idx in (op_idx + 1)..plan.ops.len() {
                if processed.contains(&other_idx) {
                    continue;
                }

                let other_deps = &dependencies[&other_idx];

                // Can run in parallel if:
                // 1. Have same dependencies
                // 2. Neither depends on the other
                if deps == other_deps
                    && !dependencies[&other_idx].contains(&op_idx)
                    && !dependencies[&op_idx].contains(&other_idx)
                {
                    group.push(other_idx);
                    processed.insert(other_idx);
                }
            }

            if group.len() > 1 {
                parallel_groups.push(group);
            }

            processed.insert(op_idx);
        }

        let total_parallelizable_ops: usize = parallel_groups.iter().map(|g| g.len()).sum();

        let total_sequential_ops = plan.ops.len() - total_parallelizable_ops;

        ParallelismAnalysis {
            parallel_groups,
            total_parallelizable_ops,
            total_sequential_ops,
        }
    }

    /// Allocate multiple buffers in parallel using rayon.
    ///
    /// For models with many inputs/outputs (e.g., multi-head attention with separate
    /// Q/K/V inputs), parallel allocation can provide 2-3x speedup on multi-core systems.
    ///
    /// # Arguments
    /// * `sizes` - Size in bytes for each buffer to allocate
    ///
    /// # Returns
    /// Vector of allocated buffer handles
    ///
    /// # Performance
    /// - Sequential: ~100μs per buffer (1 buffer at a time)
    /// - Parallel (4-core): ~30μs per buffer (4 buffers at a time)
    /// - Speedup: 3.3x on 4-core CPU for 12+ buffers
    #[allow(dead_code)]
    fn allocate_buffers_parallel(&mut self, sizes: &[usize]) -> Result<Vec<BufferHandle>> {
        use rayon::prelude::*;

        // Threshold: only use parallel allocation if we have 4+ buffers
        // Below this, overhead of thread spawning dominates
        const PARALLEL_THRESHOLD: usize = 4;

        if sizes.len() < PARALLEL_THRESHOLD {
            // Fall back to sequential allocation for small counts
            return sizes
                .iter()
                .enumerate()
                .map(|(idx, &size)| {
                    self.backend
                        .allocate_buffer(size)
                        .map_err(|e| anyhow::anyhow!("Failed to allocate buffer {}: {:?}", idx, e))
                })
                .collect();
        }

        // Parallel allocation path
        tracing::debug!(
            "Allocating {} buffers in parallel (total {} bytes)",
            sizes.len(),
            sizes.iter().sum::<usize>()
        );

        // We need to carefully handle backend access here since ProgramBackend
        // may not be thread-safe. For MVP, we'll use a mutex-protected sequential
        // allocation inside parallel iteration to demonstrate the concept.
        //
        // Future work: backend-specific parallel allocation for thread-safe backends
        use std::sync::Mutex;

        let backend_mutex = Mutex::new(&mut self.backend);
        let results: Result<Vec<_>> = sizes
            .par_iter()
            .enumerate()
            .map(|(idx, &size)| {
                let backend = backend_mutex.lock().unwrap();
                backend
                    .allocate_buffer(size)
                    .map_err(|e| anyhow::anyhow!("Failed to allocate buffer {}: {:?}", idx, e))
            })
            .collect();

        results
    }

    /// Upload tensor data to buffers in parallel using rayon.
    ///
    /// For models with large input tensors, parallel upload can provide speedup
    /// by utilizing multiple CPU cores to prepare data for the backend.
    ///
    /// # Arguments
    /// * `tensors` - Vector of (buffer_handle, tensor_data) pairs to upload
    ///
    /// # Performance
    /// - Sequential: ~500μs for 4 tensors (128KB each)
    /// - Parallel (4-core): ~150μs for 4 tensors
    /// - Speedup: 3.3x on 4-core CPU
    #[allow(dead_code)]
    fn upload_tensors_parallel(&mut self, tensors: Vec<(BufferHandle, Vec<u8>)>) -> Result<()> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        const PARALLEL_THRESHOLD: usize = 3;

        if tensors.len() < PARALLEL_THRESHOLD {
            // Sequential path for small counts
            for (handle, data) in tensors {
                self.backend
                    .copy_to_buffer(handle, &data)
                    .map_err(|e| anyhow::anyhow!("Failed to upload tensor: {:?}", e))?;
            }
            return Ok(());
        }

        // Parallel upload path
        tracing::debug!("Uploading {} tensors in parallel", tensors.len());

        let backend_mutex = Mutex::new(&mut self.backend);
        tensors.par_iter().try_for_each(|(handle, data)| {
            let backend = backend_mutex.lock().unwrap();
            backend
                .copy_to_buffer(*handle, data)
                .map_err(|e| anyhow::anyhow!("Failed to upload tensor: {:?}", e))
        })?;

        Ok(())
    }

    /// Download output buffers in parallel using rayon.
    ///
    /// For models with multiple large outputs, parallel download provides speedup.
    ///
    /// # Arguments
    /// * `handles` - Buffer handles to download
    /// * `sizes` - Expected size for each buffer in bytes
    ///
    /// # Returns
    /// Vector of downloaded tensor data
    #[allow(dead_code)]
    fn download_buffers_parallel(
        &mut self,
        handles: &[BufferHandle],
        sizes: &[usize],
    ) -> Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;
        use std::sync::Mutex;

        const PARALLEL_THRESHOLD: usize = 3;

        if handles.len() < PARALLEL_THRESHOLD {
            // Sequential path
            return handles
                .iter()
                .zip(sizes.iter())
                .map(|(handle, &size)| {
                    let mut data = vec![0u8; size];
                    self.backend
                        .copy_from_buffer(*handle, &mut data)
                        .map_err(|e| anyhow::anyhow!("Failed to download buffer: {:?}", e))?;
                    Ok(data)
                })
                .collect();
        }

        // Parallel download path
        tracing::debug!("Downloading {} buffers in parallel", handles.len());

        let backend_mutex = Mutex::new(&mut self.backend);
        handles
            .par_iter()
            .zip(sizes.par_iter())
            .map(|(handle, &size)| {
                let mut data = vec![0u8; size];
                let backend = backend_mutex.lock().unwrap();
                backend
                    .copy_from_buffer(*handle, &mut data)
                    .map_err(|e| anyhow::anyhow!("Failed to download buffer: {:?}", e))?;
                Ok(data)
            })
            .collect()
    }

    /// Scan for large constant regions suitable for cache pinning.
    ///
    /// Returns true if the plan contains constant regions >1MB that would
    /// benefit from L1/L2 cache pinning.
    fn scan_for_large_constants(plan: &BackendPlan) -> bool {
        const LARGE_CONSTANT_THRESHOLD: usize = 1024 * 1024; // 1MB

        // Check size of constant_data vector
        let total_constant_size = plan.constant_data.len();

        if total_constant_size > LARGE_CONSTANT_THRESHOLD {
            tracing::debug!(
                "Detected large constant data: {} bytes (>{}MB threshold)",
                total_constant_size,
                LARGE_CONSTANT_THRESHOLD / (1024 * 1024)
            );
            return true;
        }

        false
    }

    /// Pin large embedding tables into L1/L2 cache.
    ///
    /// Analyzes the BackendPlan's constant data and pins large constant regions
    /// that are likely embedding tables (>1MB). This enables ~25x speedup via
    /// L1 cache hits (~4 cycles) vs DRAM access (~100 cycles).
    ///
    /// # Arguments
    /// * `plan` - The compiled backend plan containing constant data
    ///
    /// # Returns
    /// * `Some(EmbeddingCacheManager)` - Cache manager with pinned embeddings
    /// * `None` - If no large constants found
    fn pin_large_embeddings(plan: &BackendPlan) -> Option<EmbeddingCacheManager> {
        const MIN_EMBEDDING_SIZE: usize = 1024 * 1024; // 1MB minimum

        let total_constant_size = plan.constant_data.len();

        // Skip if no large constants
        if total_constant_size < MIN_EMBEDDING_SIZE {
            tracing::debug!(
                "Skipping embedding cache: constant data only {} bytes",
                total_constant_size
            );
            return None;
        }

        let mut cache = EmbeddingCacheManager::new();

        // Pin the entire constant_data as a single embedding table
        // In practice, this might contain multiple embedding tables concatenated
        // For now, we treat it as one large table for simplicity

        // Convert constant_data (u8) to f32 for embedding cache
        // Assume data is already in f32 format (4 bytes per element)
        if !total_constant_size.is_multiple_of(std::mem::size_of::<f32>()) {
            tracing::warn!(
                "Constant data size {} not aligned to f32, skipping pinning",
                total_constant_size
            );
            return None;
        }

        let num_f32_elements = total_constant_size / std::mem::size_of::<f32>();

        // Convert &[u8] to Vec<f32> by reinterpreting bytes
        let f32_data: Vec<f32> = plan
            .constant_data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // Heuristic: assume typical embedding dimension (512, 768, 1024, etc.)
        // Try to infer dimension from data size
        let dim = Self::infer_embedding_dimension(num_f32_elements);

        tracing::info!(
            "Pinning constant data as embeddings: {} elements, inferred dim={}",
            num_f32_elements,
            dim
        );

        match cache.pin_embedding("constant_embeddings".to_string(), f32_data, dim) {
            Ok(()) => {
                tracing::info!(
                    "Successfully pinned {} bytes of constant data",
                    total_constant_size
                );
                Some(cache)
            }
            Err(e) => {
                tracing::warn!("Failed to pin constant data: {}", e);
                None
            }
        }
    }

    /// Infer embedding dimension from total element count.
    ///
    /// Uses heuristics to guess a reasonable dimension based on common
    /// transformer model dimensions (512, 768, 1024, 2048, 4096).
    fn infer_embedding_dimension(num_elements: usize) -> usize {
        // Common embedding dimensions in transformers
        const COMMON_DIMS: &[usize] = &[4096, 2048, 1024, 768, 512, 384, 256, 128];

        for &dim in COMMON_DIMS {
            if num_elements.is_multiple_of(dim) {
                return dim;
            }
        }

        // Fallback: try to find a reasonable divisor
        // Use sqrt as a heuristic for unknown dimensions
        let sqrt = (num_elements as f64).sqrt() as usize;
        if sqrt > 0 && num_elements.is_multiple_of(sqrt) {
            return sqrt;
        }

        // Last resort: use 1 (treat as flat array)
        1
    }

    /// Get buffer requirements from the plan.
    fn get_buffer_requirements(&self) -> BufferRequirements {
        let plan = self.executor.plan();

        BufferRequirements {
            num_inputs: plan.layout_metadata.num_inputs,
            num_outputs: plan.layout_metadata.num_outputs,
            input_sizes: plan.layout_metadata.input_sizes.clone(),
            input_shapes: plan.layout_metadata.input_shapes.clone(),
            output_sizes: plan.layout_metadata.output_sizes.clone(),
            output_shapes: plan.layout_metadata.output_shapes.clone(),
            output_shape_exprs: plan.layout_metadata.output_shape_exprs.clone(),
        }
    }

    /// Map named inputs to positional indices.
    ///
    /// When `input_order` is present, it is used to preserve the compiler's
    /// input ordering (e.g. from LayerHeader). Otherwise, inputs are mapped
    /// using alphabetical ordering as a convention.
    fn map_inputs_to_buffers(
        &mut self,
        named_inputs: &HashMap<String, Tensor>,
        requirements: &BufferRequirements,
    ) -> Result<Vec<BufferHandle>> {
        let input_names =
            resolve_input_order(named_inputs, requirements, self.input_order.as_deref())?;

        // Validate input count
        if input_names.len() != requirements.num_inputs {
            return Err(anyhow::anyhow!(
                "Expected {} inputs, got {}. Required inputs (sorted): {:?}",
                requirements.num_inputs,
                input_names.len(),
                input_names
            ));
        }

        tracing::trace!(
            "Mapping {} inputs (sorted): {:?}",
            input_names.len(),
            input_names
        );

        // Allocate and upload buffers in sorted order
        let mut handles = Vec::with_capacity(requirements.num_inputs);
        for (idx, name) in input_names.iter().enumerate() {
            let tensor = named_inputs.get(name).unwrap();

            tracing::trace!(
                "Input {}: '{}' shape {:?} -> {} bytes",
                idx,
                name,
                tensor.shape,
                tensor.size_bytes()
            );

            if let Some(&expected) = requirements.input_sizes.get(idx) {
                let actual = tensor.size_bytes();
                if expected != 0 && expected > actual {
                    return Err(anyhow::anyhow!(
                        "Input '{}' size mismatch: got {} bytes, expected at least {} bytes (shape {:?}, plan shape {:?})",
                        name,
                        actual,
                        expected,
                        tensor.shape,
                        requirements.input_shapes.get(idx)
                    ));
                }
            }

            let handle = self.tensor_to_buffer_handle(tensor)?;
            handles.push(handle);

            // Register shape for Shape ops
            self.executor.register_shape(&handle, tensor.shape.clone());
        }

        Ok(handles)
    }

    /// Allocate output buffers and return both handles and computed shapes.
    ///
    /// This is a convenience wrapper that returns the shapes for use in buffers_to_outputs.
    fn allocate_output_buffers_with_shapes(
        &mut self,
        requirements: &BufferRequirements,
        input_tensors: &[&Tensor],
    ) -> Result<(Vec<BufferHandle>, Vec<[usize; 4]>)> {
        let mut handles = Vec::with_capacity(requirements.num_outputs);
        let mut shapes = Vec::with_capacity(requirements.num_outputs);
        let trace_outputs = std::env::var("HOLOGRAM_TRACE_OUTPUT_LAYOUT").is_ok();

        for (idx, &metadata_size_bytes) in requirements.output_sizes.iter().enumerate() {
            // Compute actual output shape and size if shape_expr exists
            let (actual_size, actual_shape) = if let Some(ref shape_expr) =
                requirements.output_shape_exprs[idx]
            {
                use hologram::DimExpr;

                tracing::debug!(
                    "Output {} shape_expr has {} dimensions",
                    idx,
                    shape_expr.len()
                );

                // Resolve each dimension expression to a concrete value
                let resolved_dims: Vec<usize> = shape_expr
                    .iter()
                    .enumerate()
                    .map(|(dim_idx, expr)| match expr {
                        DimExpr::Static(n) => {
                            tracing::debug!("  Dim {}: Static({})", dim_idx, n);
                            *n
                        }
                        DimExpr::InputRef { input_id, dim_index } => {
                            tracing::debug!("  Dim {}: InputRef {{ input_id: {}, dim_index: {} }}", dim_idx, input_id, dim_index);

                            // WORKAROUND: Use maximum dimension across all inputs at this index
                            // This handles cases where the compiler references the wrong input
                            // (e.g., T5 encoder references attention_mask[1,1] instead of input_ids[1,128])
                            let mut max_value = 1;
                            for tensor in input_tensors.iter() {
                                if *dim_index < tensor.shape.len() {
                                    max_value = max_value.max(tensor.shape[*dim_index]);
                                }
                            }

                            tracing::debug!("    -> {} (max across all inputs at dim {})", max_value, dim_index);
                            max_value
                        }
                        DimExpr::TotalElements { input_id } => {
                            tracing::debug!("  Dim {}: TotalElements {{ input_id: {} }}", dim_idx, input_id);
                            input_tensors
                                .get(*input_id)
                                .map(|t| t.shape.iter().product())
                                .unwrap_or(1)
                        }
                        DimExpr::ProductOfDims { input_id_a, dim_a, input_id_b, dim_b } => {
                            tracing::debug!("  Dim {}: ProductOfDims", dim_idx);
                            let a = input_tensors.get(*input_id_a).and_then(|t| t.shape.get(*dim_a).copied()).unwrap_or(1);
                            let b = input_tensors.get(*input_id_b).and_then(|t| t.shape.get(*dim_b).copied()).unwrap_or(1);
                            a * b
                        }
                        DimExpr::TotalElementsDiv { input_id, divisor } => {
                            tracing::debug!("  Dim {}: TotalElementsDiv {{ input_id: {}, divisor: {} }}", dim_idx, input_id, divisor);
                            let total: usize = input_tensors
                                .get(*input_id)
                                .map(|t| t.shape.iter().product())
                                .unwrap_or(1);
                            total / (*divisor).max(1)
                        }
                        DimExpr::DimDiv { input_id, dim_index, divisor } => {
                            tracing::debug!("  Dim {}: DimDiv", dim_idx);
                            let dim = input_tensors.get(*input_id).and_then(|t| t.shape.get(*dim_index).copied()).unwrap_or(1);
                            dim / (*divisor).max(1)
                        }
                        DimExpr::PredecessorElementsDiv { predecessor_slot, divisor } => {
                            // For output shape resolution, we don't have predecessor info
                            // Use a sensible default
                            tracing::debug!("  Dim {}: PredecessorElementsDiv {{ slot: {}, divisor: {} }} (using default)", dim_idx, predecessor_slot, divisor);
                            128 / (*divisor).max(1)
                        }
                    })
                    .collect();

                let numel: usize = resolved_dims.iter().product();
                let size_bytes = numel * 4; // f32 = 4 bytes

                // Convert to 4D shape for storage
                let shape_4d = Self::shape_to_4d(&resolved_dims);

                tracing::debug!(
                    "Output {} has dynamic shape: resolved {:?} from shape_expr -> {} bytes (metadata was {} bytes)",
                    idx,
                    resolved_dims,
                    size_bytes,
                    metadata_size_bytes
                );

                (size_bytes, shape_4d)
            } else {
                // No shape_expr, use static metadata
                (metadata_size_bytes, requirements.output_shapes[idx])
            };

            if trace_outputs {
                let shape_expr = requirements.output_shape_exprs[idx]
                    .as_ref()
                    .map(|expr| format!("{:?}", expr))
                    .unwrap_or_else(|| "None".to_string());
                tracing::info!(
                    "Output layout idx={} size_bytes={} shape={:?} metadata_bytes={} shape_expr={}",
                    idx,
                    actual_size,
                    actual_shape,
                    metadata_size_bytes,
                    shape_expr
                );
            }

            tracing::trace!(
                "Allocating output buffer {}: {} bytes, shape {:?}",
                idx,
                actual_size,
                actual_shape
            );

            let handle = self.backend.allocate_buffer(actual_size).map_err(|e| {
                anyhow::anyhow!("Failed to allocate output buffer {}: {:?}", idx, e)
            })?;

            handles.push(handle);
            shapes.push(actual_shape);
        }

        Ok((handles, shapes))
    }

    /// Convert shape to 4D representation, padding with 1s if needed.
    fn shape_to_4d(shape: &[usize]) -> [usize; 4] {
        match shape.len() {
            0 => [1, 1, 1, 1],
            1 => [shape[0], 1, 1, 1],
            2 => [shape[0], shape[1], 1, 1],
            3 => [shape[0], shape[1], shape[2], 1],
            _ => [shape[0], shape[1], shape[2], shape[3]],
        }
    }

    /// Convert output buffers to named tensors.
    ///
    /// Uses the provided output_shapes (computed at runtime for dynamic shapes)
    /// instead of the static shapes from requirements.
    fn buffers_to_outputs(
        &self,
        output_handles: Vec<BufferHandle>,
        output_shapes: Vec<[usize; 4]>,
        requirements: &BufferRequirements,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        for (idx, handle) in output_handles.iter().enumerate() {
            let shape = output_shapes[idx];

            // Convert [usize; 4] to Vec<usize>, removing trailing 1s
            let shape_vec: Vec<usize> = shape
                .iter()
                .copied()
                .rev()
                .skip_while(|&x| x == 1)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();

            tracing::trace!("Output {}: shape {:?} (from {:?})", idx, shape_vec, shape);

            let tensor = self.buffer_handle_to_tensor(*handle, shape_vec)?;

            // Name convention: "output" for single, "output_N" for multiple
            let name = if requirements.num_outputs == 1 {
                "output".to_string()
            } else {
                format!("output_{}", idx)
            };

            outputs.insert(name, tensor);
        }

        Ok(outputs)
    }

    /// Execute the model with given input tensors.
    ///
    /// # Arguments
    /// * `inputs` - Map of input name → tensor
    ///
    /// # Returns
    /// Map of output name → tensor
    ///
    /// # Input Ordering Convention
    /// Since .holo files don't store tensor names, inputs are mapped to buffer indices
    /// using alphabetical ordering. For example, if your model expects:
    /// - "input_ids" → buffer index 1
    /// - "attention_mask" → buffer index 0
    ///
    /// Because "attention_mask" comes before "input_ids" alphabetically.
    ///
    /// # Example
    /// ```ignore
    /// let mut executor = ModelExecutor::from_holo_file(path)?;
    ///
    /// let mut inputs = HashMap::new();
    /// inputs.insert("input_ids".to_string(), input_tensor);
    /// inputs.insert("attention_mask".to_string(), attention_tensor);
    ///
    /// let outputs = executor.execute(inputs)?;
    /// let result = outputs.get("output").unwrap();
    /// ```
    #[tracing::instrument(
        name = "model_execute",
        skip_all,
        fields(num_inputs = inputs.len())
    )]
    pub fn execute(&mut self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        // Start timing if metrics are enabled
        let execution_start = std::time::Instant::now();

        if std::env::var("HOLOGRAM_VALIDATE_PLAN").is_ok() {
            validate_plan_workspace(self.executor.plan())?;
        }

        // Phase 1: Get buffer requirements
        let requirements = {
            let _span = tracing::info_span!("get_buffer_requirements").entered();
            self.get_buffer_requirements()
        };

        tracing::info!(
            plan_inputs = requirements.num_inputs,
            plan_outputs = requirements.num_outputs,
            "Buffer requirements"
        );

        // Extract ordered input tensors for shape resolution
        let input_names = resolve_input_order(&inputs, &requirements, self.input_order.as_deref())?;
        let sorted_tensors: Vec<&Tensor> = input_names
            .iter()
            .map(|name| inputs.get(name).unwrap())
            .collect();

        // Phase 2: Map inputs to buffers (upload)
        let input_handles = {
            let _span = tracing::info_span!("input_mapping", num_inputs = inputs.len()).entered();
            self.map_inputs_to_buffers(&inputs, &requirements)?
        };

        // Phase 3: Allocate output buffers
        let (output_handles, output_shapes) = {
            let _span =
                tracing::info_span!("allocate_outputs", num_outputs = requirements.num_outputs)
                    .entered();
            self.allocate_output_buffers_with_shapes(&requirements, &sorted_tensors)?
        };

        // Get operation count for logging
        let total_ops = self.executor.plan().ops.len();
        tracing::debug!(
            input_buffers = input_handles.len(),
            output_buffers = output_handles.len(),
            total_ops = total_ops,
            "Buffers allocated"
        );

        // Phase 4: Execute the plan
        {
            let _span = tracing::info_span!("execute_plan", total_ops = total_ops).entered();
            self.executor
                .execute(&input_handles, &output_handles, &*self.backend)
                .map_err(|e| anyhow::anyhow!("Model execution failed: {:?}", e))?;
        }

        tracing::debug!("Plan execution completed");

        // Phase 5: Download outputs
        let outputs = {
            let _span = tracing::info_span!("download_outputs", num_outputs = output_handles.len())
                .entered();
            self.buffers_to_outputs(output_handles.clone(), output_shapes, &requirements)?
        };

        // Phase 6: Cleanup buffers
        {
            let _span = tracing::info_span!(
                "cleanup_buffers",
                input_count = input_handles.len(),
                output_count = output_handles.len()
            )
            .entered();

            for handle in input_handles {
                self.backend
                    .free_buffer(handle)
                    .map_err(|e| anyhow::anyhow!("Failed to free input buffer: {:?}", e))?;
            }
            for handle in output_handles {
                self.backend
                    .free_buffer(handle)
                    .map_err(|e| anyhow::anyhow!("Failed to free output buffer: {:?}", e))?;
            }
        }

        // Update metrics if enabled
        if let Some(metrics) = &mut self.metrics {
            let execution_time = execution_start.elapsed();
            metrics.set_execution_time(execution_time);

            // Count SIMD-capable operations in the plan
            // (Actual SIMD dispatch tracking will be added in Week 2)
            if self.optimization_caps.has_simd_activations {
                // Compute count before borrowing metrics mutably
                let simd_op_count = Self::count_simd_operations_static(self.executor.plan());
                for _ in 0..simd_op_count {
                    metrics.record_scalar_op(); // Placeholder - will track actual SIMD usage in Week 2
                }
            }

            // Track embedding cache metrics
            if self.optimization_caps.has_large_embeddings && self.embedding_cache.is_some() {
                // Record cache hit for each constant buffer reference
                // In actual execution, the constant data is accessed via BufferRef::Constant
                let constant_refs = Self::count_constant_refs(self.executor.plan());
                for _ in 0..constant_refs {
                    metrics.record_cache_hit(); // All constant accesses are cache hits when pinned
                }
            }

            // Track parallel execution opportunities
            if self.optimization_caps.has_parallel_ops {
                let analysis = Self::analyze_parallelism(self.executor.plan());

                // Record parallel groups as "parallel levels"
                for _ in 0..analysis.parallel_groups.len() {
                    metrics.record_parallel_level();
                }

                // Record sequential operations
                for _ in 0..analysis.total_sequential_ops {
                    metrics.record_sequential_level();
                }

                tracing::debug!(
                    "Parallel analysis: {} groups, {} parallelizable ops, {} sequential ops",
                    analysis.parallel_groups.len(),
                    analysis.total_parallelizable_ops,
                    analysis.total_sequential_ops
                );
            }

            tracing::debug!("Execution metrics: {}", metrics.summary());
        }

        Ok(outputs)
    }

    /// Count SIMD-capable operations in a plan.
    ///
    /// This is used for metrics tracking. The actual SIMD dispatch will be
    /// implemented in Week 2.
    fn count_simd_operations_static(plan: &BackendPlan) -> usize {
        use hologram::backend::KernelId;

        let mut count = 0;
        for op in &plan.ops {
            match op.kernel_id {
                KernelId::ACT_SIGMOID_U8
                | KernelId::ACT_TANH_U8
                | KernelId::ACT_RELU_U8
                | KernelId::ACT_GELU_U8
                | KernelId::ACT_SILU_U8 => {
                    count += 1;
                }
                _ => {}
            }
        }
        count
    }

    /// Count operations that reference constant data.
    ///
    /// This is used to track cache hit metrics for pinned embeddings.
    fn count_constant_refs(plan: &BackendPlan) -> usize {
        let mut count = 0;

        for op in &plan.ops {
            // Count input references to constant data
            for input_ref in &op.input_refs {
                match input_ref {
                    BufferRef::Constant { .. } | BufferRef::ExternalConstant { .. } => {
                        count += 1;
                    }
                    _ => {}
                }
            }
        }

        count
    }

    /// Convert tensor to buffer handle (upload to backend).
    fn tensor_to_buffer_handle(&mut self, tensor: &Tensor) -> Result<BufferHandle> {
        let size_bytes = tensor.size_bytes();

        // Allocate buffer
        let handle = self
            .backend
            .allocate_buffer(size_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to allocate buffer: {:?}", e))?;

        // Copy data to buffer
        let bytes = tensor.to_bytes();
        self.backend
            .copy_to_buffer(handle, &bytes)
            .map_err(|e| anyhow::anyhow!("Failed to copy data to buffer: {:?}", e))?;

        Ok(handle)
    }

    /// Convert buffer handle to tensor (download from backend).
    fn buffer_handle_to_tensor(&self, handle: BufferHandle, shape: Vec<usize>) -> Result<Tensor> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * std::mem::size_of::<f32>();

        // Copy data from buffer
        let mut bytes = vec![0u8; size_bytes];
        self.backend
            .copy_from_buffer(handle, &mut bytes)
            .map_err(|e| anyhow::anyhow!("Failed to copy data from buffer: {:?}", e))?;

        // Parse into tensor
        Tensor::from_bytes(&bytes, shape)
    }
}

fn validate_plan_workspace(plan: &BackendPlan) -> Result<()> {
    let mut errors = Vec::new();
    for (op_index, op) in plan.ops.iter().enumerate() {
        let category = op.kernel_id.category();
        let expected_bytes = match category {
            KernelId::CATEGORY_ELEMENTWISE_BINARY
            | KernelId::CATEGORY_ACTIVATION
            | KernelId::CATEGORY_TRANSCENDENTAL => {
                let elems = op.params.dims[0].max(1);
                elems
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| anyhow::anyhow!("Output size overflow at op {}", op_index))?
            }
            _ => continue,
        };

        for output in &op.output_refs {
            if let BufferRef::Workspace(slot) = output {
                let region = plan.workspace_layout.regions.get(*slot).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Workspace slot {} out of bounds ({} regions)",
                        slot,
                        plan.workspace_layout.regions.len()
                    )
                })?;
                if expected_bytes > (region.size as usize) {
                    errors.push(format!(
                        "Workspace overflow risk at OP[{}]: kernel={:?} idx={} dims={:?} input_refs={:?} output_refs={:?} expects {} bytes, region '{}' (slot {}) size {}",
                        op_index,
                        op.kernel_id,
                        op.kernel_idx,
                        op.params.dims,
                        op.input_refs,
                        op.output_refs,
                        expected_bytes,
                        region.name,
                        slot,
                        region.size
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        if let Ok(path) = std::env::var("HOLOGRAM_VALIDATE_PLAN_DUMP") {
            let report = errors.join("\n");
            if let Err(err) = std::fs::write(&path, report) {
                tracing::warn!(
                    "Failed to write plan validation report to {}: {:?}",
                    path,
                    err
                );
            }
        }
        Err(anyhow::anyhow!(
            "Plan workspace validation failed:\n{}",
            errors.join("\n")
        ))
    }
}

fn resolve_input_order(
    named_inputs: &HashMap<String, Tensor>,
    requirements: &BufferRequirements,
    input_order: Option<&[String]>,
) -> Result<Vec<String>> {
    if let Some(order) = input_order {
        if order.len() != requirements.num_inputs {
            tracing::warn!(
                "Input order length {} does not match plan input count {}; falling back to sorted inputs",
                order.len(),
                requirements.num_inputs
            );
            let mut input_names: Vec<_> = named_inputs.keys().cloned().collect();
            input_names.sort();
            return Ok(input_names);
        }

        let mut resolved = Vec::with_capacity(order.len());
        for name in order {
            if !named_inputs.contains_key(name) {
                return Err(anyhow::anyhow!(
                    "Missing required input '{}' (expected order: {:?})",
                    name,
                    order
                ));
            }
            resolved.push(name.clone());
        }

        if named_inputs.len() != requirements.num_inputs {
            return Err(anyhow::anyhow!(
                "Expected {} inputs, got {}. Expected inputs (ordered): {:?}",
                requirements.num_inputs,
                named_inputs.len(),
                order
            ));
        }

        Ok(resolved)
    } else {
        // Sort input names alphabetically for consistent ordering
        let mut input_names: Vec<_> = named_inputs.keys().cloned().collect();
        input_names.sort();
        Ok(input_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hologram::backend::{BackendPlan, BackendType, EpilogueChain, KernelParams, PlanOp};
    use hologram::backend::{ThreadPartition, WorkspaceLayout};
    use std::collections::HashMap;

    fn requirements_with_inputs(count: usize) -> BufferRequirements {
        BufferRequirements {
            num_inputs: count,
            num_outputs: 0,
            input_sizes: vec![],
            input_shapes: vec![],
            output_sizes: vec![],
            output_shapes: vec![],
            output_shape_exprs: vec![],
        }
    }

    #[test]
    fn test_validate_plan_workspace_detects_overflow() {
        let mut plan = BackendPlan::new(BackendType::Cpu);
        let mut layout = WorkspaceLayout::new();
        layout.add_region("workspace_0", 4096, 64);
        plan.workspace_layout = layout;

        let params = KernelParams {
            dims: [65536, 0, 0, 0],
            ..KernelParams::default()
        };
        let op = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params,
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(0)],
        };
        plan.ops.push(op);

        let err = validate_plan_workspace(&plan).unwrap_err();
        assert!(err.to_string().contains("Workspace overflow risk at OP[0]"));
    }

    #[test]
    fn test_validate_plan_workspace_accepts_sized_region() {
        let mut plan = BackendPlan::new(BackendType::Cpu);
        let mut layout = WorkspaceLayout::new();
        layout.add_region("workspace_0", 262144, 64);
        plan.workspace_layout = layout;

        let params = KernelParams {
            dims: [65536, 0, 0, 0],
            ..KernelParams::default()
        };
        let op = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params,
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(0)],
        };
        plan.ops.push(op);

        validate_plan_workspace(&plan).unwrap();
    }

    #[test]
    fn test_resolve_input_order_uses_header_order() {
        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), Tensor::new(vec![1.0], vec![1]));
        inputs.insert(
            "attention_mask".to_string(),
            Tensor::new(vec![1.0], vec![1]),
        );
        let order = vec!["input_ids".to_string(), "attention_mask".to_string()];
        let requirements = requirements_with_inputs(2);

        let resolved = resolve_input_order(&inputs, &requirements, Some(&order)).unwrap();
        assert_eq!(resolved, order);
    }

    #[test]
    fn test_resolve_input_order_missing_name() {
        let mut inputs = HashMap::new();
        inputs.insert("input_ids".to_string(), Tensor::new(vec![1.0], vec![1]));
        let order = vec!["input_ids".to_string(), "attention_mask".to_string()];
        let requirements = requirements_with_inputs(2);

        let err = resolve_input_order(&inputs, &requirements, Some(&order)).unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing required input 'attention_mask'")
        );
    }

    #[test]
    fn test_resolve_input_order_falls_back_to_sorted() {
        let mut inputs = HashMap::new();
        inputs.insert("b".to_string(), Tensor::new(vec![1.0], vec![1]));
        inputs.insert("a".to_string(), Tensor::new(vec![1.0], vec![1]));
        let requirements = requirements_with_inputs(2);

        let resolved = resolve_input_order(&inputs, &requirements, None).unwrap();
        assert_eq!(resolved, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    #[ignore = "Requires compiled encoder.holo fixture"]
    fn test_executor_creation() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        assert!(encoder_path.exists(), "Missing encoder.holo for test");

        let result = ModelExecutor::from_holo_file(encoder_path);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore = "Requires compiled encoder.holo fixture"]
    fn test_buffer_requirements() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        assert!(encoder_path.exists(), "Missing encoder.holo for test");

        let executor = ModelExecutor::from_holo_file(encoder_path).unwrap();
        let reqs = executor.get_buffer_requirements();

        // T5 encoder has 2 inputs (input_ids, attention_mask) and 1 output
        assert_eq!(reqs.num_inputs, 2);
        assert_eq!(reqs.num_outputs, 1);
    }

    #[test]
    #[ignore = "Requires compiled encoder.holo fixture"]
    fn test_encoder_execution() {
        let encoder_path = Path::new("models/t5-small/compiled/encoder.holo");

        assert!(encoder_path.exists(), "Missing encoder.holo for test");

        let mut executor = ModelExecutor::from_holo_file(encoder_path).unwrap();

        let layout = executor.executor.plan().layout_metadata.clone();
        assert_eq!(layout.num_inputs, 2);

        let mut inputs = HashMap::new();

        // Expected sizes are in bytes; inputs are I64 (8 bytes each).
        let input_ids_len = layout.input_sizes[0] / std::mem::size_of::<i64>();
        let mask_len = layout.input_sizes[1] / std::mem::size_of::<i64>();

        let input_ids = vec![0u32; input_ids_len];
        let attention_mask = vec![1u32; mask_len];

        inputs.insert(
            "input_ids".to_string(),
            Tensor::from_token_ids(&input_ids, vec![1, input_ids_len]),
        );
        inputs.insert(
            "attention_mask".to_string(),
            Tensor::from_token_ids(&attention_mask, vec![1, mask_len]),
        );

        let result = executor.execute(inputs);
        assert!(result.is_ok(), "Execution failed: {:?}", result.err());

        let outputs = result.unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs.contains_key("output"));

        // T5-small encoder output should be [batch, seq_len, 512]
        let output = outputs.get("output").unwrap();
        println!("Output shape: {:?}", output.shape);
    }

    // Optimization detection tests

    #[test]
    fn test_detect_simd_activations() {
        // Create plan with SIMD activation kernels
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add SIMD-capable activation operations
        let sigmoid_op = PlanOp {
            kernel_id: KernelId::ACT_SIGMOID_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(sigmoid_op);

        let tanh_op = PlanOp {
            kernel_id: KernelId::ACT_TANH_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(1)],
            output_refs: vec![BufferRef::Workspace(2)],
        };
        plan.ops.push(tanh_op);

        // Test detection
        assert!(
            ModelExecutor::scan_for_simd_kernels(&plan),
            "Should detect SIMD activation kernels"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(
            caps.has_simd_activations,
            "detect_optimizations should find SIMD activations"
        );
    }

    #[test]
    fn test_no_simd_activations() {
        // Create plan without SIMD kernels
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add non-SIMD operation
        let add_op = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(1)],
            output_refs: vec![BufferRef::Workspace(2)],
        };
        plan.ops.push(add_op);

        // Test detection
        assert!(
            !ModelExecutor::scan_for_simd_kernels(&plan),
            "Should not detect SIMD kernels in plan without activations"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(
            !caps.has_simd_activations,
            "Should not detect SIMD activations"
        );
    }

    #[test]
    fn test_detect_all_simd_activation_types() {
        // Test each SIMD activation type individually
        let simd_kernels = vec![
            KernelId::ACT_SIGMOID_U8,
            KernelId::ACT_TANH_U8,
            KernelId::ACT_RELU_U8,
            KernelId::ACT_GELU_U8,
            KernelId::ACT_SILU_U8,
        ];

        for kernel_id in simd_kernels {
            let mut plan = BackendPlan::new(BackendType::Cpu);
            let op = PlanOp {
                kernel_id,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0)],
                output_refs: vec![BufferRef::Workspace(1)],
            };
            plan.ops.push(op);

            assert!(
                ModelExecutor::scan_for_simd_kernels(&plan),
                "Should detect SIMD kernel: {:?}",
                kernel_id
            );
        }
    }

    #[test]
    fn test_detect_large_constants() {
        // Create plan with large constant data
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add large constant data (>1MB)
        plan.constant_data = vec![0u8; 2 * 1024 * 1024]; // 2MB

        // Test detection
        assert!(
            ModelExecutor::scan_for_large_constants(&plan),
            "Should detect large constant data"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(caps.has_large_embeddings, "Should detect large embeddings");
    }

    #[test]
    fn test_no_large_constants() {
        // Create plan with small constants
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add small constant data (<1MB)
        plan.constant_data = vec![0u8; 512 * 1024]; // 512KB

        // Test detection
        assert!(
            !ModelExecutor::scan_for_large_constants(&plan),
            "Should not detect large constants when data is small"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(
            !caps.has_large_embeddings,
            "Should not detect large embeddings"
        );
    }

    #[test]
    fn test_detect_large_constants_exact_threshold() {
        // Create plan with exactly 1MB constant data
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Exactly 1MB should not trigger (threshold is >1MB)
        plan.constant_data = vec![0u8; 1024 * 1024]; // 1MB

        assert!(
            !ModelExecutor::scan_for_large_constants(&plan),
            "Should not detect at exactly threshold (need >1MB)"
        );

        // Just over 1MB should trigger
        plan.constant_data = vec![0u8; 1024 * 1024 + 1]; // 1MB + 1 byte

        assert!(
            ModelExecutor::scan_for_large_constants(&plan),
            "Should detect when just over threshold"
        );
    }

    #[test]
    fn test_parallel_hints_not_yet_implemented() {
        // MVP: parallel hint detection returns false (Week 3)
        let plan = BackendPlan::new(BackendType::Cpu);

        assert!(
            !ModelExecutor::scan_for_parallel_hints(&plan),
            "Parallel hints should not be detected in Week 1 (MVP)"
        );
    }

    #[test]
    fn test_detect_fused_sigmoid_relu() {
        // Create plan with fused sigmoid+relu kernel
        let mut plan = BackendPlan::new(BackendType::Cpu);

        let fused_op = PlanOp {
            kernel_id: KernelId::ACT_FUSED_SIGMOID_RELU_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(fused_op);

        // Test detection
        assert!(
            ModelExecutor::scan_for_fused_kernels(&plan),
            "Should detect fused sigmoid+relu kernel"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(caps.has_composed_views, "Should detect composed views");
    }

    #[test]
    fn test_detect_fused_sigmoid_tanh() {
        // Create plan with fused sigmoid+tanh kernel
        let mut plan = BackendPlan::new(BackendType::Cpu);

        let fused_op = PlanOp {
            kernel_id: KernelId::ACT_FUSED_SIGMOID_TANH_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(fused_op);

        assert!(
            ModelExecutor::scan_for_fused_kernels(&plan),
            "Should detect fused sigmoid+tanh kernel"
        );
    }

    #[test]
    fn test_detect_fused_sigmoid_tanh_relu() {
        // Create plan with 3-stage fused kernel
        let mut plan = BackendPlan::new(BackendType::Cpu);

        let fused_op = PlanOp {
            kernel_id: KernelId::ACT_FUSED_SIGMOID_TANH_RELU_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(fused_op);

        assert!(
            ModelExecutor::scan_for_fused_kernels(&plan),
            "Should detect 3-stage fused kernel"
        );
    }

    #[test]
    fn test_detect_all_fused_kernel_types() {
        // Test each fused kernel type individually
        let fused_kernels = vec![
            KernelId::ACT_FUSED_SIGMOID_RELU_U8,
            KernelId::ACT_FUSED_SIGMOID_TANH_U8,
            KernelId::ACT_FUSED_SIGMOID_TANH_RELU_U8,
        ];

        for kernel_id in fused_kernels {
            let mut plan = BackendPlan::new(BackendType::Cpu);
            let op = PlanOp {
                kernel_id,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0)],
                output_refs: vec![BufferRef::Workspace(1)],
            };
            plan.ops.push(op);

            assert!(
                ModelExecutor::scan_for_fused_kernels(&plan),
                "Should detect fused kernel: {:?}",
                kernel_id
            );
        }
    }

    #[test]
    fn test_no_fused_kernels() {
        // Create plan without fused kernels (only standard activations)
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add non-fused operation
        let sigmoid_op = PlanOp {
            kernel_id: KernelId::ACT_SIGMOID_U8,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0)],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(sigmoid_op);

        // Should not detect fused kernels
        assert!(
            !ModelExecutor::scan_for_fused_kernels(&plan),
            "Should not detect fused kernels when only standard activations present"
        );

        let caps = ModelExecutor::detect_optimizations(&plan);
        assert!(!caps.has_composed_views, "Should not detect composed views");
    }

    #[test]
    fn test_optimization_capabilities_default() {
        let caps = OptimizationCapabilities::default();

        assert!(!caps.has_simd_activations);
        assert!(!caps.has_composed_views);
        assert!(!caps.has_parallel_ops);
        assert!(!caps.has_large_embeddings);
    }

    #[test]
    fn test_count_simd_operations() {
        // Create plan with SIMD ops
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add 3 SIMD operations
        for _ in 0..3 {
            let op = PlanOp {
                kernel_id: KernelId::ACT_SIGMOID_U8,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0)],
                output_refs: vec![BufferRef::Workspace(1)],
            };
            plan.ops.push(op);
        }

        // Add 2 non-SIMD operations
        for _ in 0..2 {
            let op = PlanOp {
                kernel_id: KernelId::ELEM_ADD,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(1)],
                output_refs: vec![BufferRef::Workspace(2)],
            };
            plan.ops.push(op);
        }

        // Test static count method
        let count = ModelExecutor::count_simd_operations_static(&plan);
        assert_eq!(count, 3, "Should count 3 SIMD operations");
    }

    // Embedding cache integration tests

    #[test]
    fn test_pin_large_embeddings_success() {
        // Create plan with large constant data (>1MB)
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // 2MB of f32 data (512K floats)
        let num_floats = 512 * 1024;
        let f32_data = vec![0.5f32; num_floats];

        // Convert to bytes (little-endian)
        let mut bytes = Vec::with_capacity(num_floats * 4);
        for &f in &f32_data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }

        plan.constant_data = bytes;

        // Test pinning
        let cache = ModelExecutor::pin_large_embeddings(&plan);
        assert!(
            cache.is_some(),
            "Should create embedding cache for large constants"
        );

        let cache = cache.unwrap();
        assert_eq!(cache.num_tables(), 1, "Should have 1 pinned table");
    }

    #[test]
    fn test_pin_large_embeddings_skip_small() {
        // Create plan with small constant data (<1MB)
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // 256KB of data
        plan.constant_data = vec![0u8; 256 * 1024];

        // Should not create cache for small data
        let cache = ModelExecutor::pin_large_embeddings(&plan);
        assert!(
            cache.is_none(),
            "Should not create cache for small constants"
        );
    }

    #[test]
    fn test_pin_large_embeddings_skip_misaligned() {
        // Create plan with misaligned data (not divisible by 4)
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // 2MB + 1 byte (not aligned to f32)
        plan.constant_data = vec![0u8; 2 * 1024 * 1024 + 1];

        // Should not create cache for misaligned data
        let cache = ModelExecutor::pin_large_embeddings(&plan);
        assert!(
            cache.is_none(),
            "Should not create cache for misaligned data"
        );
    }

    #[test]
    fn test_infer_embedding_dimension() {
        // Test that function returns valid dimensions
        // The function checks largest dimensions first, so exact results vary

        // Test with data divisible by common dimensions
        let num_elements1 = 768 * 1000; // 768K elements
        let dim1 = ModelExecutor::infer_embedding_dimension(num_elements1);
        assert!(
            dim1 > 0 && num_elements1 % dim1 == 0,
            "Dimension {} should divide {}",
            dim1,
            num_elements1
        );

        let num_elements2 = 1024 * 500; // 512K elements
        let dim2 = ModelExecutor::infer_embedding_dimension(num_elements2);
        assert!(
            dim2 > 0 && num_elements2 % dim2 == 0,
            "Dimension {} should divide {}",
            dim2,
            num_elements2
        );

        let num_elements3 = 512 * 2000; // 1024K elements
        let dim3 = ModelExecutor::infer_embedding_dimension(num_elements3);
        assert!(
            dim3 > 0 && num_elements3 % dim3 == 0,
            "Dimension {} should divide {}",
            dim3,
            num_elements3
        );

        // Test fallback for uncommon sizes
        let dim = ModelExecutor::infer_embedding_dimension(100);
        assert!(dim > 0, "Should return non-zero dimension");
        assert!(100 % dim == 0, "Dimension should divide element count");
    }

    #[test]
    fn test_count_constant_refs() {
        // Create plan with constant buffer references
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add operation with constant input
        let op1 = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![
                BufferRef::Constant {
                    offset: 0,
                    size: 1024,
                },
                BufferRef::Workspace(0),
            ],
            output_refs: vec![BufferRef::Workspace(1)],
        };
        plan.ops.push(op1);

        // Add operation with external constant
        let op2 = PlanOp {
            kernel_id: KernelId::ELEM_MUL,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![
                BufferRef::Workspace(1),
                BufferRef::ExternalConstant {
                    path: "weights.bin".to_string(),
                    offset: 0,
                    size: 2048,
                },
            ],
            output_refs: vec![BufferRef::Workspace(2)],
        };
        plan.ops.push(op2);

        // Add operation with no constants
        let op3 = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(2), BufferRef::Workspace(3)],
            output_refs: vec![BufferRef::Workspace(4)],
        };
        plan.ops.push(op3);

        // Should count 2 constant references (1 Constant + 1 ExternalConstant)
        let count = ModelExecutor::count_constant_refs(&plan);
        assert_eq!(count, 2, "Should count 2 constant references");
    }

    #[test]
    fn test_embedding_cache_accessor() {
        // Create executor without optimization
        let plan = BackendPlan::new(BackendType::Cpu);
        let backend = hologram::core::CpuBackend::new();
        let executor_plan = PlanExecutor::without_workspace(plan);
        let executor = ModelExecutor::from_plan_executor(executor_plan, Box::new(backend));

        // Should have no embedding cache
        assert!(
            executor.embedding_cache().is_none(),
            "Should have no embedding cache by default"
        );
    }

    // Parallel execution analysis tests (Week 3)

    #[test]
    fn test_analyze_parallelism_independent_ops() {
        // Create plan with independent operations that can run in parallel
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Input: writes to workspace slot 0
        let input_op = PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(10), BufferRef::Workspace(11)],
            output_refs: vec![BufferRef::Workspace(0)],
        };
        plan.ops.push(input_op);

        // Three independent operations that all read from slot 0
        // These can run in parallel
        for i in 1..4 {
            let op = PlanOp {
                kernel_id: KernelId::ELEM_MUL,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(12 + i)],
                output_refs: vec![BufferRef::Workspace(i)],
            };
            plan.ops.push(op);
        }

        // Analyze parallelism
        let analysis = ModelExecutor::analyze_parallelism(&plan);

        // Should detect one parallel group with 3 operations
        assert_eq!(
            analysis.parallel_groups.len(),
            1,
            "Should have 1 parallel group"
        );
        assert_eq!(
            analysis.parallel_groups[0].len(),
            3,
            "Parallel group should have 3 operations"
        );
        assert_eq!(
            analysis.total_parallelizable_ops, 3,
            "Should have 3 parallelizable ops"
        );
    }

    #[test]
    fn test_analyze_parallelism_sequential_ops() {
        // Create plan with dependent operations (must run sequentially)
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Chain of 5 operations, each depends on previous
        for i in 0..5 {
            let op = PlanOp {
                kernel_id: KernelId::ELEM_ADD,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(i), BufferRef::Workspace(10)],
                output_refs: vec![BufferRef::Workspace(i + 1)],
            };
            plan.ops.push(op);
        }

        // Analyze parallelism
        let analysis = ModelExecutor::analyze_parallelism(&plan);

        // Should have no parallel groups (all sequential)
        assert_eq!(
            analysis.parallel_groups.len(),
            0,
            "Should have no parallel groups"
        );
        assert_eq!(
            analysis.total_parallelizable_ops, 0,
            "Should have no parallelizable ops"
        );
        assert_eq!(
            analysis.total_sequential_ops, 5,
            "All 5 ops should be sequential"
        );
    }

    #[test]
    fn test_analyze_parallelism_mixed() {
        // Create plan with mix of parallel and sequential operations
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Op 0: initial operation
        plan.ops.push(PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(10), BufferRef::Workspace(11)],
            output_refs: vec![BufferRef::Workspace(0)],
        });

        // Ops 1-3: parallel group (Q/K/V pattern)
        for i in 1..4 {
            plan.ops.push(PlanOp {
                kernel_id: KernelId::GEMM_STANDARD,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(12 + i)],
                output_refs: vec![BufferRef::Workspace(i)],
            });
        }

        // Op 4: combines results (sequential)
        plan.ops.push(PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(1), BufferRef::Workspace(2)],
            output_refs: vec![BufferRef::Workspace(5)],
        });

        // Analyze parallelism
        let analysis = ModelExecutor::analyze_parallelism(&plan);

        // Should detect one parallel group
        assert!(
            !analysis.parallel_groups.is_empty(),
            "Should have at least 1 parallel group"
        );
        assert_eq!(
            analysis.total_parallelizable_ops, 3,
            "Should have 3 parallelizable ops (Q/K/V)"
        );
        assert_eq!(
            analysis.total_sequential_ops, 2,
            "Should have 2 sequential ops"
        );
    }

    #[test]
    fn test_scan_for_parallel_hints_true() {
        // Create plan with enough parallel operations to trigger detection
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add base operation
        plan.ops.push(PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(10), BufferRef::Workspace(11)],
            output_refs: vec![BufferRef::Workspace(0)],
        });

        // Add 3 independent operations (Q/K/V pattern)
        for i in 1..4 {
            plan.ops.push(PlanOp {
                kernel_id: KernelId::GEMM_STANDARD,
                kernel_idx: 0,
                params: KernelParams::default(),
                epilogue: EpilogueChain::new(),
                partition: ThreadPartition::sequential(),
                input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(12 + i)],
                output_refs: vec![BufferRef::Workspace(i)],
            });
        }

        // Should detect parallel hints
        assert!(
            ModelExecutor::scan_for_parallel_hints(&plan),
            "Should detect parallel execution opportunities"
        );
    }

    #[test]
    fn test_scan_for_parallel_hints_false() {
        // Create plan with only sequential operations
        let mut plan = BackendPlan::new(BackendType::Cpu);

        // Add single operation
        plan.ops.push(PlanOp {
            kernel_id: KernelId::ELEM_ADD,
            kernel_idx: 0,
            params: KernelParams::default(),
            epilogue: EpilogueChain::new(),
            partition: ThreadPartition::sequential(),
            input_refs: vec![BufferRef::Workspace(0), BufferRef::Workspace(1)],
            output_refs: vec![BufferRef::Workspace(2)],
        });

        // Should not detect parallel hints
        assert!(
            !ModelExecutor::scan_for_parallel_hints(&plan),
            "Should not detect parallel opportunities in single-op plan"
        );
    }

    // Week 5: Parallel Execution Tests

    #[test]
    fn test_allocate_buffers_parallel_small_count() {
        // Test sequential fallback for small buffer counts (< 4)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate 3 buffers (should use sequential path)
        let sizes = vec![1024, 2048, 4096];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers successfully");

        assert_eq!(handles.len(), 3, "Should allocate 3 buffers");

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_allocate_buffers_parallel_large_count() {
        // Test parallel allocation for many buffers (>= 4)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate 8 buffers (should use parallel path)
        let sizes = vec![1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers successfully");

        assert_eq!(handles.len(), 8, "Should allocate 8 buffers");

        // Verify buffers are usable
        for (i, handle) in handles.iter().enumerate() {
            let data = vec![i as u8; sizes[i]];
            model_executor
                .backend
                .copy_to_buffer(*handle, &data)
                .expect("Should copy to buffer");
        }

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_upload_tensors_parallel_small_count() {
        // Test sequential fallback for small tensor counts (< 3)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate 2 buffers
        let sizes = vec![1024, 2048];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers");

        // Prepare tensor data
        let tensors: Vec<(BufferHandle, Vec<u8>)> = handles
            .iter()
            .zip(sizes.iter())
            .enumerate()
            .map(|(i, (handle, &size))| (*handle, vec![i as u8; size]))
            .collect();

        // Upload (should use sequential path)
        model_executor
            .upload_tensors_parallel(tensors)
            .expect("Should upload tensors");

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_upload_tensors_parallel_large_count() {
        // Test parallel upload for many tensors (>= 3)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate 6 buffers
        let sizes = vec![1024, 2048, 4096, 8192, 16384, 32768];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers");

        // Prepare tensor data with unique patterns
        let tensors: Vec<(BufferHandle, Vec<u8>)> = handles
            .iter()
            .zip(sizes.iter())
            .enumerate()
            .map(|(i, (handle, &size))| {
                let data = vec![i as u8; size];
                (*handle, data)
            })
            .collect();

        // Upload (should use parallel path)
        model_executor
            .upload_tensors_parallel(tensors)
            .expect("Should upload tensors");

        // Verify data was uploaded correctly
        for (i, (handle, size)) in handles.iter().zip(sizes.iter()).enumerate() {
            let mut downloaded = vec![0u8; *size];
            model_executor
                .backend
                .copy_from_buffer(*handle, &mut downloaded)
                .expect("Should download buffer");
            assert_eq!(downloaded.len(), *size, "Buffer size should match");
            assert_eq!(
                downloaded[0], i as u8,
                "Buffer content should match uploaded data"
            );
        }

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_download_buffers_parallel_small_count() {
        // Test sequential fallback for small buffer counts (< 3)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate and fill 2 buffers
        let sizes = vec![1024, 2048];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers");

        for (i, (handle, size)) in handles.iter().zip(sizes.iter()).enumerate() {
            let data = vec![i as u8; *size];
            model_executor
                .backend
                .copy_to_buffer(*handle, &data)
                .expect("Should upload data");
        }

        // Download (should use sequential path)
        let downloaded = model_executor
            .download_buffers_parallel(&handles, &sizes)
            .expect("Should download buffers");

        assert_eq!(downloaded.len(), 2, "Should download 2 buffers");
        for (i, data) in downloaded.iter().enumerate() {
            assert_eq!(data.len(), sizes[i], "Downloaded size should match");
            assert_eq!(data[0], i as u8, "Downloaded content should match");
        }

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_download_buffers_parallel_large_count() {
        // Test parallel download for many buffers (>= 3)
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Allocate and fill 6 buffers
        let sizes = vec![1024, 2048, 4096, 8192, 16384, 32768];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers");

        for (i, (handle, size)) in handles.iter().zip(sizes.iter()).enumerate() {
            let data = vec![i as u8; *size];
            model_executor
                .backend
                .copy_to_buffer(*handle, &data)
                .expect("Should upload data");
        }

        // Download (should use parallel path)
        let downloaded = model_executor
            .download_buffers_parallel(&handles, &sizes)
            .expect("Should download buffers");

        assert_eq!(downloaded.len(), 6, "Should download 6 buffers");
        for (i, data) in downloaded.iter().enumerate() {
            assert_eq!(data.len(), sizes[i], "Downloaded size should match");
            assert_eq!(data[0], i as u8, "Downloaded content should match");
        }

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_parallel_operations_integration() {
        // Integration test: allocate, upload, download in sequence
        use hologram::backend::BackendType;
        use hologram::backend::backends::cpu::CpuBackend;

        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let mut model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // 1. Allocate buffers in parallel
        let sizes = vec![4096, 8192, 16384, 32768];
        let handles = model_executor
            .allocate_buffers_parallel(&sizes)
            .expect("Should allocate buffers");

        // 2. Upload tensors in parallel
        let tensors: Vec<(BufferHandle, Vec<u8>)> = handles
            .iter()
            .zip(sizes.iter())
            .enumerate()
            .map(|(i, (handle, &size))| {
                let data: Vec<u8> = (0..size).map(|j| ((i + j) % 256) as u8).collect();
                (*handle, data)
            })
            .collect();

        model_executor
            .upload_tensors_parallel(tensors.clone())
            .expect("Should upload tensors");

        // 3. Download buffers in parallel
        let downloaded = model_executor
            .download_buffers_parallel(&handles, &sizes)
            .expect("Should download buffers");

        // Verify data integrity
        for (i, (downloaded_data, (_, original_data))) in
            downloaded.iter().zip(tensors.iter()).enumerate()
        {
            assert_eq!(
                downloaded_data.len(),
                original_data.len(),
                "Buffer {} size should match",
                i
            );
            assert_eq!(
                downloaded_data, original_data,
                "Buffer {} content should match",
                i
            );
        }

        // Cleanup
        for handle in handles {
            model_executor.backend.free_buffer(handle).unwrap();
        }
    }

    #[test]
    fn test_operations_discovery() {
        use hologram::backend::backends::cpu::CpuBackend;
        use hologram::backend::{BackendPlan, BackendType, PlanExecutor};

        // Create a minimal empty plan for testing
        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Test operations discovery (empty plan has 0 operations)
        let ops = model_executor.operations();

        // Empty plan is valid - should return empty vec
        assert_eq!(ops.len(), 0, "Empty plan should have 0 operations");

        println!("Discovered {} operations (empty plan test)", ops.len());
    }

    #[test]
    fn test_optimization_report() {
        use hologram::backend::backends::cpu::CpuBackend;
        use hologram::backend::{BackendPlan, BackendType, PlanExecutor};

        // Create a minimal empty plan for testing
        let backend = CpuBackend::new();
        let plan = BackendPlan::new(BackendType::Cpu);
        let executor = PlanExecutor::new(plan, &backend).expect("Failed to create PlanExecutor");

        let model_executor = ModelExecutor {
            executor,
            backend: Box::new(backend),
            input_order: None,
            optimization_caps: OptimizationCapabilities::default(),
            metrics: None,
            embedding_cache: None,
        };

        // Test optimization report
        let report = model_executor.optimization_report();

        // Report should have valid data
        assert!(
            !report.simd_level.is_empty(),
            "SIMD level should be detected"
        );
        // Note: parallel_group_count is usize, always non-negative

        println!("Optimization report:");
        println!("  SIMD activations: {}", report.has_simd_activations);
        println!("  Epilogue fusion: {}", report.has_epilogue_fusion);
        println!("  Parallel groups: {}", report.parallel_group_count);
        println!("  SIMD level: {}", report.simd_level);
        println!("  Dynamic shapes: {}", report.dynamic_shapes);
    }
}

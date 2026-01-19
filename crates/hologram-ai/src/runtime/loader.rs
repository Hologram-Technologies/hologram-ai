//! Loading .holo files to executable BackendPlan.
//!
//! This module provides functions for loading compiled .holo files and optionally
//! accompanying .weights files for memory-mapped weight access.
//!
//! # Loading Modes
//!
//! ## Unified Bundle (HOLB) - Recommended
//! Use [`load_holo_auto`] to automatically detect and load any format.
//! Unified bundles embed weights in the same file with page-aligned mmap access.
//!
//! ## Embedded Weights (legacy HOLP)
//! Use [`load_and_compile_holo`] when weights are embedded in the .holo file.
//!
//! ## External Weights (legacy HOLP + .weights)
//! Use [`load_with_external_weights`] when weights are stored separately in a .weights file.
//! This enables lazy loading of large weights (GB-sized) via memory mapping.

use anyhow::Result;
use hologram::backend::core::mapped_input::MappedInput;
use hologram::backend::executor::PlanExecutor;
use hologram::compiler::api::read_holo_from_bytes_with_header;
#[cfg(unix)]
use libc;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "onnx")]
use hologram_ai_onnx::core::sections::InputOrderSection;
#[cfg(feature = "onnx")]
use hologram_ai_onnx::core::{HoloFormat, PipelineBundleReader, UnifiedBundleReader};

/// Load a .holo file to an executable BackendPlan.
///
/// This function loads a pre-compiled .holo file and prepares it for execution:
/// 1. Read and deserialize .holo file using hologram's runtime API
/// 2. Resolve kernel IDs to function pointers based on CPU capabilities
/// 3. Create appropriate backend (with fallback to CPU if needed)
///
/// # Arguments
/// * `path` - Path to .holo file
///
/// # Returns
/// Tuple of (BackendPlan, ProgramBackend) ready for execution
///
/// # Errors
/// Returns error if:
/// - File cannot be read
/// - Deserialization fails (corrupted .holo file)
/// - Backend creation fails
#[tracing::instrument(
    name = "load_and_compile_holo",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_and_compile_holo(
    path: &Path,
) -> Result<(
    hologram::backend::BackendPlan,
    Box<dyn hologram::backend::ProgramBackend>,
)> {
    // Phase 1: Read and deserialize
    let plan = {
        let _span = tracing::info_span!("deserialize_holo").entered();
        hologram::compiler::read_holo(path)
            .map_err(|e| anyhow::anyhow!("Failed to load .holo file: {:?}", e))?
    };

    tracing::debug!("Deserialized BackendPlan from .holo file");

    // Phase 2: Create backend
    let backend = {
        let _span = tracing::info_span!(
            "create_backend",
            backend_type = ?plan.backend_type
        )
        .entered();
        match hologram::backend::create_backend(plan.backend_type.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
                hologram::backend::create_best_backend()
            }
        }
    };

    tracing::info!(backend = ?backend.backend_type(), "Successfully loaded BackendPlan");

    Ok((plan, backend))
}

/// Load a .holo file with external memory-mapped weights.
///
/// This function loads a .holo file and creates a `PlanExecutor` that uses
/// memory-mapped access to the external weights file. This enables lazy loading
/// of large weights (GB-sized) without loading them all into memory at once.
///
/// # Arguments
/// * `holo_path` - Path to the .holo file
/// * `weights_path` - Path to the .weights file (memory-mapped)
///
/// # Returns
/// Tuple of (PlanExecutor, ProgramBackend) ready for execution
///
/// # Errors
/// Returns error if:
/// - .holo file cannot be read or deserialized
/// - .weights file cannot be memory-mapped
/// - Backend creation fails
///
/// # Example
///
/// ```rust,ignore
/// use std::path::Path;
///
/// let (executor, backend) = load_with_external_weights(
///     Path::new("model.holo"),
///     Path::new("model.weights"),
/// )?;
///
/// // Execute inference
/// executor.execute(&inputs, &mut outputs, &mut *backend)?;
/// ```
#[tracing::instrument(
    name = "load_with_external_weights",
    skip_all,
    fields(
        holo_path = %holo_path.display(),
        weights_path = %weights_path.display()
    )
)]
pub fn load_with_external_weights(
    holo_path: &Path,
    weights_path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    // Phase 1: Deserialize .holo file
    let plan = {
        let _span = tracing::info_span!("deserialize_holo").entered();
        hologram::compiler::read_holo(holo_path)
            .map_err(|e| anyhow::anyhow!("Failed to load .holo file: {:?}", e))?
    };

    tracing::debug!(
        constant_data_bytes = plan.constant_data.len(),
        "Deserialized BackendPlan (will use mmap instead)"
    );

    // Phase 2: Create backend
    let backend = {
        let _span = tracing::info_span!(
            "create_backend",
            backend_type = ?plan.backend_type
        )
        .entered();
        match hologram::backend::create_backend(plan.backend_type.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
                hologram::backend::create_best_backend()
            }
        }
    };

    tracing::info!(backend = ?backend.backend_type(), "Using backend");

    // Phase 3: Create executor with mmap'd weights
    let executor = {
        let _span = tracing::info_span!("create_executor_mmap").entered();
        PlanExecutor::with_external_constants(plan, &*backend, weights_path).map_err(|e| {
            anyhow::anyhow!("Failed to create executor with external weights: {:?}", e)
        })?
    };

    tracing::info!("Successfully loaded model with external weights");

    Ok((executor, backend))
}

/// Load a .holo file with optional external weights.
///
/// This is a convenience function that automatically selects the appropriate
/// loading strategy based on whether a weights file exists.
///
/// # Arguments
/// * `holo_path` - Path to the .holo file
/// * `weights_path` - Optional path to the .weights file
///
/// # Returns
/// Tuple of (PlanExecutor, ProgramBackend) ready for execution
///
/// # Example
///
/// ```rust,ignore
/// use std::path::Path;
///
/// // Load with embedded weights
/// let (executor, backend) = load_holo_file(
///     Path::new("small_model.holo"),
///     None,
/// )?;
///
/// // Load with external weights
/// let (executor, backend) = load_holo_file(
///     Path::new("large_model.holo"),
///     Some(Path::new("large_model.weights")),
/// )?;
/// ```
pub fn load_holo_file(
    holo_path: &Path,
    weights_path: Option<&Path>,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    if let Some(wp) = weights_path {
        load_with_external_weights(holo_path, wp)
    } else {
        // Load with embedded weights, then wrap in executor
        let (plan, backend) = load_and_compile_holo(holo_path)?;
        let executor = PlanExecutor::new(plan, &*backend)
            .map_err(|e| anyhow::anyhow!("Failed to create executor: {:?}", e))?;
        Ok((executor, backend))
    }
}

/// Automatically detect file format and load appropriately.
///
/// This function auto-detects whether the file is:
/// - **Unified Bundle (HOLB)**: Single file with embedded weights (mmap'd)
/// - **Legacy Plan (HOLP)**: Separate .holo + optional .weights files
///
/// For unified bundles, the weights section is memory-mapped directly from the
/// bundle file at the page-aligned offset.
///
/// For legacy format, checks for a `.weights` file with the same stem.
///
/// # Arguments
/// * `path` - Path to the .holo file
///
/// # Returns
/// Tuple of (PlanExecutor, ProgramBackend) ready for execution
///
/// # Example
///
/// ```rust,ignore
/// use std::path::Path;
///
/// // Automatically handles any format
/// let (executor, backend) = load_holo_auto(Path::new("model.holo"))?;
/// executor.execute(&inputs, &mut outputs, &mut *backend)?;
/// ```
#[tracing::instrument(
    name = "load_holo_auto",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_holo_auto(
    path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    // Phase 1: Detect format
    let (format, magic) = {
        let _span = tracing::info_span!("detect_format").entered();
        let mut file = File::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", path.display(), e))?;

        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|e| anyhow::anyhow!("Failed to read magic bytes: {}", e))?;
        drop(file);

        (HoloFormat::detect(&magic), magic)
    };

    tracing::info!(format = ?format, "Detected file format");

    // Phase 2: Route to appropriate loader
    match format {
        HoloFormat::Bundle => load_unified_bundle(path),
        HoloFormat::Pipeline => Err(anyhow::anyhow!(
            "Pipeline bundle detected: {}. Use load_pipeline_bundle() and specify model name.",
            path.display()
        )),
        HoloFormat::Plan | HoloFormat::Legacy => {
            let weights_path = path.with_extension("weights");
            if weights_path.exists() {
                tracing::info!(weights_path = %weights_path.display(), "Found external weights file");
                load_with_external_weights(path, &weights_path)
            } else {
                load_holo_file(path, None)
            }
        }
        HoloFormat::Unknown => Err(anyhow::anyhow!(
            "Unknown file format: {:?} (magic: {:?})",
            path.display(),
            magic
        )),
    }
}

/// Load a unified bundle file (HOLB format).
///
/// The bundle contains both the computation graph and weights in a single file.
/// Weights are memory-mapped from the page-aligned section within the bundle.
#[tracing::instrument(
    name = "load_unified_bundle",
    skip_all,
    fields(path = %path.display())
)]
fn load_unified_bundle(
    path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    // Phase 1: Memory-map the bundle
    let mmap = {
        let _span = tracing::info_span!("mmap_bundle").entered();
        let m = MappedInput::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to mmap bundle '{}': {}", path.display(), e))?;
        advise_sequential(&m);
        m
    };

    let mmap_size = mmap.as_slice().len();
    tracing::debug!(mmap_size_bytes = mmap_size, "Memory-mapped bundle");

    // Phase 2: Parse header and verify checksums
    let (plan, weights_offset, _graph_size, weights_size) = {
        let _span = tracing::info_span!("parse_bundle_header").entered();

        let reader = UnifiedBundleReader::from_bytes(mmap.as_slice())
            .map_err(|e| anyhow::anyhow!("Failed to parse bundle header: {:?}", e))?;

        // Phase 2a: Verify checksums
        {
            let _checksum_span = tracing::info_span!("verify_checksums").entered();
            if !reader.verify_checksums() {
                return Err(anyhow::anyhow!("Bundle checksum verification failed"));
            }
        }

        let graph_size = reader.graph_size();
        let weights_size = reader.weights_size();
        let weights_offset = reader.weights_mmap_offset();

        tracing::info!(
            graph_bytes = graph_size,
            weights_bytes = weights_size,
            weights_offset = weights_offset.unwrap_or(0),
            "Bundle sections parsed"
        );

        // Phase 2b: Deserialize graph
        let plan = {
            let _deser_span =
                tracing::info_span!("deserialize_graph", graph_bytes = graph_size).entered();
            hologram::compiler::read_holo_from_bytes(reader.graph_bytes())
                .map_err(|e| anyhow::anyhow!("Failed to deserialize graph from bundle: {:?}", e))?
        };

        (plan, weights_offset, graph_size, weights_size)
    }; // reader goes out of scope here, releasing borrow of mmap

    // Phase 3: Create backend
    let backend = {
        let _span = tracing::info_span!(
            "create_backend",
            backend_type = ?plan.backend_type
        )
        .entered();
        match hologram::backend::create_backend(plan.backend_type.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
                hologram::backend::create_best_backend()
            }
        }
    };

    tracing::info!(backend = ?backend.backend_type(), "Using backend");

    // Phase 4: Create executor with mmap'd weights
    let mmap_arc = Arc::new(mmap);
    let executor = {
        let _span = tracing::info_span!(
            "create_executor",
            has_weights = weights_offset.is_some(),
            weights_size = weights_size
        )
        .entered();

        if let Some(offset) = weights_offset {
            PlanExecutor::with_mmap_constants_at_offset(plan, &*backend, mmap_arc, offset).map_err(
                |e| anyhow::anyhow!("Failed to create executor with bundle weights: {:?}", e),
            )?
        } else {
            PlanExecutor::new(plan, &*backend)
                .map_err(|e| anyhow::anyhow!("Failed to create executor: {:?}", e))?
        }
    };

    tracing::info!("Successfully loaded unified bundle");

    Ok((executor, backend))
}

// =============================================================================
// Pipeline Bundle Loading (HOLM format)
// =============================================================================

/// A loaded pipeline bundle that provides access to multiple models.
///
/// The bundle is memory-mapped, and individual models can be loaded on demand.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = load_pipeline_bundle(Path::new("t5-pipeline.holo"))?;
/// println!("Models: {:?}", pipeline.model_names());
///
/// let (encoder_exec, encoder_backend) = pipeline.load_model("encoder")?;
/// let (decoder_exec, decoder_backend) = pipeline.load_model("decoder")?;
/// ```
pub struct PipelineBundle {
    /// Memory-mapped pipeline file
    mmap: Arc<MappedInput>,
    /// Parsed pipeline header and index (stores entry info only, not full reader)
    model_info: Vec<(String, usize, usize)>, // (name, offset, size)
}

impl PipelineBundle {
    /// Get the list of model names in the pipeline.
    pub fn model_names(&self) -> Vec<&str> {
        self.model_info.iter().map(|(n, _, _)| n.as_str()).collect()
    }

    /// Get the number of models in the pipeline.
    pub fn model_count(&self) -> usize {
        self.model_info.len()
    }

    /// Check if a model exists in the pipeline.
    pub fn has_model(&self, name: &str) -> bool {
        self.model_info.iter().any(|(n, _, _)| n == name)
    }

    /// Prefetch a model's weights into memory.
    ///
    /// Call this before `load_model()` to overlap I/O with computation.
    /// For example, while executing layer N, call `prefetch_model("layer.N+1")`
    /// to hint the OS to start loading the next layer's weights.
    ///
    /// This uses `madvise(MADV_WILLNEED)` on Unix systems to trigger
    /// read-ahead without blocking.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the model to prefetch
    ///
    /// # Errors
    ///
    /// Returns error if the model name is not found in the pipeline.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = load_pipeline_bundle(Path::new("model.holo"))?;
    ///
    /// for (i, name) in pipeline.model_names().iter().enumerate() {
    ///     // Prefetch next model while executing current
    ///     if i + 1 < pipeline.model_count() {
    ///         pipeline.prefetch_model(&pipeline.model_names()[i + 1])?;
    ///     }
    ///
    ///     let (executor, backend) = pipeline.load_model(name)?;
    ///     // ... execute model ...
    /// }
    /// ```
    pub fn prefetch_model(&self, name: &str) -> Result<()> {
        let (_, model_offset, model_size) = self
            .model_info
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in pipeline", name))?;

        tracing::debug!(
            "Prefetching model '{}' at offset {}, size {}",
            name,
            model_offset,
            model_size
        );

        // Advise the OS to prefetch this range
        advise_willneed_range(&self.mmap, *model_offset, *model_size);

        Ok(())
    }

    /// Release a model's weights from memory.
    ///
    /// Call this after a model is no longer needed to reduce memory pressure.
    /// This hints to the OS that the pages can be freed, which is especially
    /// useful in layer-by-layer execution where previous layers won't be
    /// accessed again.
    ///
    /// This uses `madvise(MADV_DONTNEED)` on Unix systems.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the model to release
    ///
    /// # Errors
    ///
    /// Returns error if the model name is not found in the pipeline.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = load_pipeline_bundle(Path::new("model.holo"))?;
    ///
    /// for (i, name) in pipeline.model_names().iter().enumerate() {
    ///     let (executor, backend) = pipeline.load_model(name)?;
    ///     // ... execute model ...
    ///
    ///     // Release previous model's memory
    ///     if i > 0 {
    ///         pipeline.release_model(&pipeline.model_names()[i - 1])?;
    ///     }
    /// }
    /// ```
    pub fn release_model(&self, name: &str) -> Result<()> {
        let (_, model_offset, model_size) = self
            .model_info
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in pipeline", name))?;

        tracing::debug!(
            "Releasing model '{}' at offset {}, size {}",
            name,
            model_offset,
            model_size
        );

        // Advise the OS that we're done with this range
        advise_dontneed_range(&self.mmap, *model_offset, *model_size);

        Ok(())
    }

    /// Load a model from the pipeline by name.
    ///
    /// The model's weights are memory-mapped from within the pipeline file
    /// at the correct offset.
    pub fn load_model(
        &self,
        name: &str,
    ) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
        // Find the model entry
        let (_, model_offset, model_size) = self
            .model_info
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in pipeline", name))?;

        tracing::info!(
            "Loading model '{}' from pipeline at offset {}, size {}",
            name,
            model_offset,
            model_size
        );

        // Get the model bytes (HOLB format)
        let model_bytes = &self.mmap.as_slice()[*model_offset..*model_offset + *model_size];

        // Parse as HOLB bundle
        let model_reader = UnifiedBundleReader::from_bytes(model_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse model '{}' as HOLB: {:?}", name, e))?;

        // Verify checksums
        if !model_reader.verify_checksums() {
            return Err(anyhow::anyhow!(
                "Model '{}' checksum verification failed",
                name
            ));
        }

        // Deserialize the graph
        let plan =
            hologram::compiler::read_holo_from_bytes(model_reader.graph_bytes()).map_err(|e| {
                anyhow::anyhow!("Failed to deserialize model '{}' graph: {:?}", name, e)
            })?;

        // Create backend
        let backend = match hologram::backend::create_backend(plan.backend_type.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!(
                    "Failed to create backend for '{}': {}. Falling back to CPU",
                    name,
                    e
                );
                hologram::backend::create_best_backend()
            }
        };

        // Create executor with mmap'd weights
        // The weights offset is relative to the start of the HOLB section
        let executor = if let Some(weights_offset_in_holb) = model_reader.weights_mmap_offset() {
            // Calculate absolute offset in the pipeline file
            let absolute_weights_offset = *model_offset + weights_offset_in_holb;
            tracing::info!(
                "WEIGHTS OFFSET DEBUG: model='{}', model_offset={}, weights_offset_in_holb={}, absolute_weights_offset={}",
                name,
                model_offset,
                weights_offset_in_holb,
                absolute_weights_offset
            );
            PlanExecutor::with_mmap_constants_at_offset(
                plan,
                &*backend,
                Arc::clone(&self.mmap),
                absolute_weights_offset,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create executor for '{}' with weights: {:?}",
                    name,
                    e
                )
            })?
        } else {
            PlanExecutor::new(plan, &*backend)
                .map_err(|e| anyhow::anyhow!("Failed to create executor for '{}': {:?}", name, e))?
        };

        tracing::info!("Successfully loaded model '{}' from pipeline", name);

        Ok((executor, backend))
    }

    fn select_entry_layer(
        header: &hologram::compiler::format::LayerHeaderData,
    ) -> Option<&hologram::compiler::format::LayerDescriptorData> {
        if let Some(first_level) = header.schedule.first()
            && let Some(first_id) = first_level.first()
            && let Some(layer) = header.layer(*first_id)
        {
            return Some(layer);
        }

        header.layers.first()
    }

    /// Pin a model's weights in RAM for low-latency access.
    ///
    /// This locks the model's memory-mapped weights in RAM using `mlock()`,
    /// preventing them from being swapped out. This is useful for embedding
    /// tables and frequently-accessed layers in latency-critical applications.
    ///
    /// **Important**: Only use this for small models (<64MB) to avoid
    /// exhausting system memory. Larger models should use prefetching instead.
    ///
    /// # Arguments
    ///
    /// * `name` - Name of the model to pin
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Model name not found
    /// - Model size exceeds 64MB (safety limit)
    /// - Insufficient memory lock limit (check `ulimit -l`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = load_pipeline_bundle(Path::new("model.holo"))?;
    ///
    /// // Pin embedding table for zero-latency lookup
    /// if pipeline.is_embedding_layer("embedding") {
    ///     pipeline.pin_model_weights("embedding")?;
    /// }
    ///
    /// // Load and use the model
    /// let (executor, backend) = pipeline.load_model("embedding")?;
    /// ```
    pub fn pin_model_weights(&self, name: &str) -> Result<()> {
        let (_, model_offset, model_size) = self
            .model_info
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in pipeline", name))?;

        // Safety check: only pin small models (<64MB)
        const MAX_PIN_SIZE: usize = 64 * 1024 * 1024; // 64MB
        if *model_size > MAX_PIN_SIZE {
            return Err(anyhow::anyhow!(
                "Model '{}' size ({} bytes) exceeds max pinnable size ({} bytes). Use prefetching instead.",
                name,
                model_size,
                MAX_PIN_SIZE
            ));
        }

        tracing::info!(
            "Pinning model '{}' weights in RAM (size: {} bytes)",
            name,
            model_size
        );

        // First hint that we'll need this data
        advise_willneed_range(&self.mmap, *model_offset, *model_size);

        // Then lock it in RAM
        lock_memory_range(&self.mmap, *model_offset, *model_size)?;

        tracing::debug!("Successfully pinned model '{}' in RAM", name);

        Ok(())
    }

    /// Check if a model name suggests it's an embedding layer.
    ///
    /// This heuristic checks if the model name contains common embedding
    /// layer keywords: "embed", "embedding", "token", "vocab".
    ///
    /// # Arguments
    ///
    /// * `name` - Model name to check
    ///
    /// # Returns
    ///
    /// `true` if the name suggests an embedding layer.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use hologram_ai::runtime::loader::PipelineBundle;
    /// // These return true:
    /// // - "embedding"
    /// // - "token_embeddings"
    /// // - "vocab_embed"
    /// // - "embed_tokens"
    /// ```
    pub fn is_embedding_layer(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        lower.contains("embed") || lower.contains("token") || lower.contains("vocab")
    }

    /// Load a model from the pipeline and extract its input order if available.
    ///
    /// This uses the embedded layer header to preserve compiler input ordering
    /// when constructing a `ModelExecutor`.
    #[allow(clippy::type_complexity)]
    pub fn load_model_with_inputs(
        &self,
        name: &str,
    ) -> Result<(
        PlanExecutor,
        Box<dyn hologram::backend::ProgramBackend>,
        Option<Vec<String>>,
    )> {
        // Find the model entry
        let (_, model_offset, model_size) = self
            .model_info
            .iter()
            .find(|(n, _, _)| n == name)
            .ok_or_else(|| anyhow::anyhow!("Model '{}' not found in pipeline", name))?;

        tracing::info!(
            "Loading model '{}' from pipeline at offset {}, size {}",
            name,
            model_offset,
            model_size
        );

        // Get the model bytes (HOLB format)
        let model_bytes = &self.mmap.as_slice()[*model_offset..*model_offset + *model_size];

        // Parse as HOLB bundle
        let model_reader = UnifiedBundleReader::from_bytes(model_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse model '{}' as HOLB: {:?}", name, e))?;

        // Verify checksums
        if !model_reader.verify_checksums() {
            return Err(anyhow::anyhow!(
                "Model '{}' checksum verification failed",
                name
            ));
        }

        // Deserialize the graph with optional header
        let (plan, header) =
            read_holo_from_bytes_with_header(model_reader.graph_bytes()).map_err(|e| {
                anyhow::anyhow!("Failed to deserialize model '{}' graph: {:?}", name, e)
            })?;

        let input_order = model_reader
            .get_section::<InputOrderSection>()
            .map(|section| section.inputs)
            .or_else(|| {
                header.as_ref().and_then(|data| {
                    let plan_inputs = plan.layout_metadata.num_inputs;

                    if let Some(layer) = Self::select_entry_layer(data)
                        && layer.inputs.len() == plan_inputs
                    {
                        return Some(
                            layer
                                .inputs
                                .iter()
                                .map(|port| port.name.clone())
                                .collect::<Vec<String>>(),
                        );
                    }

                    for layer in &data.layers {
                        if layer.inputs.len() == plan_inputs {
                            return Some(
                                layer
                                    .inputs
                                    .iter()
                                    .map(|port| port.name.clone())
                                    .collect::<Vec<String>>(),
                            );
                        }
                    }

                    None
                })
            });

        // Create backend
        let backend = match hologram::backend::create_backend(plan.backend_type.clone()) {
            Ok(backend) => backend,
            Err(e) => {
                tracing::warn!(
                    "Failed to create backend for '{}': {}. Falling back to CPU",
                    name,
                    e
                );
                hologram::backend::create_best_backend()
            }
        };

        // Create executor with mmap'd weights
        let executor = if let Some(weights_offset_in_holb) = model_reader.weights_mmap_offset() {
            // Calculate absolute offset in the pipeline file
            let absolute_weights_offset = *model_offset + weights_offset_in_holb;
            tracing::info!(
                "WEIGHTS OFFSET DEBUG: model='{}', model_offset={}, weights_offset_in_holb={}, absolute_weights_offset={}",
                name,
                model_offset,
                weights_offset_in_holb,
                absolute_weights_offset
            );
            PlanExecutor::with_mmap_constants_at_offset(
                plan,
                &*backend,
                Arc::clone(&self.mmap),
                absolute_weights_offset,
            )
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to create executor for '{}' with weights: {:?}",
                    name,
                    e
                )
            })?
        } else {
            PlanExecutor::new(plan, &*backend)
                .map_err(|e| anyhow::anyhow!("Failed to create executor for '{}': {:?}", name, e))?
        };

        tracing::info!("Successfully loaded model '{}' from pipeline", name);

        Ok((executor, backend, input_order))
    }
}

/// Load a pipeline bundle file (HOLM format).
///
/// The pipeline file is memory-mapped, and individual models can be loaded
/// on demand using `PipelineBundle::load_model()`.
///
/// # Arguments
/// * `path` - Path to the .holo pipeline bundle file
///
/// # Returns
/// A `PipelineBundle` that provides access to individual models.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = load_pipeline_bundle(Path::new("t5-pipeline.holo"))?;
///
/// // Load models as needed
/// let (encoder_exec, encoder_backend) = pipeline.load_model("encoder")?;
/// let (decoder_exec, decoder_backend) = pipeline.load_model("decoder")?;
/// ```
#[tracing::instrument(
    name = "load_pipeline_bundle",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_pipeline_bundle(path: &Path) -> Result<PipelineBundle> {
    // Phase 1: Memory-map the pipeline
    let mmap = {
        let _span = tracing::info_span!("mmap_pipeline").entered();
        MappedInput::open(path).map_err(|e| {
            anyhow::anyhow!("Failed to mmap pipeline bundle '{}': {}", path.display(), e)
        })?
    };

    let mmap_size = mmap.as_slice().len();
    tracing::debug!(mmap_size_bytes = mmap_size, "Memory-mapped pipeline");

    // Phase 2: Parse header and verify checksum
    let model_info = {
        let _span = tracing::info_span!("parse_pipeline_header").entered();

        let reader = PipelineBundleReader::from_bytes(mmap.as_slice())
            .map_err(|e| anyhow::anyhow!("Failed to parse pipeline bundle header: {:?}", e))?;

        // Phase 2a: Verify checksum
        {
            let _checksum_span = tracing::info_span!("verify_index_checksum").entered();
            if !reader.verify_index_checksum() {
                return Err(anyhow::anyhow!(
                    "Pipeline index checksum verification failed"
                ));
            }
        }

        let model_count = reader.model_count();
        let model_names = reader.model_names();

        tracing::info!(
            model_count = model_count,
            models = ?model_names,
            "Pipeline bundle loaded"
        );

        // Extract model info
        reader
            .model_names()
            .iter()
            .filter_map(|name| {
                reader
                    .get_entry(name)
                    .map(|entry| (name.to_string(), entry.offset as usize, entry.size as usize))
            })
            .collect()
    };

    Ok(PipelineBundle {
        mmap: Arc::new(mmap),
        model_info,
    })
}

/// Check if a file is a pipeline bundle.
pub fn is_pipeline_bundle(path: &Path) -> Result<bool> {
    let mut file = File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", path.display(), e))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| anyhow::anyhow!("Failed to read magic bytes: {}", e))?;

    Ok(HoloFormat::detect(&magic).is_pipeline())
}

// =============================================================================
// Execution Mode Configuration
// =============================================================================

/// Execution mode for model loading.
///
/// Controls how models are loaded into memory:
/// - **FullLoading**: Load entire model at once (fast execution, high memory)
/// - **LayerByLayer**: Load and execute one layer at a time (slow, low memory)
/// - **Auto**: Automatically select based on model size and available memory
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai::runtime::loader::{ExecutionMode, LoadOptions};
///
/// // For interactive chatbots with small models
/// let options = LoadOptions {
///     mode: ExecutionMode::FullLoading,
///     ..Default::default()
/// };
///
/// // For large models on constrained memory
/// let options = LoadOptions {
///     mode: ExecutionMode::LayerByLayer { prefetch: true },
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionMode {
    /// Load entire model at once (fast, high memory).
    ///
    /// Best for:
    /// - Interactive chatbots requiring low latency
    /// - Small to medium models that fit in memory
    /// - High-throughput serving environments
    FullLoading,

    /// Load and execute one layer at a time (slow, low memory).
    ///
    /// Best for:
    /// - Large models (70B+) that don't fit in memory
    /// - Memory-constrained inference environments
    /// - Batch/offline processing where latency isn't critical
    LayerByLayer {
        /// Enable prefetching of the next layer while executing current.
        /// Uses `madvise(MADV_WILLNEED)` to overlap I/O with compute.
        prefetch: bool,
    },

    /// Automatically select mode based on model size and available memory.
    ///
    /// The runtime will choose FullLoading if the model fits comfortably
    /// in available memory (< 50% of free memory), otherwise LayerByLayer.
    #[default]
    Auto,
}

/// Options for model loading.
///
/// Provides fine-grained control over how models are loaded and executed.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai::runtime::loader::{ExecutionMode, LoadOptions};
///
/// let options = LoadOptions {
///     mode: ExecutionMode::FullLoading,
///     memory_limit_mb: Some(8 * 1024), // 8GB limit
///     enable_prefetch: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Execution mode (default: Auto).
    pub mode: ExecutionMode,

    /// Memory limit in MB (for Auto mode decision).
    /// If None, uses system available memory.
    pub memory_limit_mb: Option<usize>,

    /// Enable madvise hints for prefetching (default: true).
    /// Always beneficial for sequential access patterns.
    pub enable_prefetch: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Auto,
            memory_limit_mb: None,
            enable_prefetch: true,
        }
    }
}

impl LoadOptions {
    /// Create options for full loading (optimal for small models).
    pub fn full_loading() -> Self {
        Self {
            mode: ExecutionMode::FullLoading,
            ..Default::default()
        }
    }

    /// Create options for layer-by-layer loading (optimal for large models).
    pub fn layer_by_layer() -> Self {
        Self {
            mode: ExecutionMode::LayerByLayer { prefetch: true },
            ..Default::default()
        }
    }

    /// Create options with a memory limit.
    pub fn with_memory_limit(mut self, mb: usize) -> Self {
        self.memory_limit_mb = Some(mb);
        self
    }
}

// =============================================================================
// Layer Streaming Executor
// =============================================================================

/// Executor for layer-by-layer transformer inference.
///
/// This executor loads and executes transformer layers one at a time,
/// enabling inference of large models on memory-constrained systems.
///
/// # Memory Efficiency
///
/// For a 70B parameter model:
/// - Full loading: ~130GB peak memory
/// - Layer-by-layer: ~2GB peak memory (single layer + activations)
///
/// # Prefetching
///
/// When enabled, the executor prefetches the next layer's weights while
/// executing the current layer, hiding ~10% of I/O latency.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_ai::runtime::loader::{load_pipeline_bundle, LayerStreamingExecutor};
///
/// let pipeline = load_pipeline_bundle(Path::new("llama-70b.holo"))?;
/// let executor = LayerStreamingExecutor::new(pipeline);
///
/// // Iterate through layers with prefetching
/// for layer_ctx in executor.iter_with_prefetch() {
///     let (plan_executor, backend) = layer_ctx.load()?;
///     // Execute with your inputs/outputs using plan_executor
/// }
/// ```
pub struct LayerStreamingExecutor {
    /// The underlying pipeline bundle
    pipeline: PipelineBundle,
    /// Ordered list of layer names for execution
    layer_names: Vec<String>,
}

impl LayerStreamingExecutor {
    /// Create a new streaming executor from a pipeline bundle.
    pub fn new(pipeline: PipelineBundle) -> Self {
        let layer_names = pipeline
            .model_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        Self {
            pipeline,
            layer_names,
        }
    }

    /// Get the number of layers in the model.
    pub fn layer_count(&self) -> usize {
        self.layer_names.len()
    }

    /// Get the layer names in execution order.
    pub fn layer_names(&self) -> &[String] {
        &self.layer_names
    }

    /// Get a reference to the underlying pipeline bundle.
    pub fn pipeline(&self) -> &PipelineBundle {
        &self.pipeline
    }

    /// Load a specific layer by index.
    ///
    /// Returns the `PlanExecutor` and backend for the layer.
    pub fn load_layer_by_index(
        &self,
        index: usize,
    ) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
        let name = self
            .layer_names
            .get(index)
            .ok_or_else(|| anyhow::anyhow!("Layer index {} out of bounds", index))?;
        self.pipeline.load_model(name)
    }

    /// Load a specific layer by name.
    ///
    /// Returns the `PlanExecutor` and backend for the layer.
    pub fn load_layer_by_name(
        &self,
        name: &str,
    ) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
        self.pipeline.load_model(name)
    }

    /// Prefetch a layer by index.
    ///
    /// Call this before loading the layer to hint the OS to page in the weights.
    pub fn prefetch_layer(&self, index: usize) -> Result<()> {
        if let Some(name) = self.layer_names.get(index) {
            self.pipeline.prefetch_model(name)
        } else {
            Ok(()) // Silently ignore out-of-bounds
        }
    }

    /// Release a layer by index.
    ///
    /// Call this after the layer is no longer needed to reduce memory pressure.
    pub fn release_layer(&self, index: usize) -> Result<()> {
        if let Some(name) = self.layer_names.get(index) {
            self.pipeline.release_model(name)
        } else {
            Ok(()) // Silently ignore out-of-bounds
        }
    }

    /// Create an iterator that manages prefetching automatically.
    ///
    /// This iterator yields layer contexts that handle prefetch/release.
    pub fn iter_with_prefetch(&self) -> LayerIterator<'_> {
        LayerIterator {
            executor: self,
            current_index: 0,
        }
    }
}

/// Iterator over layers with automatic prefetching.
pub struct LayerIterator<'a> {
    executor: &'a LayerStreamingExecutor,
    current_index: usize,
}

impl<'a> Iterator for LayerIterator<'a> {
    type Item = LayerContext<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.executor.layer_count() {
            return None;
        }

        let index = self.current_index;
        self.current_index += 1;

        // Prefetch next layer
        if self.current_index < self.executor.layer_count()
            && let Err(e) = self.executor.prefetch_layer(self.current_index)
        {
            tracing::warn!("Failed to prefetch layer {}: {}", self.current_index, e);
        }

        Some(LayerContext {
            executor: self.executor,
            index,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.executor.layer_count() - self.current_index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LayerIterator<'_> {}

/// Context for a single layer during iteration.
///
/// Provides methods to load and execute the layer.
pub struct LayerContext<'a> {
    executor: &'a LayerStreamingExecutor,
    index: usize,
}

impl<'a> LayerContext<'a> {
    /// Get the layer index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the layer name.
    pub fn name(&self) -> &str {
        &self.executor.layer_names[self.index]
    }

    /// Load the layer's executor and backend.
    pub fn load(&self) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
        self.executor.load_layer_by_index(self.index)
    }

    /// Release the previous layer's memory.
    ///
    /// Call this after you've finished using the previous layer.
    pub fn release_previous(&self) -> Result<()> {
        if self.index > 0 {
            self.executor.release_layer(self.index - 1)
        } else {
            Ok(())
        }
    }
}

impl Drop for LayerContext<'_> {
    fn drop(&mut self) {
        // Automatically release this layer when the context is dropped
        let _ = self.executor.release_layer(self.index);
    }
}

#[cfg(unix)]
fn advise_sequential(mmap: &MappedInput) {
    let data = mmap.as_slice();
    if data.is_empty() {
        return;
    }
    unsafe {
        libc::madvise(data.as_ptr() as *mut _, data.len(), libc::MADV_SEQUENTIAL);
    }
}

#[cfg(not(unix))]
fn advise_sequential(_mmap: &MappedInput) {}

#[cfg(unix)]
fn advise_willneed_range(mmap: &MappedInput, offset: usize, size: usize) {
    let data = mmap.as_slice();
    if data.is_empty() || offset >= data.len() {
        return;
    }
    let end = offset.saturating_add(size).min(data.len());
    let len = end.saturating_sub(offset);
    if len == 0 {
        return;
    }
    unsafe {
        libc::madvise(
            data.as_ptr().add(offset) as *mut _,
            len,
            libc::MADV_WILLNEED,
        );
    }
}

#[cfg(not(unix))]
fn advise_willneed_range(_mmap: &MappedInput, _offset: usize, _size: usize) {}

#[cfg(unix)]
fn advise_dontneed_range(mmap: &MappedInput, offset: usize, size: usize) {
    let data = mmap.as_slice();
    if data.is_empty() || offset >= data.len() {
        return;
    }
    let end = offset.saturating_add(size).min(data.len());
    let len = end.saturating_sub(offset);
    if len == 0 {
        return;
    }
    unsafe {
        libc::madvise(
            data.as_ptr().add(offset) as *mut _,
            len,
            libc::MADV_DONTNEED,
        );
    }
}

#[cfg(not(unix))]
fn advise_dontneed_range(_mmap: &MappedInput, _offset: usize, _size: usize) {}

/// Lock a memory range in RAM to prevent swapping.
///
/// Uses `mlock()` to lock pages in physical memory. This ensures zero-latency
/// access but consumes physical RAM. Only use for small, frequently-accessed
/// data like embedding tables.
///
/// # Safety
///
/// Requires sufficient memory lock limit (`ulimit -l`). If the limit is too low,
/// this will fail with ENOMEM or EPERM.
#[cfg(unix)]
fn lock_memory_range(mmap: &MappedInput, offset: usize, size: usize) -> Result<()> {
    let data = mmap.as_slice();
    if data.is_empty() || offset >= data.len() {
        return Ok(());
    }
    let end = offset.saturating_add(size).min(data.len());
    let len = end.saturating_sub(offset);
    if len == 0 {
        return Ok(());
    }

    unsafe {
        let ptr = data.as_ptr().add(offset) as *mut libc::c_void;
        let result = libc::mlock(ptr, len);
        if result != 0 {
            let errno = *libc::__errno_location();
            return Err(anyhow::anyhow!(
                "Failed to lock memory (mlock): errno {} (size: {} bytes). \
                 Check ulimit -l for memory lock limit.",
                errno,
                len
            ));
        }
    }

    Ok(())
}

#[cfg(not(unix))]
fn lock_memory_range(_mmap: &MappedInput, _offset: usize, _size: usize) -> Result<()> {
    // Memory locking is Unix-only
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore] // Requires compiled model file to exist
    fn test_load_t5_encoder() {
        let encoder_path = PathBuf::from("models/t5-small/compiled/encoder.holo");

        assert!(
            encoder_path.exists(),
            "T5 encoder not found at {:?}",
            encoder_path
        );

        let result = load_and_compile_holo(&encoder_path);
        assert!(
            result.is_ok(),
            "Failed to load T5 encoder: {:?}",
            result.err()
        );
    }

    #[test]
    #[ignore] // Requires compiled pipeline file to exist
    fn test_pipeline_prefetch_and_release() {
        // This test demonstrates the prefetch/release API with a real pipeline
        let pipeline_path = PathBuf::from("models/t5-small/compiled/t5-pipeline.holo");

        if !pipeline_path.exists() {
            return; // Skip if file doesn't exist
        }

        let pipeline = load_pipeline_bundle(&pipeline_path).unwrap();

        // Test basic methods
        assert!(pipeline.model_count() > 0);
        let names = pipeline.model_names();

        // Prefetch should work for existing models
        if !names.is_empty() {
            assert!(pipeline.prefetch_model(names[0]).is_ok());
            assert!(pipeline.release_model(names[0]).is_ok());
        }

        // Prefetch should fail for non-existent models
        assert!(pipeline.prefetch_model("nonexistent_model").is_err());
        assert!(pipeline.release_model("nonexistent_model").is_err());
    }

    #[test]
    fn test_pipeline_bundle_methods() {
        // Test that PipelineBundle correctly stores model info
        // We can't test with real data without a file, but we can verify the struct works

        // Create a minimal test by checking that the struct fields are accessible
        // The actual functionality is tested in integration tests
    }

    #[test]
    fn test_is_embedding_layer() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file for the PipelineBundle
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0u8; 256]).unwrap();
        temp_file.flush().unwrap();

        let mmap = Arc::new(MappedInput::open(temp_file.path()).unwrap());
        let pipeline = PipelineBundle {
            mmap,
            model_info: vec![],
        };

        // Test positive cases
        assert!(pipeline.is_embedding_layer("embedding"));
        assert!(pipeline.is_embedding_layer("token_embeddings"));
        assert!(pipeline.is_embedding_layer("embed_tokens"));
        assert!(pipeline.is_embedding_layer("vocab_embed"));
        assert!(pipeline.is_embedding_layer("model.embed_tokens"));
        assert!(pipeline.is_embedding_layer("EMBEDDING")); // Case insensitive
        assert!(pipeline.is_embedding_layer("token_ids"));
        assert!(pipeline.is_embedding_layer("vocab_size"));

        // Test negative cases
        assert!(!pipeline.is_embedding_layer("encoder"));
        assert!(!pipeline.is_embedding_layer("decoder"));
        assert!(!pipeline.is_embedding_layer("attention"));
        assert!(!pipeline.is_embedding_layer("layer_0"));
        assert!(!pipeline.is_embedding_layer("feedforward"));
    }

    #[test]
    fn test_pin_model_weights_not_found() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file for the PipelineBundle
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(&[0u8; 256]).unwrap();
        temp_file.flush().unwrap();

        let mmap = Arc::new(MappedInput::open(temp_file.path()).unwrap());
        let pipeline = PipelineBundle {
            mmap,
            model_info: vec![],
        };

        // Test that pinning a non-existent model fails
        let result = pipeline.pin_model_weights("nonexistent");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("not found in pipeline")
        );
    }

    #[test]
    fn test_pin_model_weights_size_check() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a large temporary file (128MB)
        let mut temp_file = NamedTempFile::new().unwrap();
        let large_data = vec![0u8; 128 * 1024 * 1024];
        temp_file.write_all(&large_data).unwrap();
        temp_file.flush().unwrap();

        let mmap = Arc::new(MappedInput::open(temp_file.path()).unwrap());
        let pipeline = PipelineBundle {
            mmap,
            model_info: vec![("large_model".to_string(), 0, 128 * 1024 * 1024)],
        };

        // Test that pinning a large model fails with size error
        let result = pipeline.pin_model_weights("large_model");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("exceeds max pinnable size"),
            "Expected size error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_pin_model_weights_small_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a small temporary file (1KB)
        let mut temp_file = NamedTempFile::new().unwrap();
        let data = vec![0u8; 1024];
        temp_file.write_all(&data).unwrap();
        temp_file.flush().unwrap();

        let mmap = Arc::new(MappedInput::open(temp_file.path()).unwrap());
        let pipeline = PipelineBundle {
            mmap,
            model_info: vec![("small_model".to_string(), 0, 1024)],
        };

        // Test that pinning a small model succeeds (or fails gracefully on systems with low ulimit)
        let result = pipeline.pin_model_weights("small_model");
        // We accept both success and permission errors (due to ulimit)
        if let Err(e) = result {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("mlock") || err_msg.contains("ulimit"),
                "Expected mlock or ulimit error, got: {}",
                err_msg
            );
        }
    }
}

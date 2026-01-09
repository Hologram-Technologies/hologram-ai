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
use hologram::backend::executor::PlanExecutor;
use hologram::core::memory::MappedInput;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use crate::core::{HoloFormat, PipelineBundleReader, UnifiedBundleReader};

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
pub fn load_and_compile_holo(
    path: &Path,
) -> Result<(
    hologram::backend::BackendPlan,
    Box<dyn hologram::backend::ProgramBackend>,
)> {
    tracing::info!("Loading .holo file: {}", path.display());

    // Use hologram's read_holo() API
    // This automatically:
    // 1. Verifies magic bytes
    // 2. Deserializes SerializableBackendPlan
    // 3. Resolves kernel IDs to function pointers based on CPU capabilities
    let plan = hologram::compiler::read_holo(path)
        .map_err(|e| anyhow::anyhow!("Failed to load .holo file: {:?}", e))?;

    tracing::debug!("Deserialized BackendPlan from .holo file");

    // Use hologram's backend creation with proper fallback handling
    let backend = match hologram::backend::create_backend(plan.backend_type.clone()) {
        Ok(backend) => backend,
        Err(e) => {
            tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
            hologram::backend::create_best_backend()
        }
    };

    tracing::info!("Using backend: {:?}", backend.backend_type());
    tracing::info!("Successfully loaded BackendPlan from .holo file");

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
pub fn load_with_external_weights(
    holo_path: &Path,
    weights_path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    tracing::info!(
        "Loading .holo file with external weights: {} + {}",
        holo_path.display(),
        weights_path.display()
    );

    // Load and deserialize the .holo file
    let plan = hologram::compiler::read_holo(holo_path)
        .map_err(|e| anyhow::anyhow!("Failed to load .holo file: {:?}", e))?;

    tracing::debug!(
        "Deserialized BackendPlan (constant_data: {} bytes, will use mmap instead)",
        plan.constant_data.len()
    );

    // Create backend with fallback
    let backend = match hologram::backend::create_backend(plan.backend_type.clone()) {
        Ok(backend) => backend,
        Err(e) => {
            tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
            hologram::backend::create_best_backend()
        }
    };

    tracing::info!("Using backend: {:?}", backend.backend_type());

    // Create executor with memory-mapped external constants
    let executor = PlanExecutor::with_external_constants(plan, &*backend, weights_path)
        .map_err(|e| anyhow::anyhow!("Failed to create executor with external weights: {:?}", e))?;

    tracing::info!(
        "Successfully loaded model with external weights from {}",
        weights_path.display()
    );

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
pub fn load_holo_auto(
    path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    // Read magic bytes to detect format
    let mut file = File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open file '{}': {}", path.display(), e))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| anyhow::anyhow!("Failed to read magic bytes: {}", e))?;
    drop(file);

    let format = HoloFormat::detect(&magic);

    match format {
        HoloFormat::Bundle => {
            tracing::info!("Detected unified bundle format (HOLB): {}", path.display());
            load_unified_bundle(path)
        }
        HoloFormat::Pipeline => {
            // Pipeline bundles contain multiple models - use load_pipeline_bundle instead
            Err(anyhow::anyhow!(
                "Pipeline bundle detected: {}. Use load_pipeline_bundle() and specify model name.",
                path.display()
            ))
        }
        HoloFormat::Plan | HoloFormat::Legacy => {
            tracing::info!("Detected legacy format (HOLP): {}", path.display());
            // Check for companion .weights file
            let weights_path = path.with_extension("weights");
            if weights_path.exists() {
                tracing::info!("Found external weights file: {}", weights_path.display());
                load_with_external_weights(path, &weights_path)
            } else {
                load_holo_file(path, None)
            }
        }
        HoloFormat::Unknown => {
            Err(anyhow::anyhow!(
                "Unknown file format: {:?} (magic: {:?})",
                path.display(),
                magic
            ))
        }
    }
}

/// Load a unified bundle file (HOLB format).
///
/// The bundle contains both the computation graph and weights in a single file.
/// Weights are memory-mapped from the page-aligned section within the bundle.
fn load_unified_bundle(
    path: &Path,
) -> Result<(PlanExecutor, Box<dyn hologram::backend::ProgramBackend>)> {
    // Memory-map the entire bundle file
    let mmap = MappedInput::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to mmap bundle '{}': {}", path.display(), e))?;

    // Parse bundle header and validate, extracting needed values before moving mmap
    let (plan, weights_offset) = {
        let reader = UnifiedBundleReader::from_bytes(mmap.as_slice())
            .map_err(|e| anyhow::anyhow!("Failed to parse bundle header: {:?}", e))?;

        // Verify checksums
        if !reader.verify_checksums() {
            return Err(anyhow::anyhow!("Bundle checksum verification failed"));
        }

        tracing::info!(
            "Bundle: graph={} bytes, weights={} bytes, weights_offset={}",
            reader.graph_size(),
            reader.weights_size(),
            reader.weights_mmap_offset().unwrap_or(0)
        );

        // Deserialize the graph section (HOLP format)
        let plan = hologram::compiler::read_holo_from_bytes(reader.graph_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize graph from bundle: {:?}", e))?;

        let weights_offset = reader.weights_mmap_offset();

        (plan, weights_offset)
    }; // reader goes out of scope here, releasing borrow of mmap

    tracing::debug!("Deserialized BackendPlan from bundle");

    // Create backend
    let backend = match hologram::backend::create_backend(plan.backend_type.clone()) {
        Ok(backend) => backend,
        Err(e) => {
            tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
            hologram::backend::create_best_backend()
        }
    };

    tracing::info!("Using backend: {:?}", backend.backend_type());

    // Create executor with mmap'd weights at the bundle offset
    let mmap_arc = Arc::new(mmap);
    let executor = if let Some(offset) = weights_offset {
        PlanExecutor::with_mmap_constants_at_offset(plan, &*backend, mmap_arc, offset)
            .map_err(|e| anyhow::anyhow!("Failed to create executor with bundle weights: {:?}", e))?
    } else {
        // No weights in bundle - just create normal executor
        PlanExecutor::new(plan, &*backend)
            .map_err(|e| anyhow::anyhow!("Failed to create executor: {:?}", e))?
    };

    tracing::info!("Successfully loaded unified bundle: {}", path.display());

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
        let plan = hologram::compiler::read_holo_from_bytes(model_reader.graph_bytes())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize model '{}' graph: {:?}", name, e))?;

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
pub fn load_pipeline_bundle(path: &Path) -> Result<PipelineBundle> {
    // Memory-map the pipeline file
    let mmap = MappedInput::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to mmap pipeline bundle '{}': {}", path.display(), e))?;

    // Parse the pipeline header and index
    let reader = PipelineBundleReader::from_bytes(mmap.as_slice())
        .map_err(|e| anyhow::anyhow!("Failed to parse pipeline bundle header: {:?}", e))?;

    // Verify checksums
    if !reader.verify_index_checksum() {
        return Err(anyhow::anyhow!("Pipeline index checksum verification failed"));
    }

    tracing::info!(
        "Loaded pipeline bundle with {} models: {:?}",
        reader.model_count(),
        reader.model_names()
    );

    // Extract model info (we can't keep the reader because it borrows mmap)
    let model_info: Vec<(String, usize, usize)> = reader
        .model_names()
        .iter()
        .filter_map(|name| {
            reader.get_entry(name).map(|entry| {
                (name.to_string(), entry.offset as usize, entry.size as usize)
            })
        })
        .collect();

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
        assert!(result.is_ok(), "Failed to load T5 encoder: {:?}", result.err());
    }
}

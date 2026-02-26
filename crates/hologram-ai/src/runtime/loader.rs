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
//! ## External Weights
//! Use [`load_with_external_weights`] when weights are stored separately.

use anyhow::{Context, Result};
use std::path::Path;

use hologram::backend::cpu::CpuBackend;
use hologram::backend::{Backend, BackendPlan};
use hologram::holo::HolbReader;
use hologram::holo::pipeline::HolmReader;

/// Result type for model loading with optional input ordering.
pub type ModelLoadResult = (BackendPlan, Box<dyn Backend>, Option<Vec<String>>);

/// Section ID for input order metadata.
const INPUT_ORDER_SECTION_ID: &str = "input_order";

/// Deserialize a BackendPlan from rkyv bytes.
///
/// Uses rkyv 0.7's archived_root + Deserialize pattern.
fn deserialize_backend_plan(bytes: &[u8]) -> Result<BackendPlan> {
    // SAFETY: The bytes come from a valid .holo file that was serialized with rkyv
    let archived = unsafe { rkyv::archived_root::<BackendPlan>(bytes) };
    let plan: BackendPlan = rkyv::Deserialize::deserialize(archived, &mut rkyv::Infallible)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize BackendPlan: {:?}", e))?;
    Ok(plan)
}

/// Extract input order from a HolbReader's sections.
///
/// Returns None if no input_order section exists.
fn extract_input_order(reader: &HolbReader) -> Option<Vec<String>> {
    // Try to get the raw section data for input_order
    let section_data = reader.get_section(INPUT_ORDER_SECTION_ID)?;

    // Parse as JSON array of strings
    let input_names: Vec<String> = serde_json::from_slice(section_data).ok()?;

    tracing::debug!("Loaded input order from section: {:?}", input_names);
    Some(input_names)
}

/// Load a .holo file to an executable BackendPlan.
///
/// This function loads a pre-compiled .holo file and prepares it for execution.
///
/// # Arguments
/// * `path` - Path to .holo file
///
/// # Returns
/// Tuple of (BackendPlan, Backend) ready for execution
///
/// # Errors
/// Returns error if:
/// - File cannot be read
/// - Deserialization fails (corrupted .holo file)
#[tracing::instrument(
    name = "load_and_compile_holo",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_and_compile_holo(path: &Path) -> Result<(BackendPlan, Box<dyn Backend>)> {
    // Read file
    let bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read .holo file: {}", path.display()))?;

    // Parse and deserialize
    let reader = HolbReader::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse .holo file: {:?}", e))?;

    // Deserialize the graph (BackendPlan)
    let mut plan = deserialize_backend_plan(reader.graph())?;

    // Check for WeightIndexSection (future: lazy/partial loading)
    if let Ok(Some(weight_index)) = reader.weight_index() {
        tracing::debug!(
            "HOLB has WeightIndexSection with {} tensors",
            weight_index.len()
        );
    }

    // Load weights from HOLB bundle
    let weights = reader.weights();
    if !weights.is_empty() {
        tracing::debug!("Loading {} bytes of weights from HOLB", weights.len());
        plan.constants = weights.to_vec();
    }

    tracing::debug!("Deserialized BackendPlan from .holo file");

    // Create backend
    let backend: Box<dyn Backend> = Box::new(CpuBackend::new());

    tracing::info!(backend = "CPU", "Successfully loaded BackendPlan");

    Ok((plan, backend))
}

/// Load a .holo file with external memory-mapped weights.
///
/// This function loads a .holo file and creates a Backend that uses
/// memory-mapped access to the external weights file.
///
/// # Arguments
/// * `holo_path` - Path to the .holo file
/// * `weights_path` - Path to the .weights file (memory-mapped)
///
/// # Returns
/// Tuple of (BackendPlan, Backend) ready for execution
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
) -> Result<(BackendPlan, Box<dyn Backend>)> {
    // Load the plan
    let bytes = std::fs::read(holo_path)
        .with_context(|| format!("Failed to read .holo file: {}", holo_path.display()))?;

    let reader = HolbReader::from_bytes(&bytes)
        .map_err(|e| anyhow::anyhow!("Failed to parse .holo file: {:?}", e))?;

    let mut plan = deserialize_backend_plan(reader.graph())?;

    // Load external weights
    let weights_bytes = std::fs::read(weights_path)
        .with_context(|| format!("Failed to read .weights file: {}", weights_path.display()))?;

    tracing::info!(
        weights_size = weights_bytes.len(),
        "Loaded external weights"
    );

    // Replace plan constants with external weights
    plan.constants = weights_bytes;

    // Create backend
    let backend: Box<dyn Backend> = Box::new(CpuBackend::new());

    tracing::info!("Successfully loaded model with external weights");

    Ok((plan, backend))
}

/// Load a .holo file with optional external weights.
///
/// This is a convenience function that automatically selects the appropriate
/// loading strategy based on whether a weights file exists.
pub fn load_holo_file(
    holo_path: &Path,
    weights_path: Option<&Path>,
) -> Result<(BackendPlan, Box<dyn Backend>)> {
    if let Some(wp) = weights_path {
        load_with_external_weights(holo_path, wp)
    } else {
        load_and_compile_holo(holo_path)
    }
}

/// Detect file format from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoloFormat {
    /// Unified bundle (HOLB magic)
    Bundle,
    /// Pipeline bundle (HOLM magic)
    Pipeline,
    /// Legacy plan format
    Plan,
    /// Legacy format
    Legacy,
    /// Unknown format
    Unknown,
}

impl HoloFormat {
    /// Detect format from 4-byte magic header.
    pub fn detect(magic: &[u8; 4]) -> Self {
        match magic {
            b"HOLB" => Self::Bundle,
            b"HOLM" => Self::Pipeline,
            b"HOLP" => Self::Plan,
            _ => {
                // Check for rkyv archive (starts with alignment bytes)
                if magic[0] == 0 || magic.starts_with(&[0x00, 0x00, 0x00, 0x00]) {
                    Self::Legacy
                } else {
                    Self::Unknown
                }
            }
        }
    }
}

/// Automatically detect file format and load appropriately.
///
/// This function auto-detects whether the file is:
/// - **Unified Bundle (HOLB)**: Single file with embedded weights
/// - **Pipeline Bundle (HOLM)**: Multiple models in one file
/// - **Legacy Plan**: Separate .holo + optional .weights files
///
/// # Arguments
/// * `path` - Path to the .holo file
///
/// # Returns
/// Tuple of (BackendPlan, Backend) ready for execution
#[tracing::instrument(
    name = "load_holo_auto",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_holo_auto(path: &Path) -> Result<(BackendPlan, Box<dyn Backend>)> {
    let (plan, backend, _input_order) = load_holo_auto_with_inputs(path)?;
    Ok((plan, backend))
}

/// Automatically detect file format and load with input ordering.
///
/// Like [`load_holo_auto`] but also returns the input order if embedded in the .holb file.
/// This is essential for models with multiple inputs of the same size.
///
/// # Arguments
/// * `path` - Path to the .holo file
///
/// # Returns
/// Tuple of (BackendPlan, Backend, Option<Vec<String>>) where the Vec contains input names
/// in the order expected by the model.
#[tracing::instrument(
    name = "load_holo_auto_with_inputs",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_holo_auto_with_inputs(path: &Path) -> Result<ModelLoadResult> {
    // Read file bytes
    let bytes =
        std::fs::read(path).with_context(|| format!("Failed to read file: {}", path.display()))?;

    // Check magic
    if bytes.len() < 4 {
        anyhow::bail!("File too small: {}", path.display());
    }

    let magic: [u8; 4] = bytes[..4].try_into().unwrap();
    let format = HoloFormat::detect(&magic);

    tracing::info!(format = ?format, "Detected file format");

    // Route to appropriate loader
    match format {
        HoloFormat::Bundle | HoloFormat::Plan | HoloFormat::Legacy => {
            // Parse HOLB to extract input order before loading
            let reader = HolbReader::from_bytes(&bytes)
                .map_err(|e| anyhow::anyhow!("Failed to parse .holo file: {:?}", e))?;

            // Extract input order from embedded section
            let input_order = extract_input_order(&reader);
            if let Some(ref order) = input_order {
                tracing::info!("Found embedded input order: {:?}", order);
            }

            // Deserialize the graph (BackendPlan)
            let mut plan = deserialize_backend_plan(reader.graph())?;

            // Check for WeightIndexSection (future: lazy/partial loading)
            if let Ok(Some(weight_index)) = reader.weight_index() {
                tracing::debug!(
                    "HOLB has WeightIndexSection with {} tensors",
                    weight_index.len()
                );
            }

            // Check for external weights file first
            let weights_path = path.with_extension("weights");
            if weights_path.exists() {
                tracing::info!(weights_path = %weights_path.display(), "Found external weights file");
                let weights_bytes = std::fs::read(&weights_path).with_context(|| {
                    format!("Failed to read .weights file: {}", weights_path.display())
                })?;
                plan.constants = weights_bytes;
            } else {
                // Load weights from HOLB bundle
                let weights = reader.weights();
                if !weights.is_empty() {
                    tracing::debug!("Loading {} bytes of weights from HOLB", weights.len());
                    plan.constants = weights.to_vec();
                }
            }

            tracing::debug!("Deserialized BackendPlan from .holo file");

            // Create backend
            let backend: Box<dyn Backend> = Box::new(CpuBackend::new());

            tracing::info!(backend = "CPU", "Successfully loaded BackendPlan");

            Ok((plan, backend, input_order))
        }
        HoloFormat::Pipeline => Err(anyhow::anyhow!(
            "Pipeline bundle detected: {}. Use load_pipeline_bundle() instead.",
            path.display()
        )),
        HoloFormat::Unknown => Err(anyhow::anyhow!("Unknown file format: {}", path.display())),
    }
}

/// A loaded pipeline bundle that provides access to multiple models.
///
/// The bundle contains multiple models that can be loaded on demand.
pub struct PipelineBundle {
    /// Raw bundle bytes (memory-mapped or read)
    data: Vec<u8>,
    /// Model info: (name, offset, size)
    model_info: Vec<(String, usize, usize)>,
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
    pub fn load_model(&self, name: &str) -> Result<(BackendPlan, Box<dyn Backend>)> {
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

        // Get the model bytes
        let model_bytes = &self.data[*model_offset..*model_offset + *model_size];

        // Parse as HOLB bundle
        let reader = HolbReader::from_bytes(model_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to parse model '{}': {:?}", name, e))?;

        tracing::info!(
            "  HOLB version: {}, graph size: {} bytes, weights size: {} bytes",
            reader.version(),
            reader.graph().len(),
            reader.weights().len()
        );

        // Deserialize the graph (BackendPlan)
        let mut plan = deserialize_backend_plan(reader.graph())
            .map_err(|e| anyhow::anyhow!("Failed to deserialize model '{}': {:?}", name, e))?;

        tracing::info!(
            "  Deserialized plan: {} buffers, {} instructions, constants size: {} bytes",
            plan.buffers.len(),
            plan.instructions.len(),
            plan.constants.len()
        );

        // Check for WeightIndexSection for partial/indexed loading
        if let Ok(Some(weight_index)) = reader.weight_index() {
            tracing::debug!(
                "Model '{}' has WeightIndexSection with {} tensors",
                name,
                weight_index.len()
            );
            // Future: use weight_index for lazy/partial loading
        }

        // Load the weights from the HOLB bundle
        let weights = reader.weights();
        if !weights.is_empty() {
            tracing::info!(
                "  Loading {} bytes of weights from HOLB (overriding {} bytes from plan)",
                weights.len(),
                plan.constants.len()
            );
            plan.constants = weights.to_vec();
        } else {
            tracing::info!(
                "  No separate weights in HOLB, using {} bytes from plan constants",
                plan.constants.len()
            );
        }

        let backend: Box<dyn Backend> = Box::new(CpuBackend::new());

        tracing::info!(
            "Successfully loaded model '{}' (final constants: {} bytes)",
            name,
            plan.constants.len()
        );

        Ok((plan, backend))
    }

    /// Load a model with explicit input ordering.
    pub fn load_model_with_inputs(&self, name: &str) -> Result<ModelLoadResult> {
        let (plan, backend) = self.load_model(name)?;
        // Input order could be extracted from metadata if available
        Ok((plan, backend, None))
    }
}

/// Load a pipeline bundle (HOLM format).
///
/// # Arguments
/// * `path` - Path to the pipeline bundle file
///
/// # Returns
/// PipelineBundle that provides access to individual models
#[tracing::instrument(
    name = "load_pipeline_bundle",
    skip_all,
    fields(path = %path.display())
)]
pub fn load_pipeline_bundle(path: &Path) -> Result<PipelineBundle> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read pipeline bundle: {}", path.display()))?;

    // Use HolmReader from hologram-holo (correctly handles format)
    let reader = HolmReader::from_bytes(&data)
        .map_err(|e| anyhow::anyhow!("Failed to parse HOLM: {:?}", e))?;

    // Build model_info from reader entries
    let model_info: Vec<(String, usize, usize)> = reader
        .entries()
        .iter()
        .map(|e| (e.name.clone(), e.offset as usize, e.size as usize))
        .collect();

    tracing::info!(
        "Loaded pipeline bundle with {} models: {:?}",
        model_info.len(),
        reader.model_names()
    );

    Ok(PipelineBundle { data, model_info })
}

/// Check if a loaded plan contains CallLayer instructions and print diagnostic info.
///
/// This is useful for debugging the CallLayer dependencies issue.
#[cfg(test)]
fn debug_plan_calllayer(plan: &BackendPlan) {
    use hologram::holo::IsaInstruction;

    let call_layer_count = plan
        .instructions
        .iter()
        .filter(|i| matches!(i, IsaInstruction::CallLayer { .. }))
        .count();

    println!("Instructions: {}", plan.instructions.len());
    println!("CallLayer count: {}", call_layer_count);
    println!("Dependencies: {}", plan.dependencies.len());

    for instr in &plan.instructions {
        if let IsaInstruction::CallLayer {
            layer_id,
            inputs,
            outputs,
        } = instr
        {
            println!(
                "  CallLayer: layer_id=0x{:016x}, inputs={:?}, outputs={:?}",
                layer_id, inputs, outputs
            );
        }
    }

    for dep in &plan.dependencies {
        println!("  Dependency: {:?}", dep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_holo_format_detect() {
        assert_eq!(HoloFormat::detect(b"HOLB"), HoloFormat::Bundle);
        assert_eq!(HoloFormat::detect(b"HOLM"), HoloFormat::Pipeline);
        assert_eq!(HoloFormat::detect(b"HOLP"), HoloFormat::Plan);
        assert_eq!(HoloFormat::detect(&[0, 0, 0, 0]), HoloFormat::Legacy);
        assert_eq!(HoloFormat::detect(b"XXXX"), HoloFormat::Unknown);
    }

    #[test]
    fn test_pipeline_bundle_empty() {
        let bundle = PipelineBundle {
            data: vec![],
            model_info: vec![],
        };
        assert_eq!(bundle.model_count(), 0);
        assert!(bundle.model_names().is_empty());
        assert!(!bundle.has_model("encoder"));
    }

    #[test]
    #[ignore = "requires compiled model file"]
    fn test_inspect_encoder_for_calllayer() {
        let path = Path::new("/tmp/test_encoder.holb");
        if !path.exists() {
            println!("Skipping: {} does not exist", path.display());
            return;
        }

        let (plan, _backend) = load_and_compile_holo(path).expect("Failed to load plan");

        debug_plan_calllayer(&plan);
    }
}

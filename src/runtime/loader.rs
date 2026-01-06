//! Loading and compiling .holo files to executable BackendPlan.

use anyhow::{Context, Result};
use std::path::Path;

use hologram_ir::OperationGraph;

/// Load a .holo file and compile it to an executable BackendPlan.
///
/// This function performs the critical bridge from IR to execution:
/// 1. Read .holo file bytes
/// 2. Deserialize to OperationGraph (IR representation)
/// 3. Convert IR → CompileGraph (compiler representation)
/// 4. Compile CompileGraph → BackendPlan (executable)
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
/// - IR → CompileGraph conversion fails (unsupported operations)
/// - Backend compilation fails
pub fn load_and_compile_holo(
    path: &Path,
) -> Result<(
    hologram_backend::BackendPlan,
    Box<dyn hologram_backend::ProgramBackend>,
)> {
    tracing::info!("Loading .holo file: {}", path.display());

    // Step 1: Read .holo file
    let holo_bytes = std::fs::read(path)
        .with_context(|| format!("Failed to read .holo file: {}", path.display()))?;

    tracing::debug!("Read {} bytes from .holo file", holo_bytes.len());

    // Step 2: Deserialize to OperationGraph (IR)
    let ir_graph = OperationGraph::from_bytes(&holo_bytes)
        .map_err(|e| anyhow::anyhow!("Failed to deserialize .holo file: {:?}", e))?;

    tracing::debug!("Deserialized OperationGraph from .holo file");

    // Step 3: Convert IR → CompileGraph
    let compile_graph = hologram_compiler::convert_from_ir(&ir_graph)
        .map_err(|e| anyhow::anyhow!("IR → CompileGraph conversion failed: {:?}", e))?;

    tracing::debug!("Converted IR to CompileGraph");

    // Step 4: Create backend
    let backend = hologram_backend::create_best_backend();
    let backend_type = backend.backend_type();

    // Create capabilities matching the backend type
    let caps = match backend_type {
        hologram_backend::BackendType::Cpu => hologram_backend::BackendCapabilities::cpu(),
        hologram_backend::BackendType::Cuda => hologram_backend::BackendCapabilities::cuda(0),
        hologram_backend::BackendType::Metal => hologram_backend::BackendCapabilities::metal(),
        hologram_backend::BackendType::WebGpu => hologram_backend::BackendCapabilities::webgpu(),
        hologram_backend::BackendType::Custom(_) => {
            // For custom backends, default to CPU capabilities
            hologram_backend::BackendCapabilities::cpu()
        }
    };

    tracing::info!("Using backend: {:?}", backend_type);

    // Step 5: Compile CompileGraph → BackendPlan
    let pipeline_config = hologram_compiler::PipelineConfig::default();
    let pipeline =
        hologram_compiler::CompilationPipeline::with_config(backend_type, pipeline_config);

    let plan = pipeline
        .compile(&compile_graph, &caps)
        .map_err(|e| anyhow::anyhow!("BackendPlan compilation failed: {:?}", e))?;

    tracing::info!("Successfully compiled BackendPlan from .holo file");

    Ok((plan, backend))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore] // Requires compiled T5 models
    fn test_load_t5_encoder() {
        let encoder_path = PathBuf::from("models/t5-small/compiled/encoder.holo");

        if !encoder_path.exists() {
            println!("Skipping test: T5 encoder not found at {:?}", encoder_path);
            return;
        }

        let result = load_and_compile_holo(&encoder_path);
        assert!(result.is_ok(), "Failed to load T5 encoder: {:?}", result.err());
    }
}

//! Loading .holo files to executable BackendPlan.

use anyhow::Result;
use std::path::Path;

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
    hologram_backend::BackendPlan,
    Box<dyn hologram_backend::ProgramBackend>,
)> {
    tracing::info!("Loading .holo file: {}", path.display());

    // Use hologram's read_holo() API
    // This automatically:
    // 1. Verifies magic bytes
    // 2. Deserializes SerializableBackendPlan
    // 3. Resolves kernel IDs to function pointers based on CPU capabilities
    let plan = hologram_compiler::read_holo(path)
        .map_err(|e| anyhow::anyhow!("Failed to load .holo file: {:?}", e))?;

    tracing::debug!("Deserialized BackendPlan from .holo file");

    // Use hologram's backend creation with proper fallback handling
    let backend = match hologram_backend::create_backend(plan.backend_type.clone()) {
        Ok(backend) => backend,
        Err(e) => {
            tracing::warn!("Failed to create backend: {}. Falling back to CPU", e);
            hologram_backend::create_best_backend()
        }
    };

    tracing::info!("Using backend: {:?}", backend.backend_type());
    tracing::info!("Successfully loaded BackendPlan from .holo file");

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

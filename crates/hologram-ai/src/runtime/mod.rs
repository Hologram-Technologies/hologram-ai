//! Runtime execution for compiled .holo models.
//!
//! This module provides the bridge between compiled .holo files and hologram-backend
//! execution. It handles:
//! - Loading and compiling .holo files to BackendPlan
//! - Tensor I/O conversions
//! - Model execution with hologram-backend's PlanExecutor

mod executor;
mod loader;
mod tensors;

pub use executor::ModelExecutor;
pub use loader::{PipelineBundle, is_pipeline_bundle, load_pipeline_bundle};
pub use loader::{load_and_compile_holo, load_holo_file, load_with_external_weights};
pub use tensors::{Tensor, infer_tensor_dtype, infer_tensor_shape};

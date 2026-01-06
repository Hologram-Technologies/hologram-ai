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
pub use loader::load_and_compile_holo;
pub use tensors::{Tensor, infer_tensor_shape};

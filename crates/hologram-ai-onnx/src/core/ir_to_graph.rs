//! Convert hologram-ir OperationGraph (compatibility layer).
//!
//! This module provides compatibility shims for the old IR conversion API.
//! The actual translation is now handled directly by hologram-ir.

use crate::Result;
use hologram::ir::OperationGraph;
use std::path::Path;

/// Default threshold for streaming weights to external file (256KB in elements).
pub const DEFAULT_WEIGHT_THRESHOLD_ELEMENTS: usize = 64 * 1024;

/// Options for converting IR to OperationGraph.
#[derive(Debug, Clone, Default)]
pub struct ConversionOptions {
    /// Threshold for streaming weights to external file (in elements).
    pub weight_threshold_elements: usize,
    /// Enable Resize upscaling. When false, Resize ops pass through without scaling.
    pub enable_resize_upscaling: bool,
}

impl ConversionOptions {
    /// Create default options with upscaling enabled.
    pub fn new() -> Self {
        Self {
            weight_threshold_elements: DEFAULT_WEIGHT_THRESHOLD_ELEMENTS,
            enable_resize_upscaling: true,
        }
    }
}

/// Convert an OperationGraph with streaming weights support.
///
/// This is a compatibility shim - the input is already an OperationGraph,
/// so we just return it as-is. Weight streaming is handled elsewhere.
pub fn ir_to_operation_graph_streaming_with_options(
    operation_graph: &OperationGraph,
    _weights_path: impl AsRef<Path>,
    _options: ConversionOptions,
) -> Result<OperationGraph> {
    Ok(operation_graph.clone())
}

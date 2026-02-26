//! Shared .holo file serialization for all format compilers.
//!
//! This module provides serialization of BackendPlan to .holb format using
//! hologram's HolbWriter and rkyv serialization.

use anyhow::Result;
use hologram::backend::BackendPlan;
use hologram::holo::HolbWriter;

/// Alignment boundary for page-aligned weight sections (4KB)
pub const HOLO_ALIGN: usize = 4096;

/// Layer header data for serialization.
///
/// This contains metadata about the layer(s) being serialized.
#[derive(Debug, Clone)]
pub struct LayerHeaderData {
    /// Layer information
    pub layers: Vec<LayerInfo>,
}

impl LayerHeaderData {
    /// Create a new header with a single layer.
    pub fn single(name: impl Into<String>) -> Self {
        Self {
            layers: vec![LayerInfo { name: name.into() }],
        }
    }
}

/// Information about a single layer.
#[derive(Debug, Clone)]
pub struct LayerInfo {
    /// Layer name
    pub name: String,
}

/// Weight management strategy for serialization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightStrategy {
    /// Embed weights in BackendPlan.constants (< 100MB)
    EmbeddedInPlan,
    /// Separate page-aligned section in same .holo file (100MB - 1GB)
    PageAlignedInBundle,
    /// Separate .weights file (> 1GB)
    ExternalFile,
}

impl WeightStrategy {
    /// Auto-select based on model size
    pub fn auto_select(model_size_bytes: usize) -> Self {
        let mb = model_size_bytes / (1024 * 1024);
        if mb < 100 {
            WeightStrategy::EmbeddedInPlan
        } else if mb < 1000 {
            WeightStrategy::PageAlignedInBundle
        } else {
            WeightStrategy::ExternalFile
        }
    }
}

/// Serialize a BackendPlan to .holb format.
///
/// This function uses hologram's HolbWriter to create a proper .holb file.
///
/// # Arguments
///
/// * `plan` - The compiled BackendPlan to serialize
/// * `header` - Layer header data (used for validation)
/// * `strategy` - Weight management strategy
///
/// # Returns
///
/// Returns `Ok((holo_bytes, weights_bytes))`:
/// - For `EmbeddedInPlan` and `PageAlignedInBundle`: weights are in holo_bytes, weights_bytes is empty
/// - For `ExternalFile`: weights are in weights_bytes, must be saved separately
///
/// # Errors
///
/// Returns error if:
/// - LayerHeader has no layers
/// - Serialization fails
pub fn serialize_backend_plan_with_header(
    plan: &BackendPlan,
    header: &LayerHeaderData,
    strategy: WeightStrategy,
) -> Result<(Vec<u8>, Vec<u8>)> {
    if header.layers.is_empty() {
        anyhow::bail!("LayerHeader must contain at least one layer");
    }

    // Serialize the BackendPlan using rkyv 0.7
    let plan_bytes = rkyv::to_bytes::<_, 1024>(plan)
        .map_err(|e| anyhow::anyhow!("Failed to serialize BackendPlan: {:?}", e))?
        .to_vec();

    // Build the .holb file using HolbWriter
    let mut writer = HolbWriter::new();
    writer.set_graph(&plan_bytes);

    // Handle weight strategy
    match strategy {
        WeightStrategy::EmbeddedInPlan => {
            // Weights are already in plan.constants, no separate data needed
            let bundle = writer
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HOLB bundle: {:?}", e))?;
            Ok((bundle, Vec::new()))
        }
        WeightStrategy::PageAlignedInBundle => {
            // Extract weights from plan and store page-aligned in the same file
            // The weights are in plan.constants
            if !plan.constants.is_empty() {
                writer.set_weights(&plan.constants);
            }
            let bundle = writer
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HOLB bundle: {:?}", e))?;
            Ok((bundle, Vec::new()))
        }
        WeightStrategy::ExternalFile => {
            // Weights go to separate file
            let bundle = writer
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build HOLB bundle: {:?}", e))?;
            Ok((bundle, plan.constants.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_strategy_auto_select() {
        assert_eq!(
            WeightStrategy::auto_select(50 * 1024 * 1024),
            WeightStrategy::EmbeddedInPlan
        );

        assert_eq!(
            WeightStrategy::auto_select(500 * 1024 * 1024),
            WeightStrategy::PageAlignedInBundle
        );

        assert_eq!(
            WeightStrategy::auto_select(2 * 1024 * 1024 * 1024),
            WeightStrategy::ExternalFile
        );
    }

    #[test]
    fn test_alignment_calculation() {
        let payload_len = 1000;
        let pad = (HOLO_ALIGN - (payload_len % HOLO_ALIGN)) % HOLO_ALIGN;
        assert_eq!(pad, 4096 - 1000);

        let payload_len = 4096;
        let pad = (HOLO_ALIGN - (payload_len % HOLO_ALIGN)) % HOLO_ALIGN;
        assert_eq!(pad, 0);
    }
}

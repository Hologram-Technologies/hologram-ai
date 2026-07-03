pub mod builder;
pub mod parametric;

use anyhow::{Context, Result};
use hologram_ai_common::ir::graph::AiGraph;
use serde_json::Value;

pub fn build_graph_from_safetensors(
    config_json: &str,
    safetensors_shards: &[&[u8]],
) -> Result<AiGraph> {
    let config: Value = serde_json::from_str(config_json).context("Failed to parse config.json")?;

    parametric::build_parametric_graph(&config, safetensors_shards)
}

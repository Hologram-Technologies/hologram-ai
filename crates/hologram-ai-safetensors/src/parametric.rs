use crate::builder::GraphBuilder;
use anyhow::{anyhow, Result};
use hologram_ai_common::ir::{dtype::DType, graph::AiGraph};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde_json::Value;

#[allow(dead_code)]
fn map_dtype(d: SafeDtype) -> Result<DType> {
    match d {
        SafeDtype::F32 => Ok(DType::F32),
        SafeDtype::F16 => Ok(DType::F16),
        SafeDtype::I64 => Ok(DType::INT64),
        SafeDtype::I32 => Ok(DType::INT32),
        _ => Err(anyhow!("Unsupported safetensors dtype: {:?}", d)),
    }
}

pub fn build_parametric_graph(config: &Value, safetensors_bytes: &[u8]) -> Result<AiGraph> {
    let mut builder = GraphBuilder::new("parametric_model".to_string());

    let st = SafeTensors::deserialize(safetensors_bytes)?;
    let tensors = st.tensors();
    let keys: Vec<&String> = tensors.iter().map(|(k, _)| k).collect();

    let mut num_layers = 0;
    for key in &keys {
        if let Some(idx) = extract_layer_idx(key) {
            if idx >= num_layers {
                num_layers = idx + 1;
            }
        }
    }

    if num_layers == 0 {
        return Err(anyhow!(
            "Could not infer any layers from tensor keys. Is this a transformer?"
        ));
    }

    // Config defaults
    let hidden_size = config
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(4096) as u32;
    let num_heads = config
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(32) as u32;
    let _num_kv_heads = config
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads as u64) as u32;
    let _head_dim = hidden_size / num_heads;

    // Inputs
    let batch = builder.register_var("batch");
    let seq = builder.register_var("seq");
    let _input_ids = builder.add_input("input_ids", DType::INT64, vec![batch.clone(), seq.clone()]);

    // Output (logits)
    let vocab_size = config
        .get("vocab_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as u32;

    let vocab = builder.register_var("vocab");
    // We add a dummy output for now so the graph has something, in reality we'd string together the full nodes
    let logits = builder.add_tensor("logits", DType::F32, vec![batch, seq, vocab]);
    builder.add_output(logits, "logits");

    let mut graph = builder.build();

    graph.metadata.insert(
        "vocab_size".to_string(),
        hologram_ai_common::MetaValue::Int(vocab_size as i64),
    );
    graph.metadata.insert(
        "arch".to_string(),
        hologram_ai_common::MetaValue::Str("parametric_transformer".to_string()),
    );
    graph.metadata.insert(
        "n_layers".to_string(),
        hologram_ai_common::MetaValue::Int(num_layers as i64),
    );

    Ok(graph)
}

fn extract_layer_idx(key: &str) -> Option<usize> {
    let parts: Vec<&str> = key.split('.').collect();
    for (i, part) in parts.iter().enumerate() {
        if (*part == "layers" || *part == "h" || *part == "blocks") && i + 1 < parts.len() {
            if let Ok(idx) = parts[i + 1].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

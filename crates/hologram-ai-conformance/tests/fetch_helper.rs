//! Live-HuggingFace acquisition helper: resolve a repo's shard set and
//! stream-parse each shard's safetensors header into the tensor manifest
//! (names, κ-placeholders, shapes, dtypes) consumed by the streamed
//! weightless compile. The header walk itself is the shared
//! [`hologram_ai_conformance::witness`] parse — the same code path the
//! `safetensors-header-streaming` BDD row holds to the reference crate.

use hologram_ai_common::DType;
use hologram_ai_conformance::witness::parse_streamed_header;
use reqwest::Client;
use serde_json::Value;
use std::convert::TryInto;

pub async fn fetch_authoritative_metadata(
    model_name: &str,
) -> (String, Vec<String>, Vec<String>, Vec<Vec<u64>>, Vec<DType>) {
    let client = Client::new();
    let config_url = format!("https://huggingface.co/{}/raw/main/config.json", model_name);
    let config_json = client
        .get(&config_url)
        .send()
        .await
        .expect("fetching config.json")
        .text()
        .await
        .expect("reading config.json body");

    let mut keys = Vec::new();
    let mut kappas = Vec::new();
    let mut shapes = Vec::new();
    let mut dtypes = Vec::new();

    let index_url = format!(
        "https://huggingface.co/{}/raw/main/model.safetensors.index.json",
        model_name
    );
    let index_resp = client
        .get(&index_url)
        .send()
        .await
        .expect("querying the shard index");

    let mut shard_files = std::collections::HashSet::new();
    if index_resp.status().is_success() {
        let index_text = index_resp.text().await.expect("reading the shard index");
        let index: Value = serde_json::from_str(&index_text).expect("parsing the shard index");
        if let Some(weight_map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for v in weight_map.values() {
                if let Some(f) = v.as_str() {
                    shard_files.insert(f.to_string());
                }
            }
        }
    } else {
        shard_files.insert("model.safetensors".to_string());
    }

    for file in shard_files {
        let st_url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            model_name, file
        );
        let len_resp = client
            .get(&st_url)
            .header("Range", "bytes=0-7")
            .send()
            .await
            .expect("fetching the 8-byte header length");
        let len_bytes = len_resp.bytes().await.expect("reading the header length");
        let header_len = u64::from_le_bytes(
            len_bytes[0..8]
                .try_into()
                .expect("an 8-byte range response"),
        );

        let header_resp = client
            .get(&st_url)
            .header("Range", format!("bytes=8-{}", 7 + header_len))
            .send()
            .await
            .expect("fetching the JSON header");
        let header_bytes = header_resp.bytes().await.expect("reading the JSON header");

        for entry in parse_streamed_header(&header_bytes).expect("stream-parsing the header") {
            kappas.push(format!("blake3:{}", entry.name));
            keys.push(entry.name);
            dtypes.push(entry.dtype);
            shapes.push(entry.shape);
        }
    }

    (config_json, keys, kappas, shapes, dtypes)
}

/// The second registered architecture family (Qwen2: attention QKV biases,
/// tied embeddings) streams through the same weightless compile as the
/// canonical Llama family. The repo id comes from the model registry's
/// use-case table — never hard-coded here.
#[tokio::test]
async fn test_second_family_streamed_compile() {
    let model = hologram_ai_model::Model::load().expect("the conceptual model loads");
    let usecase = model
        .usecase("qwen2_5-0_5b")
        .expect("the qwen2_5-0_5b use-case is registered");
    let (config, keys, kappas, shapes, dtypes) =
        fetch_authoritative_metadata(&usecase.hf_repo).await;
    let source = hologram_ai::compiler::ModelSource::SafetensorsStreamed {
        config_json: config,
        keys,
        kappas,
        shapes,
        dtypes,
    };
    let compiler = hologram_ai::compiler::ModelCompiler::default();
    let prepared = compiler
        .prepare(source)
        .expect("prepare the streamed manifest");
    let _ = prepared
        .compile_at(Some(128), Default::default())
        .expect("weightless compile at seq 128");
}

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

/// Streamed authority metadata plus the weight-byte audit of the walk that
/// produced it.
pub struct StreamedAuthority {
    pub config_json: String,
    pub keys: Vec<String>,
    pub kappas: Vec<String>,
    pub shapes: Vec<Vec<u64>>,
    pub dtypes: Vec<DType>,
    /// Bytes received from within any shard's data section. The walk issues
    /// only ranged requests for the 8-byte length prefix and the JSON header,
    /// so any non-zero count means weight payload actually flowed (e.g. a
    /// server that ignored the Range header).
    pub weight_bytes_fetched: u64,
}

pub async fn fetch_authoritative_metadata(
    model_name: &str,
) -> (String, Vec<String>, Vec<String>, Vec<Vec<u64>>, Vec<DType>) {
    let m = fetch_authoritative_metadata_at(model_name, "main").await;
    (m.config_json, m.keys, m.kappas, m.shapes, m.dtypes)
}

/// Resolve a repo's shard set at an explicit revision (a pinned commit or a
/// branch name) and stream-parse each shard's safetensors header into the
/// tensor manifest — metadata and headers only, never weight bytes.
pub async fn fetch_authoritative_metadata_at(
    model_name: &str,
    revision: &str,
) -> StreamedAuthority {
    let client = Client::new();
    let config_url = format!(
        "https://huggingface.co/{}/raw/{}/config.json",
        model_name, revision
    );
    let config_resp = client
        .get(&config_url)
        .send()
        .await
        .expect("fetching config.json");
    assert!(
        config_resp.status().is_success(),
        "fetching {config_url}: HTTP {}",
        config_resp.status()
    );
    let config_json = config_resp.text().await.expect("reading config.json body");

    let mut keys = Vec::new();
    let mut kappas = Vec::new();
    let mut shapes = Vec::new();
    let mut dtypes = Vec::new();
    let mut weight_bytes_fetched = 0u64;

    let index_url = format!(
        "https://huggingface.co/{}/raw/{}/model.safetensors.index.json",
        model_name, revision
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
            "https://huggingface.co/{}/resolve/{}/{}",
            model_name, revision, file
        );
        let len_resp = client
            .get(&st_url)
            .header("Range", "bytes=0-7")
            .send()
            .await
            .expect("fetching the 8-byte header length");
        assert!(
            len_resp.status().is_success(),
            "fetching {st_url} length prefix: HTTP {}",
            len_resp.status()
        );
        let len_bytes = len_resp.bytes().await.expect("reading the header length");
        let header_len = u64::from_le_bytes(
            len_bytes[0..8]
                .try_into()
                .expect("an 8-byte range response"),
        );
        // The shard's data section (the weights) begins at 8 + header_len;
        // anything received past it is weight payload.
        weight_bytes_fetched += (len_bytes.len() as u64).saturating_sub(8 + header_len);

        let header_resp = client
            .get(&st_url)
            .header("Range", format!("bytes=8-{}", 7 + header_len))
            .send()
            .await
            .expect("fetching the JSON header");
        assert!(
            header_resp.status().is_success(),
            "fetching {st_url} JSON header: HTTP {}",
            header_resp.status()
        );
        let header_bytes = header_resp.bytes().await.expect("reading the JSON header");
        weight_bytes_fetched += (header_bytes.len() as u64).saturating_sub(header_len);
        let header = &header_bytes[..header_bytes.len().min(header_len as usize)];

        for entry in parse_streamed_header(header).expect("stream-parsing the header") {
            kappas.push(format!("blake3:{}", entry.name));
            keys.push(entry.name);
            dtypes.push(entry.dtype);
            shapes.push(entry.shape);
        }
    }

    StreamedAuthority {
        config_json,
        keys,
        kappas,
        shapes,
        dtypes,
        weight_bytes_fetched,
    }
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

use hologram_ai_common::DType;
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
        .unwrap()
        .text()
        .await
        .unwrap();

    let mut keys = Vec::new();
    let mut kappas = Vec::new();
    let mut shapes = Vec::new();
    let mut dtypes = Vec::new();

    let index_url = format!(
        "https://huggingface.co/{}/raw/main/model.safetensors.index.json",
        model_name
    );
    let index_resp = client.get(&index_url).send().await.unwrap();

    let mut shard_files = std::collections::HashSet::new();
    if index_resp.status().is_success() {
        let index_text = index_resp.text().await.unwrap();
        let index: Value = serde_json::from_str(&index_text).unwrap();
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
            .unwrap();
        let len_bytes = len_resp.bytes().await.unwrap();
        let header_len = u64::from_le_bytes(len_bytes[0..8].try_into().unwrap());

        let header_resp = client
            .get(&st_url)
            .header("Range", format!("bytes=8-{}", 7 + header_len))
            .send()
            .await
            .unwrap();
        let header_bytes = header_resp.bytes().await.unwrap();
        let header: Value = serde_json::from_slice(&header_bytes).unwrap();

        if let Some(obj) = header.as_object() {
            for (k, v) in obj {
                if k != "__metadata__" {
                    keys.push(k.clone());
                    kappas.push(format!("blake3:{}", k));

                    if let Some(meta) = v.as_object() {
                        let dtype_str = meta.get("dtype").and_then(|d| d.as_str()).unwrap_or("F32");
                        let dtype = match dtype_str {
                            "F32" => DType::F32,
                            "F16" => DType::F16,
                            "BF16" => DType::BF16,
                            "I64" => DType::INT64,
                            "I32" => DType::INT32,
                            "I8" => DType::INT8,
                            "U8" => DType::U8,
                            "BOOL" => DType::BOOL,
                            _ => DType::F32,
                        };
                        dtypes.push(dtype);

                        let shape = meta
                            .get("shape")
                            .and_then(|s| s.as_array())
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|s| s.as_u64().unwrap_or(1))
                            .collect();
                        shapes.push(shape);
                    } else {
                        dtypes.push(DType::F32);
                        shapes.push(vec![1]);
                    }
                }
            }
        }
    }

    (config_json, keys, kappas, shapes, dtypes)
}

#[tokio::test]
async fn test_phi4_streamed_compile() {
    let (config, keys, kappas, shapes, dtypes) =
        fetch_authoritative_metadata("microsoft/phi-4").await;
    let source = hologram_ai::compiler::ModelSource::SafetensorsStreamed {
        config_json: config,
        keys,
        kappas,
        shapes,
        dtypes,
    };
    let compiler = hologram_ai::compiler::ModelCompiler::default();
    let prepared = compiler.prepare(source).unwrap();
    let _ = prepared.compile_at(Some(128), Default::default()).unwrap();
}

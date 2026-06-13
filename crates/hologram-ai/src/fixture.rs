use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Component, Path};

use crate::compiler::{ArchiveSections, ModelMetadata};
use crate::runner::{HoloRunner, PortInfo};

pub(crate) const FIXTURE_SCHEMA_VERSION: u32 = 1;
pub(crate) const EMBEDDED_FIXTURE_PREFIX: &str = "fixture/";
pub(crate) const EMBEDDED_FIXTURE_MANIFEST_EXT: &str = "fixture/manifest.json";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExternalFixtureManifest {
    pub schema_version: u32,
    pub preset: String,
    pub archive: ArchiveManifest,
    pub source_model: String,
    pub model_metadata: ModelMetadataManifest,
    pub inputs: Vec<TensorManifest>,
    pub outputs: Vec<TensorManifest>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct EmbeddedFixtureManifest {
    pub schema_version: u32,
    pub preset: String,
    pub source_model: String,
    pub model_metadata: ModelMetadataManifest,
    pub inputs: Vec<TensorManifest>,
    pub outputs: Vec<TensorManifest>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ArchiveManifest {
    pub file: String,
    pub bytes: u64,
    pub node_count: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ModelMetadataManifest {
    pub arch: Option<String>,
    pub vocab_size: Option<u32>,
    pub context_len: Option<u32>,
    pub n_layers: Option<u32>,
    pub n_embd: Option<u32>,
    pub n_kv_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub kappa_label: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TensorManifest {
    pub index: usize,
    pub name: String,
    pub dtype_tag: u8,
    pub dtype_name: String,
    pub element_count: usize,
    pub shape: Vec<usize>,
    pub bytes_file: String,
    pub kappa: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EmbeddedFixtureBundle {
    pub manifest: EmbeddedFixtureManifest,
    pub inputs: Vec<Vec<u8>>,
    pub outputs: Vec<Vec<u8>>,
}

impl ModelMetadataManifest {
    pub(crate) fn from_model_metadata(metadata: &ModelMetadata) -> Self {
        Self {
            arch: metadata.arch.clone(),
            vocab_size: metadata.vocab_size,
            context_len: metadata.context_len,
            n_layers: metadata.n_layers,
            n_embd: metadata.n_embd,
            n_kv_heads: metadata.n_kv_heads,
            head_dim: metadata.head_dim,
            kappa_label: metadata.kappa_label.clone(),
        }
    }
}

pub(crate) fn tensor_manifest(
    index: usize,
    port: &PortInfo,
    bytes: &[u8],
    bytes_file: String,
) -> TensorManifest {
    TensorManifest {
        index,
        name: port.name.clone(),
        dtype_tag: port.dtype,
        dtype_name: dtype_name(port.dtype).to_string(),
        element_count: port.element_count,
        shape: port.shape.clone(),
        bytes_file,
        kappa: blake3_kappa(bytes),
    }
}

pub(crate) fn build_embedded_fixture_sections(
    manifest: &EmbeddedFixtureManifest,
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Result<ArchiveSections> {
    validate_manifest_tensor_counts(manifest, inputs, outputs)?;

    let mut sections = ArchiveSections::new();
    let manifest_json =
        serde_json::to_vec(manifest).context("serializing embedded fixture manifest")?;
    sections.add_extension(EMBEDDED_FIXTURE_MANIFEST_EXT, manifest_json);
    add_tensor_sections(&mut sections, &manifest.inputs, inputs, "input")?;
    add_tensor_sections(&mut sections, &manifest.outputs, outputs, "output")?;
    Ok(sections)
}

pub(crate) fn read_embedded_fixture(runner: &HoloRunner) -> Result<EmbeddedFixtureBundle> {
    let manifest_bytes = runner
        .extension(EMBEDDED_FIXTURE_MANIFEST_EXT)
        .context("archive does not contain an embedded fixture manifest")?;
    let manifest: EmbeddedFixtureManifest =
        serde_json::from_slice(manifest_bytes).context("parsing embedded fixture manifest")?;
    let inputs = read_tensor_sections(runner, &manifest.inputs, "input")?;
    let outputs = read_tensor_sections(runner, &manifest.outputs, "output")?;
    Ok(EmbeddedFixtureBundle {
        manifest,
        inputs,
        outputs,
    })
}

pub(crate) fn tensor_file_stem(prefix: &str, index: usize, name: &str) -> String {
    let stem = sanitize_file_component(name);
    if stem.is_empty() {
        return format!("{prefix}_{index}");
    }
    format!("{prefix}_{index}_{stem}")
}

pub(crate) fn sanitize_file_component(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

pub(crate) fn dtype_name(tag: u8) -> &'static str {
    match tag {
        0 => "bool",
        1 => "u8",
        2 => "i8",
        3 => "u64",
        4 => "i32",
        5 => "i64",
        6 => "f16",
        7 => "bf16",
        8 => "f32",
        9 => "f64",
        10 => "i4",
        _ => "unknown",
    }
}

pub(crate) fn blake3_kappa(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn validate_manifest_tensor_counts(
    manifest: &EmbeddedFixtureManifest,
    inputs: &[Vec<u8>],
    outputs: &[Vec<u8>],
) -> Result<()> {
    validate_tensor_count("input", manifest.inputs.len(), inputs.len())?;
    validate_tensor_count("output", manifest.outputs.len(), outputs.len())
}

fn validate_tensor_count(kind: &str, expected: usize, actual: usize) -> Result<()> {
    if expected == actual {
        return Ok(());
    }
    bail!("embedded fixture has {expected} {kind} manifest entry(s) but {actual} {kind} buffer(s)")
}

fn add_tensor_sections(
    sections: &mut ArchiveSections,
    manifests: &[TensorManifest],
    buffers: &[Vec<u8>],
    kind: &str,
) -> Result<()> {
    for (tensor, bytes) in manifests.iter().zip(buffers.iter()) {
        validate_tensor_kappa(kind, tensor, bytes)?;
        let key = embedded_tensor_extension_key(&tensor.bytes_file)?;
        sections.add_extension(key, bytes.clone());
    }
    Ok(())
}

fn read_tensor_sections(
    runner: &HoloRunner,
    manifests: &[TensorManifest],
    kind: &str,
) -> Result<Vec<Vec<u8>>> {
    manifests
        .iter()
        .map(|tensor| {
            let key = embedded_tensor_extension_key(&tensor.bytes_file)?;
            let bytes = runner.extension(&key).with_context(|| {
                format!(
                    "archive missing embedded {kind}[{}] bytes at {key}",
                    tensor.index
                )
            })?;
            validate_tensor_kappa(kind, tensor, bytes)?;
            Ok(bytes.to_vec())
        })
        .collect()
}

fn validate_tensor_kappa(kind: &str, tensor: &TensorManifest, bytes: &[u8]) -> Result<()> {
    let actual = blake3_kappa(bytes);
    if actual == tensor.kappa {
        return Ok(());
    }
    bail!(
        "embedded {kind}[{}] bytes at {:?} re-addressed to {actual}, expected {}",
        tensor.index,
        tensor.bytes_file,
        tensor.kappa
    )
}

fn embedded_tensor_extension_key(bytes_file: &str) -> Result<String> {
    let relative = normalized_relative_path(bytes_file)?;
    Ok(format!("{EMBEDDED_FIXTURE_PREFIX}{relative}"))
}

fn normalized_relative_path(path: &str) -> Result<String> {
    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            _ => bail!("fixture path must be a relative normal path, got {path:?}"),
        }
    }
    if parts.is_empty() {
        bail!("fixture path must not be empty");
    }
    Ok(parts.join("/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::run_fixture::{execute as run_fixture_execute, RunFixtureArgs};
    use crate::{HoloArchive, ModelCompiler, ModelSource};
    use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};
    use std::collections::HashMap;

    fn bert_fixture_graph() -> AiGraph {
        let (input_ids, attention_mask, token_type_ids) = (0u32, 1u32, 2u32);
        let (out_ids, out_mask, out_segments) = (3u32, 4u32, 5u32);
        let mut tensor_info = HashMap::new();
        for tid in [
            input_ids,
            attention_mask,
            token_type_ids,
            out_ids,
            out_mask,
            out_segments,
        ] {
            tensor_info.insert(
                tid,
                TensorInfo::new(DType::INT64, shape_from_concrete(&[1, 8])),
            );
        }

        AiGraph {
            name: "embedded-fixture-host".into(),
            nodes: vec![
                AiNode::new(0, AiOp::Identity, vec![input_ids], vec![out_ids]),
                AiNode::new(1, AiOp::Identity, vec![attention_mask], vec![out_mask]),
                AiNode::new(2, AiOp::Identity, vec![token_type_ids], vec![out_segments]),
            ],
            inputs: vec![input_ids, attention_mask, token_type_ids],
            outputs: vec![out_ids, out_mask, out_segments],
            input_names: vec![
                "input_ids".into(),
                "attention_mask".into(),
                "token_type_ids".into(),
            ],
            output_names: vec![
                "input_ids_out".into(),
                "attention_mask_out".into(),
                "token_type_ids_out".into(),
            ],
            params: HashMap::new(),
            tensor_info,
            metadata: HashMap::new(),
            warnings: Vec::new(),
            dim_vars: Default::default(),
            shape_constraints: Default::default(),
            subgraphs: HashMap::new(),
            tensor_names: HashMap::new(),
            topo_cache: Default::default(),
        }
    }

    fn i64_le(values: &[i64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn bert_fixture_buffers() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let input_ids = i64_le(&[101, 2023, 2003, 1037, 3231, 102, 0, 0]);
        let attention_mask = i64_le(&[1, 1, 1, 1, 1, 1, 0, 0]);
        let token_type_ids = i64_le(&[0, 0, 0, 0, 0, 0, 0, 0]);
        let inputs = vec![
            input_ids.clone(),
            attention_mask.clone(),
            token_type_ids.clone(),
        ];
        let outputs = vec![input_ids, attention_mask, token_type_ids];
        (inputs, outputs)
    }

    fn fixture_manifest() -> EmbeddedFixtureManifest {
        let ports = [
            PortInfo {
                name: "input_ids".into(),
                dtype: 5,
                element_count: 8,
                shape: vec![1, 8],
            },
            PortInfo {
                name: "attention_mask".into(),
                dtype: 5,
                element_count: 8,
                shape: vec![1, 8],
            },
            PortInfo {
                name: "token_type_ids".into(),
                dtype: 5,
                element_count: 8,
                shape: vec![1, 8],
            },
        ];
        let (inputs, outputs) = bert_fixture_buffers();
        EmbeddedFixtureManifest {
            schema_version: FIXTURE_SCHEMA_VERSION,
            preset: "bert-base-uncased".into(),
            source_model: "memory://bert-fixture-host".into(),
            model_metadata: ModelMetadataManifest::from_model_metadata(&ModelMetadata {
                arch: Some("bert".into()),
                vocab_size: Some(30_522),
                context_len: Some(8),
                n_layers: None,
                n_embd: None,
                n_kv_heads: None,
                head_dim: None,
                kappa_label: None,
            }),
            inputs: ports
                .iter()
                .enumerate()
                .map(|(index, port)| {
                    tensor_manifest(
                        index,
                        port,
                        &inputs[index],
                        format!(
                            "inputs/{}.bin",
                            tensor_file_stem("input", index, &port.name)
                        ),
                    )
                })
                .collect(),
            outputs: ports
                .iter()
                .enumerate()
                .map(|(index, port)| {
                    tensor_manifest(
                        index,
                        port,
                        &outputs[index],
                        format!(
                            "outputs/{}.bin",
                            tensor_file_stem("output", index, &port.name)
                        ),
                    )
                })
                .collect(),
        }
    }

    fn compile_embedded_fixture_archive() -> (
        HoloArchive,
        EmbeddedFixtureManifest,
        Vec<Vec<u8>>,
        Vec<Vec<u8>>,
    ) {
        let manifest = fixture_manifest();
        let (inputs, outputs) = bert_fixture_buffers();
        let sections =
            build_embedded_fixture_sections(&manifest, &inputs, &outputs).expect("build sections");
        let archive = ModelCompiler::default()
            .compile_with_sections(ModelSource::AiGraph(bert_fixture_graph()), sections)
            .expect("compile archive with embedded fixture");
        (archive, manifest, inputs, outputs)
    }

    #[test]
    fn embedded_fixture_sections_roundtrip() {
        let (archive, manifest, inputs, outputs) = compile_embedded_fixture_archive();
        let runner = HoloRunner::from_bytes(archive.bytes).expect("load embedded fixture archive");
        let embedded = read_embedded_fixture(&runner).expect("read embedded fixture");

        assert_eq!(embedded.manifest, manifest);
        assert_eq!(embedded.inputs, inputs);
        assert_eq!(embedded.outputs, outputs);
    }

    #[test]
    fn run_fixture_executes_embedded_inputs_and_matches_outputs() {
        let (archive, _, _, _) = compile_embedded_fixture_archive();
        let dir = tempfile::tempdir().expect("tempdir");
        let archive_path = dir.path().join("fixture.holo");
        archive.save(&archive_path).expect("save archive");

        run_fixture_execute(RunFixtureArgs {
            file: archive_path,
            verbose: false,
            decode_top_k: None,
            decode_positions: Vec::new(),
            decode_output_index: 0,
            masked_top_k: None,
        })
        .expect("run embedded fixture");
    }
}

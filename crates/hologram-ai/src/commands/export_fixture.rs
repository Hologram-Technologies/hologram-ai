//! `hologram-ai export-fixture` — compile a model and emit a deterministic
//! holospaces-friendly fixture directory.
//!
//! The fixture packages:
//! - the compiled `.holo` archive
//! - deterministic typed input buffers for a known model preset
//! - expected output buffers and their κ-labels
//! - a manifest describing port order, dtypes, shapes, and file layout

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use hologram_archive::ContentLabel;
use serde::Serialize;
use std::path::{Path, PathBuf};

use crate::commands::{build_model_compiler, CompileCliOptions};
use crate::compiler::ModelSource;
use crate::runner::{HoloRunner, PortInfo};

#[derive(Args, Debug)]
pub struct ExportFixtureArgs {
    /// Path to the input ONNX model file.
    #[arg(short, long, value_name = "FILE")]
    pub model: PathBuf,
    /// Output directory for the fixture bundle.
    #[arg(short, long, value_name = "DIR")]
    pub output: PathBuf,
    /// Archive filename stem (the `.holo` extension is appended).
    /// Defaults to the model file stem.
    #[arg(long, value_name = "STEM")]
    pub name: Option<String>,
    /// Deterministic input preset to synthesize.
    #[arg(long, value_enum, default_value_t = FixturePreset::BertBaseUncased)]
    pub preset: FixturePreset,
    /// Fixed sequence length for compilation (default: model's context_length).
    #[arg(long, value_name = "N")]
    pub seq_len: Option<u64>,
    /// Weight quantization scheme: 'none'/'f32', 'int8', 'int4'.
    #[arg(long, value_name = "SCHEME")]
    pub quantize: Option<String>,
    /// Scale spatial dims (H, W) of 4-D inputs by this factor for lower
    /// activation memory (vision/diffusion models).
    #[arg(long, value_name = "N")]
    pub spatial_scale: Option<u32>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum FixturePreset {
    #[value(name = "bert-base-uncased")]
    BertBaseUncased,
}

impl FixturePreset {
    fn as_str(self) -> &'static str {
        match self {
            Self::BertBaseUncased => "bert-base-uncased",
        }
    }

    fn build_inputs(self, ports: &[PortInfo]) -> Result<Vec<Vec<u8>>> {
        match self {
            Self::BertBaseUncased => build_bert_inputs(ports),
        }
    }
}

#[derive(Serialize)]
struct FixtureManifest {
    schema_version: u32,
    preset: String,
    archive: ArchiveManifest,
    source_model: String,
    model_metadata: ModelMetadataManifest,
    inputs: Vec<TensorManifest>,
    outputs: Vec<TensorManifest>,
}

#[derive(Serialize)]
struct ArchiveManifest {
    file: String,
    bytes: u64,
    node_count: usize,
}

#[derive(Serialize)]
struct ModelMetadataManifest {
    arch: Option<String>,
    vocab_size: Option<u32>,
    context_len: Option<u32>,
    n_layers: Option<u32>,
    n_embd: Option<u32>,
    n_kv_heads: Option<u32>,
    head_dim: Option<u32>,
    kappa_label: Option<String>,
}

#[derive(Serialize)]
struct TensorManifest {
    index: usize,
    name: String,
    dtype_tag: u8,
    dtype_name: String,
    element_count: usize,
    shape: Vec<usize>,
    bytes_file: String,
    kappa: String,
}

pub fn execute(args: ExportFixtureArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("creating fixture directory {:?}", args.output))?;

    let compiler = build_model_compiler(&CompileCliOptions {
        seq_len: args.seq_len,
        quantize: args.quantize.clone(),
        spatial_scale: args.spatial_scale,
    })?;
    let stem = archive_stem(&args.model, args.name.as_deref());
    let archive_path = args.output.join(format!("{stem}.holo"));
    let archive = compiler
        .compile(ModelSource::OnnxPath(args.model.clone()))
        .with_context(|| format!("compiling {:?}", args.model))?;
    archive.save(&archive_path)?;

    let archive_bytes = std::fs::metadata(&archive_path)
        .with_context(|| format!("reading archive metadata {archive_path:?}"))?
        .len();
    let model_metadata = ModelMetadataManifest {
        arch: archive.metadata.arch,
        vocab_size: archive.metadata.vocab_size,
        context_len: archive.metadata.context_len,
        n_layers: archive.metadata.n_layers,
        n_embd: archive.metadata.n_embd,
        n_kv_heads: archive.metadata.n_kv_heads,
        head_dim: archive.metadata.head_dim,
        kappa_label: archive.metadata.kappa_label,
    };
    let node_count = archive.stats.node_count;

    let mut runner = HoloRunner::from_path(&archive_path, None)?;
    let input_ports = runner.input_port_info();
    let output_ports = runner.output_port_info();
    let input_bytes = args.preset.build_inputs(&input_ports)?;

    let input_labels = intern_inputs(&mut runner, &input_bytes);
    let output_labels = runner
        .execute_addressed(&input_labels)
        .context("executing fixture inputs")?;

    let input_manifest = write_input_artifacts(&args.output, &input_ports, &input_bytes)?;
    let output_manifest =
        write_output_artifacts(&args.output, &output_ports, &output_labels, &runner)?;
    let manifest = FixtureManifest {
        schema_version: 1,
        preset: args.preset.as_str().to_string(),
        archive: ArchiveManifest {
            file: archive_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("{stem}.holo")),
            bytes: archive_bytes,
            node_count,
        },
        source_model: args.model.to_string_lossy().into_owned(),
        model_metadata,
        inputs: input_manifest,
        outputs: output_manifest,
    };

    let manifest_path = args.output.join("manifest.json");
    let manifest_json =
        serde_json::to_vec_pretty(&manifest).context("serializing fixture manifest")?;
    std::fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("writing fixture manifest {manifest_path:?}"))?;

    println!(
        "Exported {} fixture to {:?} (archive: {:?}, {} input(s), {} output(s))",
        args.preset.as_str(),
        args.output,
        archive_path.file_name().unwrap_or_default(),
        manifest.inputs.len(),
        manifest.outputs.len()
    );
    Ok(())
}

fn archive_stem(model: &Path, name: Option<&str>) -> String {
    name.map(str::to_owned).unwrap_or_else(|| {
        model
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string())
    })
}

fn build_bert_inputs(ports: &[PortInfo]) -> Result<Vec<Vec<u8>>> {
    let seq_len = bert_seq_len(ports)?;
    let tokens = bert_tokens(seq_len)?;
    let mask = bert_attention_mask(seq_len);
    let segments = vec![0i64; seq_len];

    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let values = match bert_input_kind(index, &port.name)? {
                BertInputKind::InputIds => tokens.as_slice(),
                BertInputKind::AttentionMask => mask.as_slice(),
                BertInputKind::TokenTypeIds => segments.as_slice(),
            };
            encode_integer_tensor(values, port)
        })
        .collect()
}

fn bert_seq_len(ports: &[PortInfo]) -> Result<usize> {
    let ids = ports
        .iter()
        .enumerate()
        .find_map(|(index, port)| {
            matches!(
                bert_input_kind(index, &port.name),
                Ok(BertInputKind::InputIds)
            )
            .then_some(port)
        })
        .context("BERT preset requires an input_ids port")?;
    let seq_len = ids.shape.last().copied().unwrap_or(ids.element_count);
    if seq_len == 0 {
        bail!("BERT preset requires a non-empty input_ids shape");
    }
    Ok(seq_len)
}

fn bert_tokens(seq_len: usize) -> Result<Vec<i64>> {
    let prompt = [101, 2023, 2003, 1037, 3231, 102];
    if seq_len < prompt.len() {
        bail!(
            "BERT preset needs seq_len >= {} to encode the canonical prompt, got {seq_len}",
            prompt.len()
        );
    }
    let mut out = vec![0i64; seq_len];
    out[..prompt.len()].copy_from_slice(&prompt);
    Ok(out)
}

fn bert_attention_mask(seq_len: usize) -> Vec<i64> {
    let real = 6usize.min(seq_len);
    let mut out = vec![0i64; seq_len];
    out[..real].fill(1);
    out
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BertInputKind {
    InputIds,
    AttentionMask,
    TokenTypeIds,
}

fn bert_input_kind(index: usize, name: &str) -> Result<BertInputKind> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("input_ids") || index == 0 {
        return Ok(BertInputKind::InputIds);
    }
    if lower.contains("attention_mask") || lower == "mask" || index == 1 {
        return Ok(BertInputKind::AttentionMask);
    }
    if lower.contains("token_type_ids") || lower.contains("segment") || index == 2 {
        return Ok(BertInputKind::TokenTypeIds);
    }
    bail!("unrecognized BERT input port {index}: {name:?}")
}

fn encode_integer_tensor(values: &[i64], port: &PortInfo) -> Result<Vec<u8>> {
    if values.len() != port.element_count {
        bail!(
            "port {:?} expects {} element(s), but the preset produced {}",
            port.name,
            port.element_count,
            values.len()
        );
    }
    let mut out = Vec::with_capacity(port.element_count * integer_dtype_width(port.dtype)?);
    for value in values {
        match port.dtype {
            1 => {
                out.push(u8::try_from(*value).with_context(|| {
                    format!("value {value} does not fit u8 for {:?}", port.name)
                })?)
            }
            2 => out.push(
                i8::try_from(*value)
                    .with_context(|| format!("value {value} does not fit i8 for {:?}", port.name))?
                    as u8,
            ),
            4 => out.extend_from_slice(
                &i32::try_from(*value)
                    .with_context(|| format!("value {value} does not fit i32 for {:?}", port.name))?
                    .to_le_bytes(),
            ),
            5 => out.extend_from_slice(&value.to_le_bytes()),
            other => bail!(
                "BERT preset needs integer inputs, but port {:?} has unsupported dtype tag {other}",
                port.name
            ),
        }
    }
    Ok(out)
}

fn integer_dtype_width(dtype: u8) -> Result<usize> {
    Ok(match dtype {
        1 | 2 => 1,
        4 => 4,
        5 => 8,
        other => bail!("unsupported integer dtype tag {other}"),
    })
}

fn intern_inputs(runner: &mut HoloRunner, inputs: &[Vec<u8>]) -> Vec<ContentLabel> {
    inputs
        .iter()
        .map(|bytes| runner.intern_input(bytes))
        .collect()
}

fn write_input_artifacts(
    output_dir: &Path,
    ports: &[PortInfo],
    inputs: &[Vec<u8>],
) -> Result<Vec<TensorManifest>> {
    let input_dir = output_dir.join("inputs");
    std::fs::create_dir_all(&input_dir)
        .with_context(|| format!("creating input fixture directory {input_dir:?}"))?;

    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let stem = tensor_file_stem("input", index, &port.name);
            let file_name = format!("{stem}.bin");
            let relative = format!("inputs/{file_name}");
            let path = input_dir.join(&file_name);
            std::fs::write(&path, &inputs[index])
                .with_context(|| format!("writing input fixture {path:?}"))?;
            Ok(tensor_manifest(index, port, &inputs[index], relative))
        })
        .collect()
}

fn write_output_artifacts(
    output_dir: &Path,
    ports: &[PortInfo],
    labels: &[ContentLabel],
    runner: &HoloRunner,
) -> Result<Vec<TensorManifest>> {
    let output_dir = output_dir.join("outputs");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output fixture directory {output_dir:?}"))?;

    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let stem = tensor_file_stem("output", index, &port.name);
            let bytes_file = format!("{stem}.bin");
            let kappa_file = format!("{stem}.kappa");
            let bytes_relative = format!("outputs/{bytes_file}");
            let bytes_path = output_dir.join(&bytes_file);
            let kappa_path = output_dir.join(&kappa_file);
            let bytes = runner
                .resolve(&labels[index])
                .with_context(|| format!("resolving output label {}", labels[index].as_str()))?;
            std::fs::write(&bytes_path, bytes)
                .with_context(|| format!("writing output fixture {bytes_path:?}"))?;
            let kappa = blake3_kappa(bytes);
            std::fs::write(&kappa_path, &kappa)
                .with_context(|| format!("writing output label {kappa_path:?}"))?;
            Ok(tensor_manifest(index, port, bytes, bytes_relative))
        })
        .collect()
}

fn tensor_manifest(
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

fn tensor_file_stem(prefix: &str, index: usize, name: &str) -> String {
    let stem = sanitize_file_component(name);
    if stem.is_empty() {
        return format!("{prefix}_{index}");
    }
    format!("{prefix}_{index}_{stem}")
}

fn sanitize_file_component(name: &str) -> String {
    name.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn dtype_name(tag: u8) -> &'static str {
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

fn blake3_kappa(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_port(name: &str, dtype: u8, element_count: usize, shape: Vec<usize>) -> PortInfo {
        PortInfo {
            name: name.to_string(),
            dtype,
            element_count,
            shape,
        }
    }

    fn decode_i64(bytes: &[u8]) -> Vec<i64> {
        bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("8-byte i64 chunk")))
            .collect()
    }

    #[test]
    fn bert_preset_matches_canonical_prompt() {
        let ports = vec![
            test_port("input_ids", 5, 8, vec![1, 8]),
            test_port("attention_mask", 5, 8, vec![1, 8]),
            test_port("token_type_ids", 5, 8, vec![1, 8]),
        ];

        let inputs = FixturePreset::BertBaseUncased
            .build_inputs(&ports)
            .expect("build BERT inputs");

        assert_eq!(
            decode_i64(&inputs[0]),
            vec![101, 2023, 2003, 1037, 3231, 102, 0, 0]
        );
        assert_eq!(decode_i64(&inputs[1]), vec![1, 1, 1, 1, 1, 1, 0, 0]);
        assert_eq!(decode_i64(&inputs[2]), vec![0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn sanitize_file_component_rewrites_symbols() {
        assert_eq!(sanitize_file_component("foo/bar:baz"), "foo_bar_baz");
        assert_eq!(tensor_file_stem("output", 0, ""), "output_0");
    }
}

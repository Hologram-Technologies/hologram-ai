//! `hologram-ai export-fixture` — compile a model and emit a deterministic
//! fixture witness for both archive-first and file-first consumers.
//!
//! The fixture packages:
//! - the compiled `.holo` archive
//! - the same fixture embedded in archive extension sections
//! - deterministic typed input buffers for a known model preset
//! - expected output buffers and their κ-labels
//! - a manifest describing port order, dtypes, shapes, and file layout

use anyhow::{bail, Context, Result};
use clap::{Args, ValueEnum};
use hologram_archive::ContentLabel;
use std::path::{Path, PathBuf};

use crate::commands::{build_model_compiler, CompileCliOptions};
use crate::compiler::{ArchiveSections, ModelSource};
use crate::fixture::{
    build_embedded_fixture_sections, tensor_file_stem, tensor_manifest, ArchiveManifest,
    EmbeddedFixtureManifest, ExternalFixtureManifest, ModelMetadataManifest, TensorManifest,
    FIXTURE_SCHEMA_VERSION,
};
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
    #[value(name = "bert-base-uncased-masked")]
    BertBaseUncasedMasked,
}

impl FixturePreset {
    fn as_str(self) -> &'static str {
        match self {
            Self::BertBaseUncased => "bert-base-uncased",
            Self::BertBaseUncasedMasked => "bert-base-uncased-masked",
        }
    }

    fn build_inputs(self, ports: &[PortInfo]) -> Result<Vec<Vec<u8>>> {
        match self {
            Self::BertBaseUncased | Self::BertBaseUncasedMasked => build_bert_inputs(ports, self),
        }
    }
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
    let prepared = compiler
        .prepare(ModelSource::OnnxPath(args.model.clone()))
        .with_context(|| format!("preparing {:?}", args.model))?;
    let provisional_archive = prepared
        .clone()
        .compile_at(args.seq_len, ArchiveSections::new())
        .with_context(|| format!("compiling provisional fixture archive for {:?}", args.model))?;
    let provisional_model_metadata =
        ModelMetadataManifest::from_model_metadata(&provisional_archive.metadata);

    let mut runner = HoloRunner::from_bytes(provisional_archive.bytes)
        .context("loading provisional fixture archive")?;
    let input_ports = runner.input_port_info();
    let output_ports = runner.output_port_info();
    let input_bytes = args.preset.build_inputs(&input_ports)?;

    let input_labels = intern_inputs(&mut runner, &input_bytes);
    let output_labels = runner
        .execute_addressed(&input_labels)
        .context("executing fixture inputs")?;
    let output_bytes = resolve_output_bytes(&output_labels, &runner)?;
    let input_manifest = build_input_manifest(&input_ports, &input_bytes);
    let output_manifest = build_output_manifest(&output_ports, &output_bytes);
    let embedded_manifest = EmbeddedFixtureManifest {
        schema_version: FIXTURE_SCHEMA_VERSION,
        preset: args.preset.as_str().to_string(),
        source_model: args.model.to_string_lossy().into_owned(),
        model_metadata: provisional_model_metadata,
        inputs: input_manifest.clone(),
        outputs: output_manifest.clone(),
    };
    let sections =
        build_embedded_fixture_sections(&embedded_manifest, &input_bytes, &output_bytes)?;
    let archive = prepared
        .compile_at(args.seq_len, sections)
        .with_context(|| format!("compiling embedded fixture archive for {:?}", args.model))?;
    archive.save(&archive_path)?;

    let archive_bytes = std::fs::metadata(&archive_path)
        .with_context(|| format!("reading archive metadata {archive_path:?}"))?
        .len();
    let node_count = archive.stats.node_count;

    write_input_artifacts(&args.output, &input_manifest, &input_bytes)?;
    write_output_artifacts(&args.output, &output_manifest, &output_bytes)?;
    let manifest = ExternalFixtureManifest {
        schema_version: FIXTURE_SCHEMA_VERSION,
        preset: embedded_manifest.preset.clone(),
        archive: ArchiveManifest {
            file: archive_path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| format!("{stem}.holo")),
            bytes: archive_bytes,
            node_count,
        },
        source_model: embedded_manifest.source_model.clone(),
        model_metadata: embedded_manifest.model_metadata.clone(),
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

fn build_bert_inputs(ports: &[PortInfo], preset: FixturePreset) -> Result<Vec<Vec<u8>>> {
    let seq_len = bert_seq_len(ports)?;
    let tokens = bert_tokens(seq_len, preset)?;
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

fn bert_tokens(seq_len: usize, preset: FixturePreset) -> Result<Vec<i64>> {
    let prompt = preset.bert_prompt_tokens();
    if seq_len < prompt.len() {
        bail!(
            "BERT preset needs seq_len >= {} to encode the canonical prompt, got {seq_len}",
            prompt.len()
        );
    }
    let mut out = vec![0i64; seq_len];
    out[..prompt.len()].copy_from_slice(prompt);
    Ok(out)
}

impl FixturePreset {
    fn bert_prompt_tokens(self) -> &'static [i64] {
        match self {
            Self::BertBaseUncased => &[101, 2023, 2003, 1037, 3231, 102],
            Self::BertBaseUncasedMasked => &[101, 2023, 2003, 1037, 103, 102],
        }
    }
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

fn build_input_manifest(ports: &[PortInfo], inputs: &[Vec<u8>]) -> Vec<TensorManifest> {
    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let relative = format!(
                "inputs/{}.bin",
                tensor_file_stem("input", index, &port.name)
            );
            tensor_manifest(index, port, &inputs[index], relative)
        })
        .collect()
}

fn build_output_manifest(ports: &[PortInfo], outputs: &[Vec<u8>]) -> Vec<TensorManifest> {
    ports
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let relative = format!(
                "outputs/{}.bin",
                tensor_file_stem("output", index, &port.name)
            );
            tensor_manifest(index, port, &outputs[index], relative)
        })
        .collect()
}

fn resolve_output_bytes(labels: &[ContentLabel], runner: &HoloRunner) -> Result<Vec<Vec<u8>>> {
    labels
        .iter()
        .enumerate()
        .map(|(index, label)| {
            runner
                .resolve(label)
                .map(|bytes| bytes.to_vec())
                .with_context(|| {
                    format!(
                        "resolving fixture output label {} at index {index}",
                        label.as_str()
                    )
                })
        })
        .collect()
}

fn write_input_artifacts(
    output_dir: &Path,
    manifests: &[TensorManifest],
    inputs: &[Vec<u8>],
) -> Result<()> {
    let input_dir = output_dir.join("inputs");
    std::fs::create_dir_all(&input_dir)
        .with_context(|| format!("creating input fixture directory {input_dir:?}"))?;

    manifests
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let file_name = Path::new(&port.bytes_file)
                .file_name()
                .context("input manifest path missing file name")?;
            let path = input_dir.join(file_name);
            std::fs::write(&path, &inputs[index])
                .with_context(|| format!("writing input fixture {path:?}"))?;
            Ok(())
        })
        .collect::<Result<Vec<_>>>()
        .map(|_| ())
}

fn write_output_artifacts(
    output_dir: &Path,
    manifests: &[TensorManifest],
    outputs: &[Vec<u8>],
) -> Result<()> {
    let output_dir = output_dir.join("outputs");
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output fixture directory {output_dir:?}"))?;

    manifests
        .iter()
        .enumerate()
        .map(|(index, port)| {
            let bytes_file = Path::new(&port.bytes_file)
                .file_name()
                .context("output manifest path missing file name")?;
            let stem = bytes_file
                .to_string_lossy()
                .trim_end_matches(".bin")
                .to_string();
            let kappa_file = format!("{stem}.kappa");
            let bytes_path = output_dir.join(bytes_file);
            let kappa_path = output_dir.join(&kappa_file);
            let bytes = &outputs[index];
            std::fs::write(&bytes_path, bytes)
                .with_context(|| format!("writing output fixture {bytes_path:?}"))?;
            let kappa = port.kappa.clone();
            std::fs::write(&kappa_path, &kappa)
                .with_context(|| format!("writing output label {kappa_path:?}"))?;
            Ok(())
        })
        .collect::<Result<Vec<_>>>()
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::sanitize_file_component;

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
    fn bert_masked_preset_injects_mask_token() {
        let ports = vec![
            test_port("input_ids", 5, 8, vec![1, 8]),
            test_port("attention_mask", 5, 8, vec![1, 8]),
            test_port("token_type_ids", 5, 8, vec![1, 8]),
        ];

        let inputs = FixturePreset::BertBaseUncasedMasked
            .build_inputs(&ports)
            .expect("build masked BERT inputs");

        assert_eq!(
            decode_i64(&inputs[0]),
            vec![101, 2023, 2003, 1037, 103, 102, 0, 0]
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

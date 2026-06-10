//! Published-model end-to-end validation against ONNX Runtime (V&V class EE).
//!
//! Uses real ONNX files already present in the shared workspace `models/`
//! directory and compares hologram-ai execution against ONNX Runtime on the
//! same deterministic inputs. Skip-safe when those local models are absent.
#![cfg(feature = "conformance")]

use std::path::{Path, PathBuf};

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::runner::{run_onnx_file_typed, OrtInputTyped};
use hologram_ai_conformance::tolerance::{compare_outputs, Tolerance};
use ort::session::Session;

fn i64_to_le(values: &[i64]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32_to_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("4-byte f32 chunk")))
        .collect()
}

fn model_path(relative: &str) -> Option<PathBuf> {
    let start = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for ancestor in start.ancestors() {
        let candidate = ancestor.join("models").join(relative);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn ort_input_names(model_path: &Path) -> Vec<String> {
    let session = Session::builder()
        .expect("create ORT session builder")
        .commit_from_file(model_path)
        .expect("load model into ORT");
    session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect()
}

fn first_output(holo: Vec<hologram_exec::OutputBuffer>) -> Vec<f32> {
    assert_eq!(holo.len(), 1, "expected exactly one output");
    le_to_f32(&holo[0].bytes)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: Tolerance, label: &str) {
    let cmp = compare_outputs(actual, expected, tolerance);
    assert!(cmp.passed, "{label}: {}", cmp.message);
    println!(
        "{label}: {} elements, max_abs={:.2e}, max_rel={:.2e}",
        cmp.total_elements, cmp.max_abs_error, cmp.max_rel_error
    );
}

fn bert_inputs_for_names(names: &[String], seq: usize) -> Vec<OrtInputTyped> {
    let input_ids = vec![101, 2023, 2003, 1037, 3231, 102, 0, 0];
    let attention_mask = vec![1, 1, 1, 1, 1, 1, 0, 0];
    let token_type_ids = vec![0; seq];

    assert_eq!(input_ids.len(), seq);
    assert_eq!(attention_mask.len(), seq);

    names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            let lower = name.to_ascii_lowercase();
            let (data, label) = if lower.contains("input_ids") || idx == 0 {
                (input_ids.clone(), "input_ids")
            } else if lower.contains("attention_mask") || lower == "mask" || idx == 1 {
                (attention_mask.clone(), "attention_mask")
            } else if lower.contains("token_type_ids") || lower.contains("segment") || idx == 2 {
                (token_type_ids.clone(), "token_type_ids")
            } else {
                panic!("unrecognized BERT input name: {name}");
            };
            OrtInputTyped::I64 {
                name: name.clone(),
                shape: vec![1, seq],
                data,
            }
            .tap(|_| println!("BERT input {label} -> {name}"))
        })
        .collect()
}

trait Tap: Sized {
    fn tap(self, f: impl FnOnce(&Self)) -> Self {
        f(&self);
        self
    }
}

impl<T> Tap for T {}

#[test]
fn bert_base_uncased_matches_ort() {
    let Some(model) = model_path("bert-base-uncased/model.onnx") else {
        eprintln!("skipping: bert-base-uncased/model.onnx not found");
        return;
    };

    let seq = 8usize;
    let input_names = ort_input_names(&model);
    let ort_inputs = bert_inputs_for_names(&input_names, seq);

    let archive = ModelCompiler {
        seq_len_override: Some(seq as u64),
        ..Default::default()
    }
    .compile(ModelSource::OnnxPath(model.clone()))
    .expect("compile bert-base-uncased");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load compiled BERT");

    let mut runner_inputs = Vec::new();
    for (idx, port) in runner.input_port_info().iter().enumerate() {
        let lower = port.name.to_ascii_lowercase();
        let bytes = if lower.contains("input_ids") || idx == 0 {
            i64_to_le(&[101, 2023, 2003, 1037, 3231, 102, 0, 0])
        } else if lower.contains("attention_mask") || lower == "mask" || idx == 1 {
            i64_to_le(&[1, 1, 1, 1, 1, 1, 0, 0])
        } else if lower.contains("token_type_ids") || lower.contains("segment") || idx == 2 {
            i64_to_le(&[0, 0, 0, 0, 0, 0, 0, 0])
        } else {
            panic!("unrecognized compiled BERT input port: {}", port.name);
        };
        runner_inputs.push(bytes);
    }
    let runner_input_refs: Vec<&[u8]> = runner_inputs.iter().map(Vec::as_slice).collect();
    let holo = first_output(runner.execute(&runner_input_refs).expect("execute BERT"));

    let ort_outputs = run_onnx_file_typed(&model, ort_inputs).expect("run BERT in ORT");
    assert!(
        !ort_outputs.is_empty(),
        "ORT produced no f32 outputs for bert-base-uncased"
    );
    let reference = &ort_outputs[0].data;

    assert_eq!(holo.len(), reference.len(), "BERT output length mismatch");
    assert_close(
        &holo,
        reference,
        Tolerance {
            atol: 5e-4,
            rtol: 5e-3,
        },
        "bert-base-uncased vs ORT",
    );
}

#[test]
fn resnet50_matches_ort() {
    let Some(model) = model_path("resnet50-v2-7.onnx") else {
        eprintln!("skipping: resnet50-v2-7.onnx not found");
        return;
    };

    let input_name = ort_input_names(&model)
        .into_iter()
        .next()
        .expect("ResNet should have one input");
    let input: Vec<f32> = (0..(3 * 224 * 224))
        .map(|i| ((i % 251) as f32 / 255.0) - 0.5)
        .collect();

    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model.clone()))
        .expect("compile ResNet-50");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load compiled ResNet");
    let input_bytes = f32_to_le(&input);
    let holo = first_output(runner.execute(&[&input_bytes]).expect("execute ResNet"));

    let ort_outputs = run_onnx_file_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: input_name,
            shape: vec![1, 3, 224, 224],
            data: input,
        }],
    )
    .expect("run ResNet in ORT");
    assert!(
        !ort_outputs.is_empty(),
        "ORT produced no f32 outputs for resnet50-v2-7"
    );
    let reference = &ort_outputs[0].data;

    assert_eq!(holo.len(), reference.len(), "ResNet output length mismatch");
    assert_close(
        &holo,
        reference,
        Tolerance {
            atol: 1e-4,
            rtol: 1e-3,
        },
        "resnet50-v2-7 vs ORT",
    );
}

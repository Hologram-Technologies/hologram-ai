//! Inference benchmarks: TTFT and decode tok/s for TinyLlama ONNX.
//!
//! Requires a pre-compiled `.holo` archive in `models/`. Run:
//!   ./scripts/download-models.sh tinyllama
//!   cargo run --release -- compile models/TinyLlama-1.1B-Chat-v1.0/model.onnx
//!
//! Then:
//!   cargo bench --bench inference
//!
//! Benchmarks are silently skipped if model files are not present.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::{Path, PathBuf};
use std::time::Duration;

use hologram::hologram_archive::section::model_meta::{ModelMetaSection, SECTION_MODEL_META};
use hologram_ai::compiler::HoloRunner;

// ── Model paths ──────────────────────────────────────────────────────────────

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn onnx_holo_path() -> PathBuf {
    workspace_root().join("models/TinyLlama-1.1B-Chat-v1.0/model.holo")
}

// ── Model wrapper ────────────────────────────────────────────────────────────

struct BenchModel {
    runner: HoloRunner,
    input_slot: u32,
    mask_slot: Option<u32>,
    pos_slot: Option<u32>,
    use_kv: bool,
    n_layers: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: usize,
    vocab_size: usize,
}

fn load_meta(runner: &HoloRunner) -> Option<ModelMetaSection> {
    let bytes = runner.archive_bytes();
    let plan = runner.plan();

    // Try loading from plan's section table first.
    if let Some(entry) = plan.sections().find(SECTION_MODEL_META) {
        let offset = entry.offset as usize;
        let size = entry.size as usize;
        if offset + size <= bytes.len() {
            if let Ok(meta) = ModelMetaSection::deserialize_from(&bytes[offset..offset + size]) {
                return Some(meta);
            }
        }
    }

    // Fallback: re-parse the archive to find the section.
    let fallback_plan = hologram::load_from_bytes(bytes).ok()?;
    let entry = fallback_plan.sections().find(SECTION_MODEL_META)?;
    let offset = entry.offset as usize;
    let size = entry.size as usize;
    if offset + size <= bytes.len() {
        ModelMetaSection::deserialize_from(&bytes[offset..offset + size]).ok()
    } else {
        None
    }
}

fn load_model(path: &Path) -> Option<BenchModel> {
    if !path.exists() {
        eprintln!("  skipping: {} not found", path.display());
        return None;
    }

    let runner = HoloRunner::from_path(path, None, None).ok()?;
    let graph = runner.plan().graph();

    let input_slot = graph
        .input_names
        .iter()
        .position(|n| n == "input_ids")
        .unwrap_or(0) as u32;

    let mask_slot = graph
        .input_names
        .iter()
        .position(|n| n == "attention_mask")
        .map(|i| i as u32);

    let pos_slot = graph
        .input_names
        .iter()
        .position(|n| n == "position_ids")
        .map(|i| i as u32);

    let meta = load_meta(&runner);

    let (use_kv, n_layers, n_kv_heads, head_dim, max_seq) = match &meta {
        Some(m) if m.n_layers > 0 => (
            true,
            m.n_layers,
            m.n_kv_heads,
            m.head_dim,
            m.max_seq_len as usize,
        ),
        _ => (false, 0, 0, 0, 2048),
    };

    Some(BenchModel {
        runner,
        input_slot,
        mask_slot,
        pos_slot,
        use_kv,
        n_layers,
        n_kv_heads,
        head_dim,
        max_seq,
        vocab_size: 32000,
    })
}

// ── Input helpers ────────────────────────────────────────────────────────────

fn build_inputs(
    model: &BenchModel,
    token_ids: &[i64],
    kv_write_pos: usize,
    is_decode: bool,
) -> hologram::GraphInputs {
    let mut inputs = hologram::GraphInputs::new();

    let tokens: Vec<i64> = if is_decode {
        vec![*token_ids.last().unwrap_or(&1)]
    } else {
        token_ids.to_vec()
    };
    let seq_len = tokens.len();

    let input_bytes: Vec<u8> = tokens.iter().flat_map(|&v| v.to_le_bytes()).collect();
    inputs.set_with_shape(model.input_slot, input_bytes, vec![1, seq_len]);

    if let Some(slot) = model.mask_slot {
        let mask_bytes: Vec<u8> = (0..seq_len).flat_map(|_| 1i64.to_le_bytes()).collect();
        inputs.set_with_shape(slot, mask_bytes, vec![1, seq_len]);
    }

    if let Some(slot) = model.pos_slot {
        let offset = if is_decode { kv_write_pos as i64 } else { 0i64 };
        let pos_bytes: Vec<u8> = (0..seq_len as i64)
            .map(|i| offset + i)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        inputs.set_with_shape(slot, pos_bytes, vec![1, seq_len]);
    }

    inputs
}

fn argmax_from_output(output: &hologram::GraphOutputs, pos: usize, vocab_size: usize) -> u32 {
    let (_, data) = match output.get(0) {
        Some(v) => v,
        None => return 0,
    };
    let bytes_per_pos = vocab_size * 4;
    let start = pos * bytes_per_pos;
    let end = start + bytes_per_pos;
    if end > data.len() {
        return 0;
    }
    data[start..end]
        .chunks_exact(4)
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let fa = f32::from_le_bytes((*a).try_into().unwrap_or([0; 4]));
            let fb = f32::from_le_bytes((*b).try_into().unwrap_or([0; 4]));
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

// ── Benchmarks ───────────────────────────────────────────────────────────────

fn bench_ttft(c: &mut Criterion, name: &str, model: &BenchModel) {
    let prompt_tokens: Vec<i64> = vec![1, 4, 5, 6, 7, 8, 9, 10];

    let mut group = c.benchmark_group("ttft");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function(BenchmarkId::new("prefill_8tok", name), |b| {
        b.iter(|| {
            let inputs = build_inputs(model, &prompt_tokens, 0, false);
            if model.use_kv {
                let mut kv = hologram::KvCacheState::new(
                    model.n_layers,
                    model.n_kv_heads,
                    model.head_dim,
                    model.max_seq,
                );
                model
                    .runner
                    .execute_with_kv(&inputs, &mut kv)
                    .expect("prefill failed");
            } else {
                model.runner.execute(&inputs).expect("prefill failed");
            }
        });
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion, name: &str, model: &BenchModel) {
    if !model.use_kv {
        return;
    }

    let prompt_tokens: Vec<i64> = vec![1, 4, 5, 6, 7, 8, 9, 10];
    let decode_steps = 20;

    let mut group = c.benchmark_group("decode");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));

    group.bench_function(BenchmarkId::new("20_tokens", name), |b| {
        b.iter(|| {
            let mut kv = hologram::KvCacheState::new(
                model.n_layers,
                model.n_kv_heads,
                model.head_dim,
                model.max_seq,
            );
            let prefill_inputs = build_inputs(model, &prompt_tokens, 0, false);
            let out = model
                .runner
                .execute_with_kv(&prefill_inputs, &mut kv)
                .expect("prefill failed");

            let mut token_ids = prompt_tokens.clone();
            let next = argmax_from_output(&out, prompt_tokens.len() - 1, model.vocab_size);
            token_ids.push(next as i64);

            for _ in 0..decode_steps {
                let inputs = build_inputs(model, &token_ids, kv.write_pos(), true);
                let out = model
                    .runner
                    .execute_with_kv(&inputs, &mut kv)
                    .expect("decode step failed");
                let next = argmax_from_output(&out, 0, model.vocab_size);
                token_ids.push(next as i64);
            }
        });
    });

    group.finish();
}

fn bench_single_decode_step(c: &mut Criterion, name: &str, model: &BenchModel) {
    if !model.use_kv {
        return;
    }

    let prompt_tokens: Vec<i64> = vec![1, 4, 5, 6, 7, 8, 9, 10];

    let mut kv = hologram::KvCacheState::new(
        model.n_layers,
        model.n_kv_heads,
        model.head_dim,
        model.max_seq,
    );
    let prefill_inputs = build_inputs(model, &prompt_tokens, 0, false);
    let out = model
        .runner
        .execute_with_kv(&prefill_inputs, &mut kv)
        .expect("prefill failed");
    let next = argmax_from_output(&out, prompt_tokens.len() - 1, model.vocab_size);
    let mut token_ids = prompt_tokens.clone();
    token_ids.push(next as i64);

    let mut group = c.benchmark_group("decode_step");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function(BenchmarkId::new("single", name), |b| {
        b.iter(|| {
            let inputs = build_inputs(model, &token_ids, kv.write_pos(), true);
            let out = model
                .runner
                .execute_with_kv(&inputs, &mut kv)
                .expect("decode step failed");
            let next = argmax_from_output(&out, 0, model.vocab_size);
            token_ids.push(next as i64);
        });
    });

    group.finish();
}

// ── Criterion setup ──────────────────────────────────────────────────────────

fn inference_benchmarks(c: &mut Criterion) {
    if let Some(model) = load_model(&onnx_holo_path()) {
        bench_ttft(c, "tinyllama_onnx", &model);
        bench_decode(c, "tinyllama_onnx", &model);
        bench_single_decode_step(c, "tinyllama_onnx", &model);
    }
}

criterion_group!(benches, inference_benchmarks);
criterion_main!(benches);

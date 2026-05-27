//! Forward-pass inference benchmark (V&V class PV).
//!
//! In the UOR-native model there is no KV-cache: a `.holo` archive is loaded
//! once into an `InferenceSession`, and every step — prefill or decode — is a
//! single forward `execute`. Autoregressive reuse across steps is structural
//! (content-addressed κ-label elision inside the session), not a host-managed
//! cache. So the honest performance metric is forward-pass latency.
//!
//! Requires a pre-compiled `.holo` archive in `models/`. Run:
//!   ./scripts/download-models.sh tinyllama
//!   cargo run --release -- compile models/TinyLlama-1.1B-Chat-v1.0/model.onnx
//!
//! Then:
//!   cargo bench --bench inference
//!
//! Benchmarks are silently skipped if the model archive is not present.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::path::{Path, PathBuf};
use std::time::Duration;

use hologram_ai::runner::HoloRunner;

fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn onnx_holo_path() -> PathBuf {
    workspace_root().join("models/TinyLlama-1.1B-Chat-v1.0/model.holo")
}

/// A loaded model plus zeroed input buffers sized to the compiled graph's
/// input ports. The archive carries concrete shapes, so the input byte sizes
/// are fixed at compile time and read straight off the session.
struct BenchModel {
    runner: HoloRunner,
    inputs: Vec<Vec<u8>>,
}

fn load_model(path: &Path) -> Option<BenchModel> {
    if !path.exists() {
        eprintln!("  skipping: {} not found", path.display());
        return None;
    }

    let runner = HoloRunner::from_path(path, None).ok()?;
    let inputs: Vec<Vec<u8>> = runner
        .input_byte_sizes()
        .into_iter()
        .map(|n| vec![0u8; n])
        .collect();

    Some(BenchModel { runner, inputs })
}

fn bench_forward(c: &mut Criterion, name: &str, model: &mut BenchModel) {
    let mut group = c.benchmark_group("forward");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    group.bench_function(BenchmarkId::new("execute", name), |b| {
        let input_refs: Vec<&[u8]> = model.inputs.iter().map(|v| v.as_slice()).collect();
        b.iter(|| {
            model
                .runner
                .execute(&input_refs)
                .expect("forward pass failed");
        });
    });

    group.finish();
}

fn inference_benchmarks(c: &mut Criterion) {
    if let Some(mut model) = load_model(&onnx_holo_path()) {
        bench_forward(c, "tinyllama_onnx", &mut model);
    }
}

criterion_group!(benches, inference_benchmarks);
criterion_main!(benches);

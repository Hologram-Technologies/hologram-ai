//! Model-size scaling benchmarks (V&V class PV).
//!
//! hologram-ai is bound by the same performance contract as hologram, whose
//! thesis is *perf is content addressing, not micro-opt*: a κ-label memo hit
//! returns a cached result O(1) in graph size, and matmul throughput holds its
//! efficiency across scale. These benches mirror hologram's own matmul sweep
//! (64 / 128 / 256 / 512) through the full hologram-ai pipeline and validate
//! both axes:
//!
//! - `matmul_compile/{n}` — compile (model → `.holo`) cost vs size.
//! - `matmul_cold/{n}` — forward with *novel* inputs each iter (full recompute);
//!   the matmul throughput-vs-size curve.
//! - `matmul_reuse_hit/{n}` — forward on *fixed* κ-labels (content-addressed
//!   memo hit); should be ~flat in n (O(1) reuse).
//! - `imported_forward` — a real imported model (`mini_transformer.onnx`),
//!   real-world verification, not just synthetic ops.
//!
//! No size is special-cased and no dimension is clamped — the sweep exists to
//! prove no arbitrary limit throttles a larger model (see `tests/perf_floor.rs`
//! for the asserted floor). Run: `cargo bench -p hologram-ai --bench scaling`.

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::ort_runner::onnx_builder;

const SIZES: &[usize] = &[64, 128, 256, 512];

fn compile(model: Vec<u8>) -> HoloRunner {
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxBytes(model))
        .expect("compile failed");
    HoloRunner::from_bytes(archive.bytes).expect("load failed")
}

/// Zeroed input buffers sized to the model's ports, plus a mutable copy used to
/// perturb bytes (forcing a novel κ-label → real recompute) for the cold path.
fn zeroed_inputs(runner: &HoloRunner) -> Vec<Vec<u8>> {
    runner
        .input_byte_sizes()
        .iter()
        .map(|&n| vec![0u8; n])
        .collect()
}

fn bench_matmul_compile(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_compile");
    for &n in SIZES {
        let model = onnx_builder::matmul(n, n, n);
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| compile(model.clone()));
        });
    }
    g.finish();
}

fn bench_matmul_cold(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_cold");
    for &n in SIZES {
        // 2·n³ flops per n×n×n matmul.
        g.throughput(Throughput::Elements((2 * n * n * n) as u64));
        let mut runner = compile(onnx_builder::matmul(n, n, n));
        let base = zeroed_inputs(&runner);
        let mut seed = 0u8;
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter_batched(
                || {
                    // Perturb one byte of each input so its content address —
                    // and thus the result — is novel: forces a real recompute,
                    // never a memo hit. (Wraps; the value is irrelevant.)
                    seed = seed.wrapping_add(1);
                    let mut ins = base.clone();
                    for buf in ins.iter_mut() {
                        if let Some(first) = buf.first_mut() {
                            *first = seed;
                        }
                    }
                    ins
                },
                |ins| {
                    let refs: Vec<&[u8]> = ins.iter().map(|v| v.as_slice()).collect();
                    runner.execute(&refs).expect("execute failed");
                },
                BatchSize::SmallInput,
            );
        });
    }
    g.finish();
}

fn bench_matmul_reuse_hit(c: &mut Criterion) {
    let mut g = c.benchmark_group("matmul_reuse_hit");
    for &n in SIZES {
        let mut runner = compile(onnx_builder::matmul(n, n, n));
        let inputs = zeroed_inputs(&runner);
        // Intern once; the labels are fixed, so every call after the first is a
        // whole-graph memo hit (cached output labels, no compute, no copy).
        let labels: Vec<_> = inputs.iter().map(|v| runner.intern_input(v)).collect();
        runner.execute_addressed(&labels).expect("warm memo");
        g.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| runner.execute_addressed(&labels).expect("reuse hit"));
        });
    }
    g.finish();
}

fn bench_imported_forward(c: &mut Criterion) {
    // Real-world verification: a complete imported transformer, not a synthetic
    // single op. Skips cleanly if the fixture is absent.
    let Some(model) = hologram_ai_conformance::ort_runner::fixtures::load("mini_transformer")
    else {
        return;
    };
    let mut runner = {
        let archive = ModelCompiler {
            seq_len_override: Some(64),
            ..Default::default()
        }
        .compile(ModelSource::OnnxBytes(model))
        .expect("compile failed");
        HoloRunner::from_bytes(archive.bytes).expect("load failed")
    };
    let inputs = zeroed_inputs(&runner);
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let mut g = c.benchmark_group("imported_forward");
    g.bench_function("mini_transformer_seq64", |b| {
        b.iter(|| runner.execute(&refs).expect("execute failed"));
    });
    g.finish();
}

criterion_group!(
    benches,
    bench_matmul_compile,
    bench_matmul_cold,
    bench_matmul_reuse_hit,
    bench_imported_forward
);
criterion_main!(benches);

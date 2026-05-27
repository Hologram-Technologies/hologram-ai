//! End-to-end smoke test for the UOR-native pipeline (LW + exec).
//!
//! Compiles a tiny ONNX fixture all the way to a `.holo` archive on the
//! canonical `OpKind` model, loads it into an `InferenceSession`, and runs a
//! forward pass with zero inputs. This validates that lowering produces a graph
//! hologram's compiler accepts and that the session executes it — the first
//! real check that the re-architecture works, not just compiles.

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use std::path::Path;

fn fixture(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../hologram-ai-conformance/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading fixture {path:?}: {e}"))
}

/// Compile + load + execute a fixture; assert it runs and produces output.
fn compile_and_run(name: &str) {
    let bytes = fixture(name);
    let compiler = ModelCompiler {
        seq_len_override: Some(4),
        ..Default::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(bytes))
        .unwrap_or_else(|e| panic!("[{name}] compile failed: {e:#}"));
    assert!(!archive.bytes.is_empty(), "[{name}] empty archive");

    let mut runner = HoloRunner::from_bytes(archive.bytes)
        .unwrap_or_else(|e| panic!("[{name}] load failed: {e:#}"));

    // Zero-fill every input at its declared byte size.
    let sizes = runner.input_byte_sizes();
    let owned: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

    let outputs = runner
        .execute(&refs)
        .unwrap_or_else(|e| panic!("[{name}] execute failed: {e:#}"));
    assert!(!outputs.is_empty(), "[{name}] no outputs");
    assert!(
        outputs.iter().any(|o| !o.bytes.is_empty()),
        "[{name}] all outputs empty"
    );
    println!(
        "[{name}] OK — {} input(s), {} output(s), {} output bytes",
        sizes.len(),
        outputs.len(),
        outputs.iter().map(|o| o.bytes.len()).sum::<usize>()
    );
}

#[test]
fn layer_norm_compiles_and_runs() {
    compile_and_run("layer_norm.onnx");
}

#[test]
fn rms_norm_compiles_and_runs() {
    compile_and_run("rms_norm.onnx");
}

#[test]
fn concat_compiles_and_runs() {
    compile_and_run("concat_4d_last_axis.onnx");
}

#[test]
fn softmax_compiles_and_runs() {
    compile_and_run("softmax_dyn_seq.onnx");
}

#[test]
fn expand_compiles_and_runs() {
    compile_and_run("expand_dynamic_shape.onnx");
}

/// `Shape(X)→Gather(idx=2)→Cast→Y` for `X=[1, seq, 64]`. The gathered dim is
/// compile-time data, so the whole cone const-folds at import to the scalar
/// `64.0` (no runtime selection op is emitted). With `seq=4` the output is the
/// hidden dim 64; we assert the folded value round-trips through hologram's
/// session — which requires hologram to resolve an output port that aliases a
/// pure constant (a fully const-folded output).
#[test]
fn gather_shape_const_folds_and_runs() {
    let bytes = fixture("gather_shape_i64.onnx");
    let compiler = ModelCompiler {
        seq_len_override: Some(4),
        ..Default::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("gather_shape compile failed");

    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let sizes = runner.input_byte_sizes();
    let owned: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

    let outputs = runner.execute(&refs).expect("execute failed");
    assert_eq!(outputs.len(), 1, "expected one output");
    let out = &outputs[0].bytes;
    assert_eq!(out.len(), 4, "expected one f32 scalar");
    let val = f32::from_le_bytes([out[0], out[1], out[2], out[3]]);
    assert_eq!(val, 64.0, "Shape(X)[2] for X=[1,4,64] must fold to 64");
}

#[test]
fn swiglu_compiles_and_runs() {
    compile_and_run("swiglu_fused_reference.onnx");
}

/// Content-addressed execution: interning inputs to κ-labels and running
/// `execute_addressed` yields the same output bytes as the byte-level
/// `execute`, and the addressed call returns stable output labels (so a
/// pipeline can compose on addresses without byte round-trips). Exercises the
/// principle that hologram operates over uor addresses, not raw values.
#[test]
fn addressed_execution_matches_byte_execution() {
    let bytes = fixture("layer_norm.onnx");
    let compiler = ModelCompiler {
        seq_len_override: Some(4),
        ..Default::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("compile failed");
    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");

    let sizes = runner.input_byte_sizes();
    let owned: Vec<Vec<u8>> = sizes.iter().map(|&n| vec![0u8; n]).collect();

    // Byte-level reference.
    let refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
    let byte_out = runner.execute(&refs).expect("byte execute failed");

    // Addressed: intern each input to a κ-label, then execute on labels.
    let labels: Vec<_> = owned.iter().map(|v| runner.intern_input(v)).collect();
    let out_labels = runner
        .execute_addressed(&labels)
        .expect("addressed execute failed");
    assert_eq!(out_labels.len(), byte_out.len(), "output arity mismatch");

    for (label, bytes_out) in out_labels.iter().zip(byte_out.iter()) {
        let resolved = runner
            .resolve(label)
            .expect("output label resolves to bytes");
        assert_eq!(
            resolved,
            bytes_out.bytes.as_slice(),
            "addressed bytes != byte-level bytes"
        );
    }

    // Re-running on the same input labels returns the same output labels
    // (deterministic content addresses → whole-graph memo / elision).
    let again = runner
        .execute_addressed(&labels)
        .expect("addressed re-run failed");
    assert_eq!(
        again, out_labels,
        "output labels must be stable across runs"
    );
}

#[test]
fn gqa_attention_compiles_and_runs() {
    compile_and_run("gqa_fused_reference.onnx");
}

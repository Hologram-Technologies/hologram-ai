//! **Authoritative ONNX operator-spec conformance (classes IM / LW / EE).**
//!
//! Validates hologram-ai against the **official ONNX backend node-test corpus**
//! — the executable form of the ONNX operator specification, published in the
//! `onnx/onnx` repository under `onnx/backend/test/data/node/`. Each pinned
//! case ships a `model.onnx` plus `test_data_set_0/{input_i.pb, output_0.pb}`
//! (serialized `TensorProto`s). For each case the test:
//!
//! 1. downloads the authoritative artifacts (model + inputs + expected output),
//!    caching them under `target/onnx-node-cache/` (git-ignored);
//! 2. imports + compiles the model through hologram-ai onto the canonical
//!    `OpKind` model and runs it with the spec's inputs; and
//! 3. asserts the output equals the spec's `output_0.pb` within tolerance.
//!
//! This is V&V against an **external authority we did not author** (the ONNX
//! project), per `VERIFICATION.md` — not a self-generated golden.
//!
//! Gated behind `HOLOGRAM_AI_LIVE=1` (the downloads need network). Build with
//! `--features onnx-spec`. Mirrors uor-addr's `external_models.rs` discipline
//! (pin by URL, cache, validate against the authoritative artifact).
#![cfg(feature = "onnx-spec")]

use std::path::PathBuf;
use std::process::Command;

use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_conformance::tolerance::tolerance_for;

/// A pinned ONNX backend node-test case.
struct NodeTest {
    /// `onnx/backend/test/data/node/test_<dir>` directory name (sans `test_`).
    dir: &'static str,
    /// Number of model inputs (input_0.pb .. input_{n-1}.pb).
    n_inputs: usize,
    /// `OpKind` name used to pick the comparison tolerance.
    op: &'static str,
}

/// The pinned authoritative corpus — single-op spec cases hologram-ai lowers
/// canonically. Pinned to the `onnx/onnx` `main` node-test layout.
const CASES: &[NodeTest] = &[
    NodeTest {
        dir: "relu",
        n_inputs: 1,
        op: "Relu",
    },
    NodeTest {
        dir: "add",
        n_inputs: 2,
        op: "Add",
    },
    NodeTest {
        dir: "matmul_2d",
        n_inputs: 2,
        op: "MatMul",
    },
    NodeTest {
        dir: "softmax_example",
        n_inputs: 1,
        op: "Softmax",
    },
    NodeTest {
        dir: "mul",
        n_inputs: 2,
        op: "Mul",
    },
    NodeTest {
        dir: "sub",
        n_inputs: 2,
        op: "Sub",
    },
    // Quantized weights / activations against the spec's own vectors (class QZ):
    // x, scale, zero-point are graph inputs in these cases, so they exercise the
    // canonical Dequantize path end to end.
    NodeTest {
        dir: "dequantizelinear",
        n_inputs: 3,
        op: "Dequantize",
    },
    NodeTest {
        dir: "dequantizelinear_axis",
        n_inputs: 3,
        op: "Dequantize",
    },
    // First-class Gather (embedding lookup): data + integer indices are graph
    // inputs, exercising the runtime-indexed Gather kernel vs ONNX's own output.
    NodeTest {
        dir: "gather_0",
        n_inputs: 2,
        op: "Gather",
    },
    NodeTest {
        dir: "gather_1",
        n_inputs: 2,
        op: "Gather",
    },
];

// Pinned to the immutable v1.17.0 release tag (registered in model/oracles.toml
// as `onnx-node-corpus`; the pin is checked live by `xtask pin-check`).
const BASE: &str = "https://raw.githubusercontent.com/onnx/onnx/v1.17.0/onnx/backend/test/data/node";

fn cache_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("target/onnx-node-cache");
    p
}

/// Fetch `url` into `dest` (cached). Returns false if the resource is absent.
fn fetch(url: &str, dest: &std::path::Path) -> bool {
    if dest.exists() {
        return true;
    }
    std::fs::create_dir_all(dest.parent().unwrap()).unwrap();
    let status = Command::new("curl")
        .args(["-fsSL", "-o", dest.to_str().unwrap(), url])
        .status()
        .expect("curl must be available for the live ONNX-spec corpus");
    status.success()
}

fn run_case(c: &NodeTest) {
    let root = cache_dir().join(c.dir);
    let model_path = root.join("model.onnx");
    assert!(
        fetch(&format!("{BASE}/test_{}/model.onnx", c.dir), &model_path),
        "[{}] could not fetch authoritative model.onnx",
        c.dir
    );

    // Inputs (spec order matches the model's graph-input order).
    let mut inputs: Vec<Vec<u8>> = Vec::new();
    for i in 0..c.n_inputs {
        let p = root.join(format!("input_{i}.pb"));
        assert!(
            fetch(
                &format!("{BASE}/test_{}/test_data_set_0/input_{i}.pb", c.dir),
                &p
            ),
            "[{}] could not fetch authoritative input_{i}.pb",
            c.dir
        );
        let (bytes, _dims, _dt) =
            hologram_ai_onnx::decode_tensor_proto_bytes(&std::fs::read(&p).unwrap())
                .unwrap_or_else(|e| panic!("[{}] decode input_{i}: {e}", c.dir));
        inputs.push(bytes);
    }

    // Authoritative expected output.
    let out_path = root.join("output_0.pb");
    assert!(
        fetch(
            &format!("{BASE}/test_{}/test_data_set_0/output_0.pb", c.dir),
            &out_path
        ),
        "[{}] could not fetch authoritative output_0.pb",
        c.dir
    );
    let expected = hologram_ai_onnx::decode_tensor_proto_f32(&std::fs::read(&out_path).unwrap())
        .unwrap_or_else(|e| panic!("[{}] decode output_0: {e}", c.dir));

    // Compile + run through hologram-ai (canonical OpKind pipeline).
    let archive = ModelCompiler::default()
        .compile(ModelSource::OnnxPath(model_path))
        .unwrap_or_else(|e| panic!("[{}] compile failed: {e:#}", c.dir));
    let mut runner = HoloRunner::from_bytes(archive.bytes)
        .unwrap_or_else(|e| panic!("[{}] load failed: {e:#}", c.dir));
    let refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();
    let outputs = runner
        .execute(&refs)
        .unwrap_or_else(|e| panic!("[{}] execute failed: {e:#}", c.dir));

    let actual: &[f32] = bytemuck::cast_slice(&outputs[0].bytes);
    assert_eq!(
        actual.len(),
        expected.len(),
        "[{}] output length: hologram-ai={} spec={}",
        c.dir,
        actual.len(),
        expected.len()
    );
    let tol = tolerance_for(c.op);
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            tol.is_close(a, e),
            "[{}] index {i}: hologram-ai={a} spec={e} (atol={}, rtol={})",
            c.dir,
            tol.atol,
            tol.rtol
        );
    }
    println!(
        "[{}] OK — matches ONNX spec output ({} elems)",
        c.dir,
        expected.len()
    );
}

#[test]
fn onnx_operator_spec_conformance() {
    if std::env::var("HOLOGRAM_AI_LIVE").as_deref() != Ok("1") {
        eprintln!("skipping: set HOLOGRAM_AI_LIVE=1 to run the live ONNX-spec corpus");
        return;
    }
    for c in CASES {
        run_case(c);
    }
}

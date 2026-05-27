//! The `run` CLI executes arbitrary compiled models — multi-input, multi-output,
//! mixed dtypes — through `--fill` (synthesize unspecified inputs) and the
//! explicit `--input` path, with typed output reporting. Drives the real
//! `hologram-ai` binary end-to-end (compile → run), no model downloads.

use std::collections::HashMap;
use std::process::Command;

use hologram_ai::{ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, DType, TensorInfo};

fn compile_to_temp(graph: AiGraph, stem: &str) -> std::path::PathBuf {
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile");
    let path = std::env::temp_dir().join(format!("hai_run_arb_{stem}.holo"));
    std::fs::write(&path, &archive.bytes).expect("write archive");
    path
}

fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dt, shape_from_concrete(dims))
}

fn run(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_hologram-ai"))
        .args(args)
        .output()
        .expect("spawn hologram-ai");
    let mut s = String::from_utf8_lossy(&out.stdout).into_owned();
    s.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.success(), s)
}

/// Two inputs `[1,4]·[4,4]` → MatMul `[1,4]`. With `--fill ones` both inputs are
/// 1.0, so each output element is 4.0.
fn matmul_2in() -> AiGraph {
    let (x, w, y) = (0u32, 1u32, 2u32);
    let mut tinfo = HashMap::new();
    tinfo.insert(x, ti(DType::F32, &[1, 4]));
    tinfo.insert(w, ti(DType::F32, &[4, 4]));
    tinfo.insert(y, ti(DType::F32, &[1, 4]));
    AiGraph {
        name: "mm2".into(),
        nodes: vec![AiNode::new(0, AiOp::MatMul, vec![x, w], vec![y])],
        inputs: vec![x, w],
        outputs: vec![y],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params: HashMap::new(),
        tensor_info: tinfo,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

/// One input `[1,4]`, two outputs: `Relu(x)` and `x + x`.
fn one_in_two_out() -> AiGraph {
    let (x, r, a) = (0u32, 1u32, 2u32);
    let mut tinfo = HashMap::new();
    tinfo.insert(x, ti(DType::F32, &[1, 4]));
    tinfo.insert(r, ti(DType::F32, &[1, 4]));
    tinfo.insert(a, ti(DType::F32, &[1, 4]));
    AiGraph {
        name: "split".into(),
        nodes: vec![
            AiNode::new(0, AiOp::Relu, vec![x], vec![r]),
            AiNode::new(1, AiOp::Add, vec![x, x], vec![a]),
        ],
        inputs: vec![x],
        outputs: vec![r, a],
        input_names: Vec::new(),
        output_names: Vec::new(),
        params: HashMap::new(),
        tensor_info: tinfo,
        metadata: HashMap::new(),
        warnings: Vec::new(),
        dim_vars: Default::default(),
        shape_constraints: Default::default(),
        subgraphs: HashMap::new(),
        tensor_names: HashMap::new(),
        topo_cache: Default::default(),
    }
}

#[test]
fn fill_ones_runs_multi_input_model_and_reports_typed_output() {
    let path = compile_to_temp(matmul_2in(), "mm2");
    let (ok, out) = run(&[
        "run",
        path.to_str().unwrap(),
        "--fill",
        "ones",
        "--verbose",
    ]);
    assert!(ok, "run failed:\n{out}");
    // Port description lines.
    assert!(out.contains("2 input(s), 1 output(s)"), "missing port summary:\n{out}");
    assert!(out.contains("input[0]: f32 × 4"), "missing input desc:\n{out}");
    // MatMul(ones[1,4], ones[4,4]) = [4,4,4,4]; typed f32 preview.
    assert!(out.contains("output[0]: f32 × 4"), "missing output desc:\n{out}");
    assert!(out.contains("[4, 4, 4, 4]"), "wrong output preview:\n{out}");
}

#[test]
fn fill_zeros_runs_any_model_multi_output() {
    let path = compile_to_temp(one_in_two_out(), "split");
    let (ok, out) = run(&["run", path.to_str().unwrap(), "--fill", "zeros", "--verbose"]);
    assert!(ok, "run failed:\n{out}");
    // Two outputs, both zero (relu(0)=0, 0+0=0).
    assert!(out.contains("output[0]: f32 × 4"), "missing output[0]:\n{out}");
    assert!(out.contains("output[1]: f32 × 4"), "missing output[1]:\n{out}");
    assert!(out.contains("[0, 0, 0, 0]"), "expected zero outputs:\n{out}");
}

#[test]
fn explicit_input_overrides_and_numeric_fill_compose() {
    // x = [1,2,3,4] (explicit), W = fill 1.0 → MatMul = [10,10,10,10].
    let path = compile_to_temp(matmul_2in(), "mm2b");
    let x: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let hex: String = x.iter().map(|b| format!("{b:02x}")).collect();
    let (ok, out) = run(&[
        "run",
        path.to_str().unwrap(),
        "--input",
        &format!("0:{hex}"),
        "--fill",
        "1",
        "--verbose",
    ]);
    assert!(ok, "run failed:\n{out}");
    assert!(out.contains("[10, 10, 10, 10]"), "wrong composed output:\n{out}");
}

#[test]
fn missing_input_without_fill_is_a_clear_error() {
    let path = compile_to_temp(matmul_2in(), "mm2c");
    let (ok, out) = run(&["run", path.to_str().unwrap()]);
    assert!(!ok, "expected failure when inputs are missing");
    assert!(out.contains("--fill zeros"), "error should suggest --fill:\n{out}");
}

#[test]
fn wrong_size_explicit_input_is_rejected() {
    let path = compile_to_temp(matmul_2in(), "mm2d");
    // input[0] expects 16 bytes (4×f32); give 8.
    let (ok, out) = run(&[
        "run",
        path.to_str().unwrap(),
        "--input",
        "0:0011223344556677",
        "--fill",
        "zeros",
    ]);
    assert!(!ok, "expected size-mismatch failure");
    assert!(out.contains("expects 16"), "error should state expected size:\n{out}");
}

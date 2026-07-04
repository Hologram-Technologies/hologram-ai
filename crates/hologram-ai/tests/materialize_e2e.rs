//! End-to-end κ-materialization witness (dictionary row `kappa-materialization`).
//!
//! A k-form (weightless) compile plus κ-store materialization must execute to
//! results byte-identical with the same graph compiled from inline weights;
//! a missing κ aborts naming the label; corrupt store content fails the
//! content-integrity check.

use hologram_ai::materialize::{
    kappa_of, kappa_requirements, materialize_archive, DirKappaStore,
};
use hologram_ai::runner::HoloRunner;
use hologram_ai::{ModelCompiler, ModelSource};
use hologram_ai_common::{
    shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo,
};
use std::collections::HashMap;

fn ti(dt: DType, dims: &[u64]) -> TensorInfo {
    TensorInfo::new(dt, shape_from_concrete(dims))
}

/// `y[1,4] = x[1,4] · w[4,4]` with `w` supplied as `param`.
fn matmul_graph(param: AiParam) -> AiGraph {
    let (x, w, y) = (0u32, 1u32, 2u32);
    let mut tinfo = HashMap::new();
    tinfo.insert(x, ti(DType::F32, &[1, 4]));
    tinfo.insert(w, ti(DType::F32, &[4, 4]));
    tinfo.insert(y, ti(DType::F32, &[1, 4]));
    let mut params = HashMap::new();
    params.insert(w, param);
    AiGraph {
        name: "materialize-e2e".into(),
        nodes: vec![AiNode::new(0, AiOp::MatMul, vec![x, w], vec![y])],
        inputs: vec![x],
        outputs: vec![y],
        input_names: vec!["x".into()],
        output_names: vec!["y".into()],
        params,
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

/// Deterministic non-trivial 4×4 F32 weight bytes.
fn weight_bytes() -> Vec<u8> {
    let vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.25 - 1.5).collect();
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn input_bytes() -> Vec<u8> {
    [1.0f32, -2.0, 3.0, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

fn compile(graph: AiGraph) -> Vec<u8> {
    ModelCompiler::default()
        .compile(ModelSource::AiGraph(graph))
        .expect("compile")
        .bytes
}

fn store_dir(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("hai-materialize-e2e-{tag}-{}", std::process::id()))
}

#[test]
fn materialized_kform_matches_inline_compile_exactly() {
    let w = weight_bytes();
    let kappa = kappa_of(&w);

    let inline_holo = compile(matmul_graph(AiParam::inline(
        w.clone(),
        ti(DType::F32, &[4, 4]),
    )));
    let kform_holo = compile(matmul_graph(AiParam::External {
        kappa: kappa.clone(),
        info: ti(DType::F32, &[4, 4]),
    }));

    // The k-form archive declares exactly its one requirement; the inline
    // archive declares none.
    let reqs = kappa_requirements(&kform_holo).expect("κ-map parses");
    assert_eq!(reqs.len(), 1, "one external weight → one requirement");
    assert_eq!(reqs[0].kappa, kappa);
    assert!(
        kappa_requirements(&inline_holo)
            .expect("no κ-map is fine")
            .is_empty(),
        "inline archives carry no κ-map"
    );

    // Materialize against a store holding the weight under its κ.
    let dir = store_dir("ok");
    let store = DirKappaStore::new(&dir);
    assert_eq!(store.insert(&w).expect("insert"), kappa);
    let mut store = store;
    let materialized =
        materialize_archive(&kform_holo, &mut store).expect("materialization succeeds");

    // Both archives must load and execute to identical bytes.
    let x = input_bytes();
    let mut inline_runner = HoloRunner::from_bytes(inline_holo).expect("inline loads");
    let inline_out = inline_runner.execute(&[&x]).expect("inline executes");
    let mut mat_runner = HoloRunner::from_bytes(materialized).expect("materialized loads");
    let mat_out = mat_runner.execute(&[&x]).expect("materialized executes");
    assert_eq!(inline_out.len(), mat_out.len());
    for (a, b) in inline_out.iter().zip(mat_out.iter()) {
        assert_eq!(a.bytes, b.bytes, "materialized execution must be byte-identical");
    }

    // And the result is genuinely the matmul, not zeros from empty weights.
    let y: Vec<f32> = mat_out[0]
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte chunks")))
        .collect();
    assert!(
        y.iter().any(|v| v.abs() > 1e-6),
        "output must reflect real weights, got {y:?}"
    );

    std::fs::remove_dir_all(&dir).expect("cleanup");
}

#[test]
fn missing_kappa_aborts_naming_the_label() {
    let w = weight_bytes();
    let kappa = kappa_of(&w);
    let kform_holo = compile(matmul_graph(AiParam::External {
        kappa: kappa.clone(),
        info: ti(DType::F32, &[4, 4]),
    }));

    let dir = store_dir("missing");
    let mut store = DirKappaStore::new(&dir);
    let err = materialize_archive(&kform_holo, &mut store)
        .expect_err("an empty store cannot materialize");
    let msg = format!("{err:#}");
    assert!(msg.contains(&kappa), "error must name the missing κ: {msg}");
}

#[test]
fn corrupt_store_content_fails_integrity() {
    let w = weight_bytes();
    let kappa = kappa_of(&w);
    let kform_holo = compile(matmul_graph(AiParam::External {
        kappa: kappa.clone(),
        info: ti(DType::F32, &[4, 4]),
    }));

    // Plant WRONG bytes under the expected κ filename.
    let dir = store_dir("corrupt");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let mut wrong = w;
    wrong[0] ^= 0xFF;
    std::fs::write(dir.join(format!("{kappa}.bin")), &wrong).expect("plant corrupt content");

    let mut store = DirKappaStore::new(&dir);
    let err = materialize_archive(&kform_holo, &mut store)
        .expect_err("corrupt content must not materialize");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("integrity"),
        "error must be the κ integrity failure: {msg}"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

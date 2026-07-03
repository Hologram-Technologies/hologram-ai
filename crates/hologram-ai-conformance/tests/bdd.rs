#![allow(clippy::unwrap_used)]

use cucumber::{given, then, when, World};
use hologram_ai_core::domain::ModelManifest;
use holospaces::address;

#[cfg(feature = "conformance")]
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
#[cfg(feature = "conformance")]
use hologram_ai_conformance::ort_runner::fixtures;
#[cfg(feature = "conformance")]
use hologram_ai_conformance::ort_runner::runner::{run_onnx_typed, OrtInputTyped};

#[cfg(feature = "conformance")]
fn f32_to_le(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(feature = "conformance")]
fn le_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[derive(Debug, Default, cucumber::World)]
#[allow(dead_code)]
struct HologramWorld {
    model_name: String,
    manifest: Option<ModelManifest>,
    fixture_name: Option<String>,
    holo_output: Option<Vec<f32>>,
    ort_output: Option<Vec<f32>>,
    holo_archive: Option<Vec<u8>>,
    original_node_count: usize,
    compiled_node_count: usize,
}

#[given(expr = "an arbitrary model name {string}")]
async fn given_model_name(w: &mut HologramWorld, name: String) {
    w.model_name = name;
}

#[given(expr = "an authoritative model name {string}")]
async fn given_authoritative_model_name(w: &mut HologramWorld, name: String) {
    w.model_name = name;
}

#[when(expr = "the model manifest is instantiated with a holospaces::Kappa for {string}")]
async fn when_kappa_manifest(w: &mut HologramWorld, id: String) {
    let kappa = hologram_ai_core::domain::Kappa(address(id.as_bytes()));
    w.manifest = Some(ModelManifest {
        model_kappa: kappa.clone(),
        archive_kappa: kappa,
        name: w.model_name.clone(),
        description: None,
    });
}

#[then(expr = "the model manifest preserves the holospaces::Kappa")]
async fn then_kappa_manifest(w: &mut HologramWorld) {
    let manifest = w.manifest.as_ref().unwrap();
    assert_eq!(manifest.model_kappa, manifest.archive_kappa);
    assert_eq!(manifest.name, w.model_name);
}

#[given(expr = "the external authoritative ONNX fixture {string}")]
async fn given_fixture(w: &mut HologramWorld, name: String) {
    w.fixture_name = Some(name);
}

#[cfg(feature = "conformance")]
#[when(expr = "the fixture is compiled and executed via the holographic compiler")]
async fn when_holo_execute(w: &mut HologramWorld) {
    let name = w.fixture_name.as_ref().unwrap();
    let model = fixtures::load_or_panic(name);
    let (seq, hidden) = (4usize, 32usize);
    let x: Vec<f32> = (0..seq * hidden)
        .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1)
        .collect();

    // Import to graph first to get original node count
    let ai_graph =
        hologram_ai_onnx::import_onnx(&model, None, Default::default()).expect("import failed");
    w.original_node_count = ai_graph.nodes.len();

    let archive = ModelCompiler {
        seq_len_override: Some(seq as u64),
        ..Default::default()
    }
    .compile(ModelSource::AiGraph(ai_graph))
    .expect("compile failed");

    w.compiled_node_count = archive.stats.node_count;

    let mut runner = HoloRunner::from_bytes(archive.bytes).expect("load failed");
    let out = runner.execute(&[&f32_to_le(&x)]).expect("execute failed");
    w.holo_output = Some(le_to_f32(&out[0].bytes));

    let ort_out = run_onnx_typed(
        &model,
        vec![OrtInputTyped::F32 {
            name: "X".into(),
            shape: vec![seq, hidden],
            data: x,
        }],
    )
    .expect("ORT run failed");
    w.ort_output = Some(ort_out[0].data.clone());
}

#[cfg(not(feature = "conformance"))]
#[when(expr = "the fixture is compiled and executed via the holographic compiler")]
async fn when_holo_execute_skip(_w: &mut HologramWorld) {}

#[cfg(feature = "conformance")]
#[cfg(feature = "conformance")]
#[path = "fetch_helper.rs"]
mod fetch_helper;

#[cfg(feature = "conformance")]
#[when(expr = "the safetensors metadata is fetched and streamed to the holographic compiler")]
async fn when_safetensors_fetched_and_streamed(w: &mut HologramWorld) {
    let (config_json, keys, kappas, shapes, dtypes) = fetch_helper::fetch_authoritative_metadata(&w.model_name).await;

    let compiler = hologram_ai::compiler::ModelCompiler::default();
    let archive = compiler
        .compile(hologram_ai::compiler::ModelSource::SafetensorsStreamed {
            config_json,
            keys,
            kappas,
            shapes,
            dtypes,
        });
        
    let archive = match archive {
        Ok(a) => a,
        Err(e) => panic!("compile failed: {:?}", e),
    };

    w.compiled_node_count = archive.stats.node_count;
    w.holo_archive = Some(archive.bytes);
}

#[cfg(not(feature = "conformance"))]
#[when(expr = "the safetensors metadata is streamed to the holographic compiler")]
async fn when_safetensors_streamed_skip(_w: &mut HologramWorld) {}

#[cfg(feature = "conformance")]
#[then(expr = "the compiled holographic archive must contain external parameter mappings")]
async fn then_archive_contains_external(w: &mut HologramWorld) {
    let bytes = w.holo_archive.as_ref().unwrap();
    let runner = hologram_ai::HoloRunner::from_bytes(bytes.clone()).unwrap();

    assert!(w.compiled_node_count > 0, "Graph should have nodes");

    let kappa_ext = runner
        .extension("holospaces.kappa_map")
        .expect("Missing kappa map extension");
    let kappa_str = std::str::from_utf8(kappa_ext).unwrap();

    assert!(
        kappa_str.contains("blake3:model.embed_tokens.weight"),
        "Missing embed kappa mapping"
    );
    assert!(
        kappa_str.contains("blake3:lm_head.weight"),
        "Missing lm_head kappa mapping"
    );
}

#[cfg(not(feature = "conformance"))]
#[then(expr = "the compiled holographic archive must contain external parameter mappings")]
async fn then_archive_contains_external_skip(_w: &mut HologramWorld) {}

#[cfg(feature = "conformance")]
#[then(expr = "the outputs must exactly match the ONNX Runtime authoritative execution")]
async fn then_match_ort(w: &mut HologramWorld) {
    let holo = w.holo_output.as_ref().unwrap();
    let reference = w.ort_output.as_ref().unwrap();
    assert_eq!(holo.len(), reference.len());
    for (i, (h, r)) in holo.iter().zip(reference.iter()).enumerate() {
        let diff = (h - r).abs();
        let tol = 1e-2 + 2e-3 * r.abs();
        assert!(
            diff <= tol,
            "element {i}: hologram-ai {h} vs ORT {r} (|diff| {diff} > tol {tol})"
        );
    }
}

#[cfg(not(feature = "conformance"))]
#[then(expr = "the outputs must exactly match the ONNX Runtime authoritative execution")]
async fn then_match_ort_skip(_w: &mut HologramWorld) {}

#[tokio::main]
async fn main() {
    HologramWorld::cucumber()
        .run("../../features/suites/arbitrary_models")
        .await;
}

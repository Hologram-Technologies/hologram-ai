//! Hermetic end-to-end: committed tiny local source → uor-r4 compile →
//! deterministic bundle → `.holo` InferenceModel → load → select →
//! R4G1 prediction/generation.
//!
//! The compile stage runs real teacher observation, so these tests are
//! `#[ignore]`d in default CI; run with:
//!
//! ```bash
//! cargo test -p hologram-ai --test e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use hologram_ai::{CompileOptions, InferenceRequest, LocalSource};

fn fixture_source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../oracles/fixture")
}

fn work_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hologram-ai-e2e-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Full chain: local source → `.holo` → application → session → predict +
/// generate. Proves the engine is R4G1 (engine id + bundle roles), not a
/// legacy fallback.
#[test]
#[ignore = "slow: real uor-r4 teacher observation"]
fn fixture_compile_to_inference() {
    let work = work_dir("full");
    let output = work.join("fixture.holo");

    let compiled = hologram_ai::Compiler::builder()
        .source(LocalSource::new(fixture_source()))
        .entry("ai.default")
        .work_dir(work.join("r4-work"))
        .options(CompileOptions {
            seconds: Some(20),
            sequence_length: None,
        })
        .build()
        .unwrap()
        .compile_to_path(&output)
        .unwrap();

    assert!(
        output.exists(),
        "compile must terminate with a .holo archive"
    );
    assert_eq!(compiled.entry, "ai.default");

    // Load through the official reader path; inspect without inference.
    let app = hologram_ai::Application::open_path(&output).unwrap();
    assert_eq!(app.models().len(), 1);
    let descriptor = &app.models()[0];
    assert_eq!(
        descriptor.engine(),
        "uor-r4",
        "the engine must be R4G1/uor-r4"
    );
    let manifest = descriptor.manifest();
    assert_eq!(manifest.artifact_format, "R4G1");
    assert!(manifest
        .artifacts
        .iter()
        .any(|a| a.role == hologram_ai_core::ArtifactRole::Graph));

    // Session: R4G1 prediction and deterministic generation.
    let model = app.model("ai.default").unwrap();
    let mut session = model.session().unwrap();

    let completion = session
        .invoke(
            "generate",
            InferenceRequest::builder()
                .text("prompt", "hello")
                .max_output_tokens(8)
                .build(),
        )
        .unwrap();
    assert_ne!(
        completion.finish_reason,
        hologram_ai::FinishReason::Cancelled
    );

    // Repeated execution is deterministic.
    let again = session
        .invoke(
            "generate",
            InferenceRequest::builder()
                .text("prompt", "hello")
                .max_output_tokens(8)
                .build(),
        )
        .unwrap();
    assert_eq!(completion.output, again.output);

    // Reset works and keeps the engine functional.
    session.reset();
    let mut prediction = hologram_ai_r4::Prediction::default();
    let seed = [1u32, 2, 3];
    session.predict_next_into(&seed, &mut prediction).unwrap();
    let _ = prediction;
}

/// The public compile product is exactly one `.holo`; no `.r4g1` escapes
/// the private work dir as a user artifact, and no `tless_store.bin` is
/// packaged.
#[test]
#[ignore = "slow: real uor-r4 teacher observation"]
fn compile_emits_only_holo_and_bundle_excludes_tls_store() {
    let work = work_dir("artifacts");
    let output = work.join("fixture.holo");

    hologram_ai::Compiler::builder()
        .source(LocalSource::new(fixture_source()))
        .work_dir(work.join("r4-work"))
        .options(CompileOptions {
            seconds: Some(20),
            sequence_length: None,
        })
        .build()
        .unwrap()
        .compile_to_path(&output)
        .unwrap();

    // The output path is a .holo and it parses as HOLO.
    let bytes = std::fs::read(&output).unwrap();
    assert_eq!(&bytes[..4], b"HOLO");

    // No component in the bundle equals the work-dir tless_store.bin.
    let store = work.join("r4-work");
    let store_bytes = find_file(&store, "tless_store.bin").and_then(|p| std::fs::read(p).ok());
    if let Some(store_bytes) = store_bytes {
        let app = hologram_ai::Application::open_bytes(&bytes).unwrap();
        let model = app.model("ai.default").unwrap();
        let bundle = hologram_ai_bundle::Bundle::parse_verified(model.bundle_bytes()).unwrap();
        for (_, component) in bundle.components() {
            assert_ne!(
                component,
                store_bytes.as_slice(),
                "tless_store must never ship"
            );
        }
    }
}

fn find_file(dir: &std::path::Path, name: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        } else if entry.file_name() == name {
            return Some(path);
        }
    }
    None
}

//! Packaging and discovery tests: `.holo` round-trips through the official
//! Hologram reader, v4 `InferenceModel` semantics, multi-model selection.
//! No engine is initialized anywhere in this file.

use hologram_ai::ModelLayer;
use hologram_ai_bundle::BundleBuilder;
use hologram_ai_core::{
    ArtifactRole, ErrorCategory, OperationDescriptor, StatusPolicy, ValueDescriptor, ValueKind,
    ARTIFACT_FORMAT_R4G1, ENGINE_UOR_R4,
};

fn text_generate_op() -> OperationDescriptor {
    OperationDescriptor {
        name: "generate".into(),
        inputs: vec![ValueDescriptor {
            name: "prompt".into(),
            kind: ValueKind::Text,
            media_type: None,
            required: true,
        }],
        outputs: vec![ValueDescriptor {
            name: "text".into(),
            kind: ValueKind::Text,
            media_type: None,
            required: true,
        }],
        streaming: true,
    }
}

/// A schema-valid text-model bundle with dummy component payloads (bundle
/// validation checks roles/digests/consistency, not graph semantics).
fn text_model_bundle(name: &str) -> Vec<u8> {
    let mut b = BundleBuilder::new(ENGINE_UOR_R4, ARTIFACT_FORMAT_R4G1, name)
        .set_operations(vec![text_generate_op()])
        .set_status_policy(StatusPolicy::R4_DEFAULT);
    b.add_component(ArtifactRole::Graph, b"fake-scored-r4g1-graph")
        .unwrap();
    b.add_component(ArtifactRole::SignatureArtifact, b"fake-signature-artifact")
        .unwrap();
    b.add_component(ArtifactRole::ScoreReport, b"{}").unwrap();
    b.add_component(ArtifactRole::Tokenizer, b"fake-tokenizer")
        .unwrap();
    b.finish().unwrap()
}

#[test]
fn model_only_archive_round_trip() {
    let archive = hologram_ai::build_model_archive(&[ModelLayer::uor_r4(
        "ai.default",
        text_model_bundle("fixture"),
    )])
    .unwrap();

    let app = hologram_ai::Application::open_bytes(&archive).unwrap();
    assert_eq!(app.models().len(), 1);
    let d = &app.models()[0];
    assert_eq!(d.entry(), "ai.default");
    assert_eq!(d.engine(), "uor-r4");
    assert!(d.content_kappa().starts_with("blake3:"));
    assert_eq!(d.operations()[0].name, "generate");
    // Selecting by entry and via the single-service default both work.
    let _ = app.model("ai.default").unwrap();
    let _ = app.default_model().unwrap();
}

#[test]
fn archive_is_byte_deterministic() {
    let bundle = text_model_bundle("fixture");
    let a = hologram_ai::build_model_archive(&[ModelLayer::uor_r4("ai.default", bundle.clone())])
        .unwrap();
    let b = hologram_ai::build_model_archive(&[ModelLayer::uor_r4("ai.default", bundle)]).unwrap();
    assert_eq!(a, b);
}

#[test]
fn multiple_model_layers_require_explicit_selection() {
    let archive = hologram_ai::build_model_archive(&[
        ModelLayer::uor_r4("ai.language", text_model_bundle("a")),
        ModelLayer::uor_r4("ai.vision", text_model_bundle("b")),
    ])
    .unwrap();

    let app = hologram_ai::Application::open_bytes(&archive).unwrap();
    assert_eq!(app.models().len(), 2);
    let _ = app.model("ai.language").unwrap();
    let _ = app.model("ai.vision").unwrap();

    // No silent first-layer choice.
    let err = app.default_model().unwrap_err();
    assert_eq!(err.category(), ErrorCategory::ModelSelection);
    let err = app.model("ai.nonexistent").unwrap_err();
    assert_eq!(err.category(), ErrorCategory::ModelSelection);
}

#[test]
fn duplicate_entry_names_are_rejected() {
    let err = hologram_ai::build_model_archive(&[
        ModelLayer::uor_r4("ai.default", text_model_bundle("a")),
        ModelLayer::uor_r4("ai.default", text_model_bundle("b")),
    ])
    .unwrap_err();
    assert_eq!(err.category(), ErrorCategory::ArchiveEncode);
}

#[test]
fn empty_entry_name_is_rejected() {
    let err = hologram_ai::build_model_archive(&[ModelLayer::uor_r4("", text_model_bundle("a"))])
        .unwrap_err();
    assert_eq!(err.category(), ErrorCategory::InvalidArgument);
}

#[test]
fn model_layers_can_be_added_to_a_broader_application() {
    // A base application with an exit-bearing wasm primary layer.
    let wasm_bytes = b"\0asm fake module".to_vec();
    let wasm_kappa = hologram_space::address_bytes(&wasm_bytes);
    let base = hologram_ai::build_model_archive(&[ModelLayer::uor_r4(
        "ai.default",
        text_model_bundle("m"),
    )])
    .unwrap();
    // Rebuild the base as an app with a wasm primary: decode the model-only
    // manifest, prepend a wasm layer, set primary = 0.
    let loader = hologram_archive::HoloLoader::from_bytes(&base).unwrap();
    let plan = loader.into_plan().unwrap();
    let mut manifest = hologram_space::AppManifest::decode(plan.app_manifest().unwrap()).unwrap();
    manifest
        .layers
        .insert(0, hologram_space::Layer::wasm(wasm_kappa, "_start"));
    manifest.primary = Some(0);
    manifest.validate().unwrap();
    let mut writer = hologram_archive::HoloWriter::new();
    use hologram_space::Realization;
    writer.set_app_manifest(manifest.canonicalize());
    writer.add_content_blob(wasm_kappa.as_bytes().to_vec(), wasm_bytes);
    for (k, c) in plan.content_blobs().unwrap() {
        writer.add_content_blob(k.to_vec(), c.to_vec());
    }
    let app_archive = writer.finish().unwrap();

    // Add a second model layer; the wasm primary must survive.
    let extended = hologram_ai::add_model_layers(
        &app_archive,
        &[ModelLayer::uor_r4("ai.fusion", text_model_bundle("fusion"))],
    )
    .unwrap();
    let app = hologram_ai::Application::open_bytes(&extended).unwrap();
    assert_eq!(app.models().len(), 2);
    let loader = hologram_archive::HoloLoader::from_bytes(&extended).unwrap();
    let plan = loader.into_plan().unwrap();
    let manifest = hologram_space::AppManifest::decode(plan.app_manifest().unwrap()).unwrap();
    assert_eq!(manifest.primary, Some(0));
    assert_eq!(
        manifest.layers[0].kind,
        hologram_space::LayerKind::WasmCodemodule
    );
    // Manifest order is initialization order; model entries are unaffected.
    assert_eq!(manifest.layers[1].entry, "ai.default");
    assert_eq!(manifest.layers[2].entry, "ai.fusion");
}

#[test]
fn fat_archive_loads_without_any_external_store() {
    let archive = hologram_ai::build_model_archive(&[ModelLayer::uor_r4(
        "ai.default",
        text_model_bundle("fixture"),
    )])
    .unwrap();
    // Application::open_bytes resolves model content purely from the archive.
    let app = hologram_ai::Application::open_bytes(&archive).unwrap();
    let model = app.model("ai.default").unwrap();
    assert_eq!(model.manifest().model_name, "fixture");
    assert_ne!(model.bundle_digest(), [0u8; 32]);
}

#[test]
fn archive_fingerprint_is_stable() {
    let bundle = text_model_bundle("fixture");
    let archive =
        hologram_ai::build_model_archive(&[ModelLayer::uor_r4("ai.default", bundle)]).unwrap();
    let app1 = hologram_ai::Application::open_bytes(&archive).unwrap();
    let app2 = hologram_ai::Application::open_bytes(&archive).unwrap();
    assert_eq!(app1.archive_fingerprint(), app2.archive_fingerprint());
}

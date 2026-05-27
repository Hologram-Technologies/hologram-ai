//! MA (model addressing) — authoritative external-model validation.
//!
//! Validates that hologram-ai's model κ-labels ([`hologram_ai::model_kappa`])
//! are byte-identical to the labels pinned in uor-addr's authoritative
//! `tests/external_models.rs` for published GGUF / ONNX models. This is the
//! cross-implementation attestation that hologram-ai addresses models through
//! uor-addr's canonical form correctly (V&V class MA, architecture §8).
//!
//! For each pinned model the test:
//!   1. ensures the model is cached locally, downloading on first run and
//!      verifying the bytes against the pinned file SHA-256 (a corrupted /
//!      substituted download is rejected);
//!   2. asserts `hologram_ai::model_kappa(...).address` equals the pin;
//!   3. asserts format auto-detection agrees; and
//!   4. round-trips the replayable TC-05 witness (`witness.verify()`).
//!
//! Gated behind `HOLOGRAM_AI_LIVE=1` (the downloads need network access and
//! total ~635 MB). The pins are copied verbatim from uor-addr 0.2.0's
//! `tests/external_models.rs` (sha256 σ-axis).

use std::path::PathBuf;
use std::process::Command;

use hologram_ai::{
    component_kappa, compose_model, compose_models, model_kappa, ModelCompiler, ModelFormat,
    ModelSource,
};

struct Pinned {
    name: &'static str,
    url: &'static str,
    file_sha256: &'static str,
    kappa: &'static str,
    format: ModelFormat,
}

/// Pins copied verbatim from uor-addr 0.2.0 `tests/external_models.rs`.
const MODELS: &[Pinned] = &[
    Pinned {
        name: "qwen2-0_5b-instruct-q8_0.gguf",
        url: "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q8_0.gguf",
        file_sha256: "834f4115ad5a836c9f17716b1577290fda96de3deb881ba45a4d5476fd202e96",
        kappa: "sha256:66c2ea8fa51317c6da91d10f131a5de64d45cb859edaf7a4f8d2557277f45b2d",
        format: ModelFormat::Gguf,
    },
    Pinned {
        name: "mobilenetv2-7.onnx",
        url: "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        file_sha256: "c1c513582d56afceff8516c73804e484c81c6a830712ab6d682253f4a3cd042f",
        kappa: "sha256:f71c815228869e2c56ad00fcf4691ffbad45ecb72a503eef35cbaabe40287378",
        format: ModelFormat::Onnx,
    },
    Pinned {
        name: "all-MiniLM-L6-v2.onnx",
        url: "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        file_sha256: "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
        kappa: "sha256:a036c7fec3409bb71116dcf79a37ce368166898b8621a75bc19299340e127422",
        format: ModelFormat::Onnx,
    },
];

fn live() -> bool {
    std::env::var("HOLOGRAM_AI_LIVE").as_deref() == Ok("1")
}

fn cache_dir() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/target/ma-models"))
}

fn file_sha256_hex(path: &PathBuf) -> String {
    let out = Command::new("sha256sum").arg(path).output().expect("run sha256sum");
    assert!(out.status.success(), "sha256sum failed");
    String::from_utf8(out.stdout).unwrap()[..64].to_string()
}

fn ensure_cached(m: &Pinned) -> PathBuf {
    let path = cache_dir().join(m.name);
    if !path.exists() {
        std::fs::create_dir_all(cache_dir()).unwrap();
        let status = Command::new("curl")
            .args(["-sL", "--fail", "-o"])
            .arg(&path)
            .arg(m.url)
            .status()
            .expect("run curl");
        assert!(status.success(), "download failed for {}", m.name);
    }
    assert_eq!(
        file_sha256_hex(&path),
        m.file_sha256,
        "{}: cached file SHA-256 mismatch — corrupted/substituted (delete cache to re-download)",
        m.name
    );
    path
}

/// Address a real (local) ONNX model through hologram-ai and confirm it
/// yields a well-formed SHA-256 κ-label with a witness that replays — the
/// no-network half of MA, exercising the same `uor_addr::onnx` pipeline the
/// pinned-corpus test validates byte-for-byte.
#[test]
fn local_onnx_fixture_addresses_and_witness_verifies() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hologram-ai-conformance/fixtures/layer_norm.onnx");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"));

    assert_eq!(ModelFormat::detect(&bytes), Some(ModelFormat::Onnx));

    let outcome = model_kappa(ModelFormat::Onnx, &bytes).expect("onnx address");
    let label = outcome.address.as_str();
    assert!(label.starts_with("sha256:"), "got {label}");
    assert_eq!(label.len(), 71, "sha256 κ-label is 71 bytes");

    // Deterministic: re-addressing the same bytes yields the same label.
    let again = model_kappa(ModelFormat::Onnx, &bytes).expect("onnx address (again)");
    assert_eq!(again.address.as_str(), label, "addressing must be deterministic");

    // TC-05 witness replays to the same κ-label.
    assert_eq!(
        outcome.witness.verify().expect("witness verify"),
        outcome.address,
    );
}

/// The compile pipeline carries the source model's κ-label in
/// `HoloArchive.metadata` (architecture §8 — model identity for dedup /
/// warm-start), and it equals the label minted by addressing the bytes
/// directly.
#[test]
fn compile_populates_model_kappa_label() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hologram-ai-conformance/fixtures/layer_norm.onnx");
    let bytes = std::fs::read(&path).unwrap();
    let expected = model_kappa(ModelFormat::Onnx, &bytes).unwrap().address.as_str().to_string();

    let compiler = ModelCompiler {
        seq_len_override: Some(4),
        ..Default::default()
    };
    let archive = compiler
        .compile(ModelSource::OnnxBytes(bytes))
        .expect("compile failed");

    assert_eq!(
        archive.metadata.kappa_label.as_deref(),
        Some(expected.as_str()),
        "compile must carry the source model's κ-label"
    );
}

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../hologram-ai-conformance/fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {path:?}: {e}"))
}

/// MA-2: a multi-component model's identity is the E₈ composition of its
/// components' κ-labels, and it is **independent of component order** — the
/// same parts compose to the same label however they were assembled.
#[test]
fn multi_component_composition_is_order_independent() {
    let a = fixture("layer_norm.onnx");
    let b = fixture("rms_norm.onnx");
    let c = fixture("softmax_dyn_seq.onnx");

    let ka = component_kappa(&a).unwrap();
    let kb = component_kappa(&b).unwrap();
    let kc = component_kappa(&c).unwrap();

    // Two-component commutativity.
    assert_eq!(
        compose_model(&[ka, kb]).unwrap(),
        compose_model(&[kb, ka]).unwrap(),
        "compose(a,b) must equal compose(b,a)"
    );

    // Three-component: every permutation yields the same identity.
    let canonical = compose_model(&[ka, kb, kc]).unwrap();
    for perm in [[ka, kc, kb], [kb, ka, kc], [kb, kc, ka], [kc, ka, kb], [kc, kb, ka]] {
        assert_eq!(
            compose_model(&perm).unwrap(),
            canonical,
            "composition must be order-independent across all permutations"
        );
    }

    // The composed identity is distinct from any single component.
    assert_ne!(canonical, ka);
    assert_ne!(canonical, kb);
    assert_ne!(canonical, kc);

    // The bytes-level convenience wrapper agrees with the label-level compose.
    let refs: [&[u8]; 3] = [&a, &b, &c];
    assert_eq!(compose_models(&refs).unwrap(), canonical);
}

#[test]
#[ignore = "MA: requires HOLOGRAM_AI_LIVE=1 + network (~635 MB)"]
fn pinned_external_models_address_to_uor_addr_labels() {
    if !live() {
        return;
    }
    for m in MODELS {
        let path = ensure_cached(m);
        let bytes = std::fs::read(&path).unwrap();

        // Format auto-detection agrees with the pin.
        assert_eq!(
            ModelFormat::detect(&bytes),
            Some(m.format),
            "{}: format detection",
            m.name
        );

        // hologram-ai κ-label == uor-addr pin.
        let outcome = model_kappa(m.format, &bytes).expect("model_kappa");
        assert_eq!(outcome.address.as_str(), m.kappa, "{}: κ-label", m.name);

        // Replayable TC-05 witness round-trips.
        assert_eq!(
            outcome.witness.verify().expect("witness verify"),
            outcome.address,
            "{}: witness verify",
            m.name
        );
    }
}

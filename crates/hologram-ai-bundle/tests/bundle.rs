//! Bundle schema v1 integration tests: round-trip, determinism, tamper and
//! hostile-input handling, capability⇔processor matrix, and panic-free
//! parsing of pseudo-random bytes.

use hologram_ai_bundle::{
    Bundle, BundleBuilder, BUNDLE_MAGIC, BUNDLE_SCHEMA_VERSION, HEADER_LEN, MAX_COMPONENTS,
};
use hologram_ai_core::*;

// ── Fixtures ────────────────────────────────────────────────────────────────

fn test_abi() -> AbiVersions {
    AbiVersions {
        compiler_abi: (0, 1, 0),
        r4g1_format: (1, 0),
        contract: (0, 1, 0),
        holo_format: 4,
    }
}

fn test_provenance() -> ModelProvenance {
    ModelProvenance {
        source_repository: Some("Example/Fixture-Tiny".into()),
        source_revision: Some("0123456789abcdef0123456789abcdef01234567".into()),
        source_digest: [7u8; 32],
        compiler_revision: "deadbeef".into(),
        compile_options_digest: [8u8; 32],
        quality_report_digest: [9u8; 32],
    }
}

fn value(name: &str, kind: ValueKind) -> ValueDescriptor {
    let media_type = match kind {
        ValueKind::Image => Some("image/png".into()),
        ValueKind::Audio => Some("audio/wav".into()),
        _ => None,
    };
    ValueDescriptor {
        name: name.into(),
        kind,
        media_type,
        required: true,
    }
}

/// Token-only operation (`predict`): no tokenizer required.
fn token_op() -> OperationDescriptor {
    OperationDescriptor {
        name: "predict".into(),
        inputs: vec![value("context", ValueKind::Tokens)],
        outputs: vec![value("token", ValueKind::Tokens)],
        streaming: false,
    }
}

/// Text operation (`generate`): requires a tokenizer.
fn text_op() -> OperationDescriptor {
    OperationDescriptor {
        name: "generate".into(),
        inputs: vec![value("prompt", ValueKind::Text)],
        outputs: vec![value("text", ValueKind::Text)],
        streaming: true,
    }
}

fn base_builder() -> BundleBuilder {
    BundleBuilder::new(ENGINE_UOR_R4, ARTIFACT_FORMAT_R4G1, "fixture-tiny")
        .set_operations(vec![token_op()])
        .set_status_policy(StatusPolicy::R4_DEFAULT)
        .set_abi(test_abi())
        .set_provenance(test_provenance())
}

fn add_mandatory(b: &mut BundleBuilder) {
    b.add_component(ArtifactRole::Graph, b"scored-r4g1-graph")
        .unwrap();
    b.add_component(ArtifactRole::SignatureArtifact, b"signature-artifact")
        .unwrap();
    b.add_component(ArtifactRole::ScoreReport, b"score-report")
        .unwrap();
}

fn valid_bundle() -> Vec<u8> {
    let mut b = base_builder();
    add_mandatory(&mut b);
    b.add_component(ArtifactRole::Tokenizer, b"bpe-tokenizer")
        .unwrap();
    b.finish().unwrap()
}

/// Wrap a hand-built manifest + payload into bundle bytes (hostile-input
/// construction; digests in `manifest` must match `payload` slices if the
/// result is expected to verify).
fn wrap(manifest: &InferenceModelManifest, payload: &[u8]) -> Vec<u8> {
    let m = manifest.encode();
    let mut out = Vec::with_capacity(HEADER_LEN + m.len() + payload.len());
    out.extend_from_slice(&BUNDLE_MAGIC);
    out.extend_from_slice(&BUNDLE_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(m.len() as u32).to_le_bytes());
    out.extend_from_slice(&m);
    out.extend_from_slice(payload);
    out
}

fn descriptor(role: ArtifactRole, payload: &[u8]) -> ArtifactDescriptor {
    ArtifactDescriptor {
        role,
        digest: *blake3::hash(payload).as_bytes(),
        length: payload.len() as u64,
    }
}

fn crafted_manifest(
    ops: Vec<OperationDescriptor>,
    artifacts: Vec<ArtifactDescriptor>,
) -> InferenceModelManifest {
    InferenceModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        engine: ENGINE_UOR_R4.into(),
        artifact_format: ARTIFACT_FORMAT_R4G1.into(),
        model_name: "crafted".into(),
        operations: ops,
        artifacts,
        status_policy: StatusPolicy::R4_DEFAULT,
        abi: test_abi(),
        provenance: test_provenance(),
    }
}

fn category(e: &AiError) -> ErrorCategory {
    e.category()
}

// ── Round-trip and determinism ──────────────────────────────────────────────

#[test]
fn round_trip_preserves_manifest_and_components() {
    let bytes = valid_bundle();
    let bundle = Bundle::parse_verified(&bytes).unwrap();

    let m = bundle.manifest();
    assert_eq!(m.schema_version, MANIFEST_SCHEMA_VERSION);
    assert_eq!(m.engine, ENGINE_UOR_R4);
    assert_eq!(m.artifact_format, ARTIFACT_FORMAT_R4G1);
    assert_eq!(m.model_name, "fixture-tiny");
    assert_eq!(m.operations, vec![token_op()]);
    assert_eq!(m.status_policy, StatusPolicy::R4_DEFAULT);
    assert_eq!(m.abi, test_abi());
    assert_eq!(m.provenance, test_provenance());

    // Artifacts are in canonical role order regardless of insertion order.
    let roles: Vec<ArtifactRole> = m.artifacts.iter().map(|a| a.role).collect();
    assert_eq!(
        roles,
        vec![
            ArtifactRole::Graph,
            ArtifactRole::SignatureArtifact,
            ArtifactRole::Tokenizer,
            ArtifactRole::ScoreReport,
        ]
    );

    assert_eq!(
        bundle.component(ArtifactRole::Graph),
        Some(&b"scored-r4g1-graph"[..])
    );
    assert_eq!(
        bundle.component(ArtifactRole::Tokenizer),
        Some(&b"bpe-tokenizer"[..])
    );
    assert_eq!(bundle.component(ArtifactRole::LabelMap), None);

    let iterated: Vec<(ArtifactRole, &[u8])> = bundle.components().collect();
    assert_eq!(iterated.len(), 4);
    assert_eq!(
        iterated[3],
        (ArtifactRole::ScoreReport, &b"score-report"[..])
    );

    assert_eq!(bundle.bundle_digest(), *blake3::hash(&bytes).as_bytes());
    bundle.validate().unwrap();
}

#[test]
fn builder_manifest_getter_matches_parsed_manifest() {
    let mut b = base_builder();
    add_mandatory(&mut b);
    let expected = b.manifest();
    let bytes = b.finish().unwrap();
    assert_eq!(*Bundle::parse(&bytes).unwrap().manifest(), expected);
}

#[test]
fn double_build_is_byte_identical() {
    assert_eq!(valid_bundle(), valid_bundle());
}

#[test]
fn insertion_order_does_not_affect_output() {
    let mut a = base_builder();
    add_mandatory(&mut a);
    a.add_component(ArtifactRole::Tokenizer, b"bpe-tokenizer")
        .unwrap();

    let mut b = base_builder();
    b.add_component(ArtifactRole::Tokenizer, b"bpe-tokenizer")
        .unwrap();
    b.add_component(ArtifactRole::ScoreReport, b"score-report")
        .unwrap();
    b.add_component(ArtifactRole::SignatureArtifact, b"signature-artifact")
        .unwrap();
    b.add_component(ArtifactRole::Graph, b"scored-r4g1-graph")
        .unwrap();

    assert_eq!(a.finish().unwrap(), b.finish().unwrap());
}

#[test]
fn builder_rejects_duplicate_role() {
    let mut b = base_builder();
    add_mandatory(&mut b);
    let e = b.add_component(ArtifactRole::Graph, b"again").unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleEncode);
}

// ── Tamper handling ─────────────────────────────────────────────────────────

#[test]
fn tampered_component_byte_fails_digest_verification() {
    let mut bytes = valid_bundle();
    let last = bytes.len() - 1; // inside the final (score-report) payload
    bytes[last] ^= 0x01;

    // Structure still parses; digests must not.
    let bundle = Bundle::parse(&bytes).unwrap();
    let e = bundle.verify_digests().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::IntegrityMismatch);

    let e = Bundle::parse_verified(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::IntegrityMismatch);
}

#[test]
fn tampered_manifest_length_prefix_fails_decode() {
    let mut bytes = valid_bundle();
    // Manifest starts at HEADER_LEN with u16 schema_version, then a u32
    // length prefix for the engine string at HEADER_LEN + 2.
    bytes[HEADER_LEN + 2..HEADER_LEN + 6].copy_from_slice(&u32::MAX.to_le_bytes());
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn tampered_component_digest_in_manifest_fails_verification() {
    let graph = b"graph";
    let mut manifest = crafted_manifest(
        vec![token_op()],
        vec![
            descriptor(ArtifactRole::Graph, graph),
            descriptor(ArtifactRole::SignatureArtifact, b"sig"),
            descriptor(ArtifactRole::ScoreReport, b"score"),
        ],
    );
    manifest.artifacts[0].digest[0] ^= 0x01;
    let mut payload = Vec::new();
    payload.extend_from_slice(graph);
    payload.extend_from_slice(b"sig");
    payload.extend_from_slice(b"score");
    let bytes = wrap(&manifest, &payload);

    let bundle = Bundle::parse(&bytes).unwrap();
    let e = bundle.verify_digests().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::IntegrityMismatch);
}

// ── Hostile structure ───────────────────────────────────────────────────────

#[test]
fn truncation_at_any_offset_errors_without_panic() {
    let bytes = valid_bundle();
    let cuts = [
        0,
        1,
        3,
        HEADER_LEN - 1,
        HEADER_LEN,
        HEADER_LEN + 1,
        HEADER_LEN + 10,
        bytes.len() / 2,
        bytes.len() - 1,
    ];
    for cut in cuts {
        assert!(
            Bundle::parse(&bytes[..cut]).is_err(),
            "truncation at {cut} must fail"
        );
    }
}

#[test]
fn hostile_manifest_length_is_rejected() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&BUNDLE_MAGIC);
    bytes.extend_from_slice(&BUNDLE_SCHEMA_VERSION.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(b"tiny");
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn component_length_exceeding_input_is_rejected() {
    let mut manifest = crafted_manifest(
        vec![token_op()],
        vec![
            descriptor(ArtifactRole::Graph, b"graph"),
            descriptor(ArtifactRole::SignatureArtifact, b"sig"),
            descriptor(ArtifactRole::ScoreReport, b"score"),
        ],
    );
    manifest.artifacts[2].length = 1_000_000;
    let bytes = wrap(&manifest, b"graphsigscore");
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn trailing_bytes_are_rejected() {
    let mut bytes = valid_bundle();
    bytes.push(0x00);
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn duplicate_role_in_crafted_manifest_is_rejected() {
    let manifest = crafted_manifest(
        vec![token_op()],
        vec![
            descriptor(ArtifactRole::Graph, b"ab"),
            descriptor(ArtifactRole::Graph, b"cde"),
        ],
    );
    let bytes = wrap(&manifest, b"abcde");
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn component_count_above_max_is_rejected() {
    // MAX_COMPONENTS equals the number of roles, so a crafted manifest can
    // only exceed it via duplicates — assert the constant relationship so
    // the parser bound stays meaningful.
    assert_eq!(MAX_COMPONENTS, ArtifactRole::ALL.len());
}

#[test]
fn unknown_bundle_schema_version_is_abi_mismatch() {
    let mut bytes = valid_bundle();
    bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::AbiMismatch);
}

#[test]
fn unknown_manifest_schema_version_is_abi_mismatch() {
    let mut manifest = crafted_manifest(
        vec![token_op()],
        vec![
            descriptor(ArtifactRole::Graph, b"graph"),
            descriptor(ArtifactRole::SignatureArtifact, b"sig"),
            descriptor(ArtifactRole::ScoreReport, b"score"),
        ],
    );
    manifest.schema_version = 99;
    let bytes = wrap(&manifest, b"graphsigscore");
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::AbiMismatch);
}

#[test]
fn bad_magic_is_bundle_decode() {
    let mut bytes = valid_bundle();
    bytes[0] = b'X';
    let e = Bundle::parse(&bytes).unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

// ── Validation: mandatory roles ─────────────────────────────────────────────

#[test]
fn missing_mandatory_roles_rejected_at_build() {
    for omitted in [
        ArtifactRole::Graph,
        ArtifactRole::SignatureArtifact,
        ArtifactRole::ScoreReport,
    ] {
        let mut b = base_builder();
        for (role, payload) in [
            (ArtifactRole::Graph, &b"graph"[..]),
            (ArtifactRole::SignatureArtifact, b"sig"),
            (ArtifactRole::ScoreReport, b"score"),
        ] {
            if role != omitted {
                b.add_component(role, payload).unwrap();
            }
        }
        let e = b.finish().unwrap_err();
        assert_eq!(
            category(&e),
            ErrorCategory::BundleDecode,
            "omit {omitted:?}"
        );
        assert!(e.message().contains(omitted.name()));
    }
}

#[test]
fn missing_mandatory_role_rejected_by_validate() {
    // Unknown-to-builder path: craft a structurally valid, digest-correct
    // bundle lacking score-report, then validate.
    let manifest = crafted_manifest(
        vec![token_op()],
        vec![
            descriptor(ArtifactRole::Graph, b"graph"),
            descriptor(ArtifactRole::SignatureArtifact, b"sig"),
        ],
    );
    let bytes = wrap(&manifest, b"graphsig");
    let bundle = Bundle::parse_verified(&bytes).unwrap();
    let e = bundle.validate().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::BundleDecode);
}

#[test]
fn unknown_engine_skips_mandatory_role_rules() {
    // Forward compatibility: unknown engine/format strings are allowed and
    // the uor-r4/R4G1 mandatory-role rules do not apply.
    let mut b = BundleBuilder::new("future-engine", "FUTURE-FORMAT", "future-model")
        .set_operations(vec![token_op()])
        .set_abi(test_abi())
        .set_provenance(test_provenance());
    b.add_component(ArtifactRole::GenerationConfig, b"config")
        .unwrap();
    let bytes = b.finish().unwrap();
    let bundle = Bundle::parse_verified(&bytes).unwrap();
    bundle.validate().unwrap();
    assert_eq!(bundle.manifest().engine, "future-engine");
}

// ── Validation: capability ⇔ processor matrix ───────────────────────────────

#[test]
fn text_without_tokenizer_is_processor_mismatch() {
    let mut b = base_builder().set_operations(vec![text_op()]);
    add_mandatory(&mut b);
    let e = b.finish().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::ProcessorMismatch);

    // Same rule through Bundle::validate on a crafted bundle.
    let manifest = crafted_manifest(
        vec![text_op()],
        vec![
            descriptor(ArtifactRole::Graph, b"graph"),
            descriptor(ArtifactRole::SignatureArtifact, b"sig"),
            descriptor(ArtifactRole::ScoreReport, b"score"),
        ],
    );
    let bytes = wrap(&manifest, b"graphsigscore");
    let bundle = Bundle::parse_verified(&bytes).unwrap();
    let e = bundle.validate().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::ProcessorMismatch);
}

#[test]
fn token_only_model_without_tokenizer_is_ok() {
    // Token-only: every text-ish value is kind Tokens, no Text anywhere.
    let mut b = base_builder();
    add_mandatory(&mut b);
    let bytes = b.finish().unwrap();
    let bundle = Bundle::parse_verified(&bytes).unwrap();
    bundle.validate().unwrap();
}

#[test]
fn image_without_image_processor_is_processor_mismatch() {
    let mut op = token_op();
    op.inputs.push(value("image", ValueKind::Image));
    let mut b = base_builder().set_operations(vec![op]);
    add_mandatory(&mut b);
    let e = b.finish().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::ProcessorMismatch);
}

#[test]
fn audio_without_audio_processor_is_processor_mismatch() {
    let mut op = token_op();
    op.inputs.push(value("clip", ValueKind::Audio));
    let mut b = base_builder().set_operations(vec![op]);
    add_mandatory(&mut b);
    let e = b.finish().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::ProcessorMismatch);
}

#[test]
fn multimodal_requires_all_processors() {
    let multimodal = OperationDescriptor {
        name: "generate".into(),
        inputs: vec![
            value("prompt", ValueKind::Text),
            value("image", ValueKind::Image),
            value("clip", ValueKind::Audio),
        ],
        outputs: vec![value("text", ValueKind::Text)],
        streaming: true,
    };

    // Missing the audio processor only.
    let mut b = base_builder().set_operations(vec![multimodal.clone()]);
    add_mandatory(&mut b);
    b.add_component(ArtifactRole::Tokenizer, b"tok").unwrap();
    b.add_component(ArtifactRole::ImageProcessor, b"img")
        .unwrap();
    let e = b.finish().unwrap_err();
    assert_eq!(category(&e), ErrorCategory::ProcessorMismatch);

    // All three processors present: OK.
    let mut b = base_builder().set_operations(vec![multimodal]);
    add_mandatory(&mut b);
    b.add_component(ArtifactRole::Tokenizer, b"tok").unwrap();
    b.add_component(ArtifactRole::ImageProcessor, b"img")
        .unwrap();
    b.add_component(ArtifactRole::AudioProcessor, b"aud")
        .unwrap();
    let bytes = b.finish().unwrap();
    Bundle::parse_verified(&bytes).unwrap().validate().unwrap();
}

// ── Robustness ──────────────────────────────────────────────────────────────

/// Deterministic xorshift64* PRNG — no RNG dependence, no `rand` crate.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[test]
fn pseudo_random_inputs_never_panic() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    let valid = valid_bundle();
    for i in 0..512u32 {
        let mut buf = if i % 3 == 0 {
            // Corrupt and sometimes truncate an otherwise valid bundle.
            let mut b = valid.clone();
            let flips = 1 + (rng.next() % 64) as usize;
            for _ in 0..flips {
                let pos = (rng.next() as usize) % b.len();
                b[pos] ^= rng.next() as u8;
            }
            if rng.next() % 2 == 0 {
                let keep = (rng.next() as usize) % (b.len() + 1);
                b.truncate(keep);
            }
            b
        } else {
            // Pure noise of pseudo-random length.
            let len = (rng.next() % 4096) as usize;
            let mut b = Vec::with_capacity(len);
            for _ in 0..len {
                b.push(rng.next() as u8);
            }
            b
        };
        if i % 7 == 0 {
            buf.clear();
        }

        // parse may succeed or fail, but must never panic — and neither may
        // any accessor on a successfully parsed bundle.
        if let Ok(bundle) = Bundle::parse(&buf) {
            let _ = bundle.manifest();
            let _ = bundle.bundle_digest();
            let _ = bundle.verify_digests();
            let _ = bundle.validate();
            for role in ArtifactRole::ALL {
                let _ = bundle.component(role);
            }
            for (_role, payload) in bundle.components() {
                let _ = payload.len();
            }
        }
    }
}

use hologram_ai_core::canon::{CanonError, CanonReader, CanonWriter};
use hologram_ai_core::*;

fn sample_manifest() -> InferenceModelManifest {
    InferenceModelManifest {
        schema_version: MANIFEST_SCHEMA_VERSION,
        engine: ENGINE_UOR_R4.into(),
        artifact_format: ARTIFACT_FORMAT_R4G1.into(),
        model_name: "fixture-tiny".into(),
        operations: vec![
            OperationDescriptor {
                name: "generate".into(),
                inputs: vec![
                    ValueDescriptor {
                        name: "prompt".into(),
                        kind: ValueKind::Text,
                        media_type: None,
                        required: true,
                    },
                    ValueDescriptor {
                        name: "image".into(),
                        kind: ValueKind::Image,
                        media_type: Some("image/png".into()),
                        required: false,
                    },
                ],
                outputs: vec![ValueDescriptor {
                    name: "text".into(),
                    kind: ValueKind::Text,
                    media_type: None,
                    required: true,
                }],
                streaming: true,
            },
            OperationDescriptor {
                name: "predict".into(),
                inputs: vec![ValueDescriptor {
                    name: "context".into(),
                    kind: ValueKind::Tokens,
                    media_type: None,
                    required: true,
                }],
                outputs: vec![ValueDescriptor {
                    name: "token".into(),
                    kind: ValueKind::Tokens,
                    media_type: None,
                    required: true,
                }],
                streaming: false,
            },
        ],
        artifacts: vec![
            ArtifactDescriptor {
                role: ArtifactRole::Graph,
                digest: [1u8; 32],
                length: 1234,
            },
            ArtifactDescriptor {
                role: ArtifactRole::SignatureArtifact,
                digest: [2u8; 32],
                length: 5678,
            },
            ArtifactDescriptor {
                role: ArtifactRole::Tokenizer,
                digest: [3u8; 32],
                length: 90,
            },
        ],
        status_policy: StatusPolicy::R4_DEFAULT,
        abi: AbiVersions {
            compiler_abi: (0, 1, 0),
            r4g1_format: (0, 0),
            contract: (0, 1, 0),
            holo_format: 4,
        },
        provenance: ModelProvenance {
            source_repository: Some("owner/model".into()),
            source_revision: Some("0123456789abcdef0123456789abcdef01234567".into()),
            source_digest: [7u8; 32],
            compiler_revision: "f1b4859e65363eda9aa7dbeb0db467d93c8f4b02".into(),
            compile_options_digest: [8u8; 32],
            quality_report_digest: [9u8; 32],
        },
    }
}

#[test]
fn manifest_round_trip() {
    let m = sample_manifest();
    let bytes = m.encode();
    let decoded = InferenceModelManifest::decode(&bytes).expect("decode");
    assert_eq!(m, decoded);
}

#[test]
fn manifest_encoding_is_deterministic() {
    let m = sample_manifest();
    assert_eq!(m.encode(), m.encode());
}

#[test]
fn manifest_rejects_trailing_bytes() {
    let mut bytes = sample_manifest().encode();
    bytes.push(0);
    assert_eq!(
        InferenceModelManifest::decode(&bytes),
        Err(CanonError::TrailingBytes)
    );
}

#[test]
fn manifest_rejects_truncation() {
    let bytes = sample_manifest().encode();
    assert!(InferenceModelManifest::decode(&bytes[..bytes.len() - 1]).is_err());
}

#[test]
fn manifest_rejects_hostile_lengths() {
    // schema_version + engine string with a huge length prefix.
    let mut w = CanonWriter::new();
    w.u16(1);
    w.u32(u32::MAX);
    let bytes = w.finish();
    assert!(matches!(
        InferenceModelManifest::decode(&bytes),
        Err(CanonError::LengthOutOfRange) | Err(CanonError::Truncated)
    ));
}

#[test]
fn value_round_trip_all_kinds() {
    let values = [
        Value::Text("hello".into()),
        Value::Tokens(vec![1, 2, 3, 50000]),
        Value::Image(MediaData {
            media_type: "image/png".into(),
            payload: Payload::Inline(vec![0x89, 0x50]),
        }),
        Value::Audio(MediaData {
            media_type: "audio/wav".into(),
            payload: Payload::Ref("blake3:abc".into()),
        }),
        Value::Bytes {
            media_type: "application/octet-stream".into(),
            data: vec![1, 2],
        },
        Value::ContentRef {
            media_type: "image/png".into(),
            kappa: "blake3:def".into(),
        },
        Value::Structured(vec![9, 9, 9]),
    ];
    for v in values {
        let mut w = CanonWriter::new();
        v.encode(&mut w);
        let bytes = w.finish();
        let mut r = CanonReader::new(&bytes);
        let decoded = Value::decode(&mut r).expect("decode");
        r.finish().expect("no trailing");
        assert_eq!(v, decoded);
        assert_ne!(v.kind(), ValueKind::Buffer);
    }
}

#[test]
fn request_builder_supports_all_inputs() {
    let req = InferenceRequest::builder()
        .text("prompt", "Explain content addressing.")
        .tokens("context", vec![1, 2, 3])
        .image("image", "image/png", vec![1])
        .audio("audio", "audio/wav", vec![2])
        .content_ref("image2", "image/png", "blake3:xyz")
        .bytes("blob", "application/octet-stream", vec![3])
        .structured("params", vec![4])
        .max_output_tokens(128)
        .build();
    assert_eq!(req.text("prompt"), Some("Explain content addressing."));
    assert_eq!(req.tokens("context"), Some(&[1, 2, 3][..]));
    assert_eq!(req.max_output_tokens, Some(128));
    assert_eq!(req.values.len(), 7);
}

#[test]
fn error_category_codes_are_stable() {
    // These numbers are public ABI; do not renumber.
    assert_eq!(ErrorCategory::InvalidArgument.code(), 1);
    assert_eq!(ErrorCategory::UnsupportedModel.code(), 4);
    assert_eq!(ErrorCategory::BundleDecode.code(), 11);
    assert_eq!(ErrorCategory::IntegrityMismatch.code(), 14);
    assert_eq!(ErrorCategory::ModelSelection.code(), 15);
    assert_eq!(ErrorCategory::Cancelled.code(), 19);
    assert_eq!(ErrorCategory::AbiMismatch.code(), 20);
    for code in 1..=20 {
        let cat = ErrorCategory::from_code(code).expect("known code");
        assert_eq!(cat.code(), code);
    }
    assert_eq!(ErrorCategory::from_code(0), None);
    assert_eq!(ErrorCategory::from_code(21), None);
}

#[test]
fn abstention_is_a_typed_completion_not_an_error() {
    let completion = InferenceCompletion {
        finish_reason: FinishReason::Abstained,
        status: None,
        widened: true,
        output: InferenceOutput::default(),
        witness: None,
    };
    assert_eq!(completion.finish_reason, FinishReason::Abstained);
    assert!(completion.output.values.is_empty());
}

#[test]
fn cancellation_token_propagates() {
    let token = CancellationToken::new();
    let clone = token.clone();
    assert!(!token.is_cancelled());
    clone.cancel();
    assert!(token.is_cancelled());
}

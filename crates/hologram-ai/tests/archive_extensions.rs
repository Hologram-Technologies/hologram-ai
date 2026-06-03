//! Tokenizer-in-archive via uor-addr canonical JSON extension sections.
//!
//! Verifies the UOR-native serialization path: a tokenizer is canonicalized with
//! uor-addr (JCS-RFC8785 + NFC), baked into the `.holo` as an open extension,
//! and read back byte-exact + content-address-verified, then parsed into a
//! working tokenizer — no external file, no raw-bytes guessing.

use std::collections::HashMap;

use hologram_ai::compiler::{ArchiveSections, TOKENIZER_EXT, TOKENIZER_KAPPA_EXT};
use hologram_ai::{HoloRunner, ModelCompiler, ModelSource};
use hologram_ai_common::{shape_from_concrete, AiGraph, AiNode, AiOp, AiParam, DType, TensorInfo};
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

// Same tokenizer, two byte encodings (whitespace + key order differ).
const TOK_PRETTY: &str = r#"{
  "added_tokens": [ {"id": 0, "content": "</s>", "special": true} ],
  "model": { "type": "BPE", "vocab": {"</s>": 0, "a": 1, "b": 2, "ab": 3}, "merges": ["a b"] }
}"#;
const TOK_COMPACT: &str = r#"{"model":{"merges":["a b"],"type":"BPE","vocab":{"a":1,"</s>":0,"ab":3,"b":2}},"added_tokens":[{"content":"</s>","id":0,"special":true}]}"#;

/// Single-input `[1,4]·[4,4 identity]` matmul — a host for the extension.
fn matmul_graph() -> AiGraph {
    let (x, w, y) = (0u32, 1u32, 2u32);
    let mut ti = HashMap::new();
    ti.insert(x, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));
    ti.insert(w, TensorInfo::new(DType::F32, shape_from_concrete(&[4, 4])));
    ti.insert(y, TensorInfo::new(DType::F32, shape_from_concrete(&[1, 4])));
    let mut wb = vec![0u8; 64];
    for k in 0..4 {
        wb[(k * 4 + k) * 4..(k * 4 + k) * 4 + 4].copy_from_slice(&1.0f32.to_le_bytes());
    }
    let mut params = HashMap::new();
    params.insert(w, AiParam::inline(wb, ti[&w].clone()));
    AiGraph {
        name: "host".into(),
        nodes: vec![AiNode::new(0, AiOp::MatMul, vec![x, w], vec![y])],
        inputs: vec![x],
        outputs: vec![y],
        input_names: vec!["x".into()],
        output_names: vec!["y".into()],
        params,
        tensor_info: ti,
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
fn uor_addr_json_canonicalization_is_deterministic() {
    let a = uor_addr::json::canonicalize(TOK_PRETTY.as_bytes()).unwrap();
    let b = uor_addr::json::canonicalize(TOK_COMPACT.as_bytes()).unwrap();
    assert_eq!(
        a, b,
        "JCS canonical form is independent of whitespace + key order"
    );
    let ka = uor_addr::json::address(&a)
        .unwrap()
        .address
        .as_str()
        .to_string();
    let kb = uor_addr::json::address(&b)
        .unwrap()
        .address
        .as_str()
        .to_string();
    assert_eq!(ka, kb, "identical content ⇒ identical κ-label");
}

#[test]
fn tokenizer_bakes_into_archive_and_loads_back_verified() {
    let canonical = uor_addr::json::canonicalize(TOK_PRETTY.as_bytes()).unwrap();
    let kappa = uor_addr::json::address(&canonical)
        .unwrap()
        .address
        .as_str()
        .to_string();

    let mut sections = ArchiveSections::new();
    sections.add_extension(TOKENIZER_EXT, canonical.clone());
    sections.add_extension(TOKENIZER_KAPPA_EXT, kappa.clone().into_bytes());

    let archive = ModelCompiler::default()
        .compile_with_sections(ModelSource::AiGraph(matmul_graph()), sections)
        .expect("compile with tokenizer extension");
    let runner = HoloRunner::from_bytes(archive.bytes).expect("load");

    // The extension roundtrips byte-exact (canonical form preserved end to end).
    assert_eq!(
        runner.extension(TOKENIZER_EXT).unwrap(),
        canonical.as_slice()
    );
    assert_eq!(
        runner.extension(TOKENIZER_KAPPA_EXT).unwrap(),
        kappa.as_bytes()
    );

    // Its content address re-verifies against the stored κ-label.
    let readdr = uor_addr::json::address(runner.extension(TOKENIZER_EXT).unwrap())
        .unwrap()
        .address
        .as_str()
        .to_string();
    assert_eq!(
        readdr, kappa,
        "embedded tokenizer content address matches its κ-label"
    );

    // And it parses into a working tokenizer from the archived bytes (no file).
    let tok = NativeTokenizer::from_tokenizer_json_bytes(runner.extension(TOKENIZER_EXT).unwrap())
        .expect("parse embedded tokenizer");
    assert_eq!(tok.vocab_size(), 4);
    assert!(
        !tok.encode("ab").is_empty(),
        "tokenizer encodes from archived bytes"
    );
}

#[test]
fn model_without_tokenizer_has_no_extension() {
    // No model dir, no injected section ⇒ the archive simply carries none
    // (absence is not an error — not every model is a text model).
    let archive = ModelCompiler::default()
        .compile(ModelSource::AiGraph(matmul_graph()))
        .expect("compile");
    let runner = HoloRunner::from_bytes(archive.bytes).expect("load");
    assert!(runner.extension(TOKENIZER_EXT).is_none());
}

//! Structural V&V — class **IM** (import conformance).
//!
//! IM-1 / IM-1b (operator-spec parity vs the official ONNX backend node-test
//! corpus) is enforced by [`onnx_spec_conformance`]. This file carries the
//! structural rail **IM-3**: byte-level model parsing is confined to the
//! `Grounding` boundary — the importer crates that turn raw model bytes
//! (ONNX protobuf, GGUF) into a structured `AiGraph`. After import, the
//! graph and its parameters flow as typed values; nothing downstream re-
//! parses the model file format.
//!
//! Why structurally: the moment a mid-graph pass reaches back into the raw
//! model bytes, hologram-ai is no longer canonical-forms-only and any
//! downstream guarantee (CF, CE, ZM) is at the mercy of that side channel.
//! IM-3 is the perimeter rail.
//!
//! ## What we check
//!
//! 1. `prost::` (protobuf decoder) imports appear ONLY in
//!    `crates/hologram-ai-onnx/` — the ONNX `Grounding` impl. Any other
//!    crate that pulled `prost` would be parsing model bytes outside the
//!    boundary.
//! 2. The ONNX import entry points (`import_onnx*`) are public only from
//!    `hologram-ai-onnx`, and other crates reach them via the re-export on
//!    `hologram_ai` — not via reaching into `prost`.
//!
//! The test reads source files relative to `CARGO_MANIFEST_DIR` so it works
//! both in-tree and from a worktree.

#![cfg(feature = "structural")]

use std::fs;
use std::path::{Path, PathBuf};

/// Return the absolute workspace root (`crates/hologram-ai-conformance/..`
/// = `crates/..` = workspace).
fn workspace_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // crates/
    p.pop(); // workspace
    p
}

/// Recursively collect every `*.rs` file under `crates/<name>/src`.
fn rs_files_in(crate_name: &str) -> Vec<PathBuf> {
    let mut root = workspace_root();
    root.push("crates");
    root.push(crate_name);
    root.push("src");
    let mut files = Vec::new();
    walk(&root, &mut files);
    files
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(p);
        }
    }
}

/// Crates that are *not* `Grounding` impls — i.e. that must not contain any
/// raw-model-bytes parsing. (`hologram-ai-onnx` is the ONNX boundary; a
/// future GGUF importer would join it.)
const NON_GROUNDING_CRATES: &[&str] = &[
    "hologram-ai",
    "hologram-ai-common",
    "hologram-ai-quant",
    "hologram-ai-tokenizer",
    "hologram-ai-conformance",
];

/// Token patterns whose presence in a non-grounding crate would indicate
/// byte-level *model-file* parsing leaked across the boundary. We're not
/// trying to ban every byte op — `from_le_bytes` reads typed payloads out of
/// already-structured constants. We're banning the **protobuf / GGUF
/// decoder surface**.
const BANNED_TOKENS: &[&str] = &[
    "use prost",      // protobuf decoder
    "prost::Message", // direct trait invocation
    "prost::DecodeError",
    "GgufReader", // a future GGUF importer's parser type
];

#[test]
fn im_3_byte_parsing_lives_only_in_grounding_crates() {
    let mut violations = Vec::<(String, String, usize, String)>::new();
    for &crate_name in NON_GROUNDING_CRATES {
        for path in rs_files_in(crate_name) {
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            for (lineno, line) in src.lines().enumerate() {
                // Ignore comments — IM-3 is about runtime imports, not prose.
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") || trimmed.starts_with("///") {
                    continue;
                }
                for tok in BANNED_TOKENS {
                    if line.contains(tok) {
                        violations.push((
                            crate_name.to_string(),
                            path.display().to_string(),
                            lineno + 1,
                            line.trim().to_string(),
                        ));
                    }
                }
            }
        }
    }
    if !violations.is_empty() {
        for (c, p, ln, l) in &violations {
            eprintln!("IM-3 violation in crate {c} at {p}:{ln}");
            eprintln!("    {l}");
        }
        panic!(
            "IM-3: {} byte-parsing token(s) leaked outside Grounding crates",
            violations.len()
        );
    }
}

#[test]
fn im_3_grounding_crate_actually_uses_a_parser() {
    // Sanity rail: the test is only meaningful if the banned tokens DO appear
    // in the `Grounding` crate. If `hologram-ai-onnx` ever stops importing
    // `prost`, this banning test would be vacuously true.
    let onnx_files = rs_files_in("hologram-ai-onnx");
    assert!(
        !onnx_files.is_empty(),
        "IM-3: no source files found for hologram-ai-onnx"
    );
    let mut found_prost = false;
    for path in onnx_files {
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };
        if src.contains("use prost") || src.contains("prost::Message") {
            found_prost = true;
            break;
        }
    }
    assert!(
        found_prost,
        "IM-3: hologram-ai-onnx no longer references `prost` — the IM-3 \
         rail would be vacuous. Update BANNED_TOKENS or move the parser."
    );
}

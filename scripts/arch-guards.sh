#!/bin/bash
# Architecture guards for the rewrite (ADR-0002, ADR-0009).
# Combines dependency inspection with source-level structure checks.
set -euo pipefail
cd "$(dirname "$0")/.."

fail() { echo "arch-guard: $*" >&2; exit 1; }

# 1. Forbidden dependencies in the resolved tree (tensor runtimes, GPU
#    backends, external inference providers).
TREE="$(cargo tree --workspace --all-features 2>/dev/null)" || fail "cargo tree failed"
for pat in onnxruntime "ort " candle tch-src torch cuda cudarc " metal" wgpu rocm opencl; do
    printf '%s' "$TREE" | grep -q "$pat" && fail "forbidden dependency pattern: $pat"
done

# 2. No public binary target anywhere in the workspace.
grep -RIn --include=Cargo.toml '^\[\[bin\]\]' crates/ && fail "public binary target found"

# 3. tless_store.bin must never be referenced as a bundle component.
#    Documentation and exclusion tests may name it; component-assembly code
#    may not. Only non-comment lines in library src/ count.
BAD_STORE="$(grep -RIn 'tless_store' crates/*/src --include='*.rs' \
    | grep -vE ':\s*//|//.*tless_store' \
    | grep -viE 'never|exclud|forbidden|guard' || true)"
[ -z "$BAD_STORE" ] || { printf '%s\n' "$BAD_STORE"; fail "tless_store reference in crate code"; }

# 4. uor-r4 crate types must not leak outside the adapter crate. (Our own
#    `hologram_ai_r4` crate, the `ModelLayer::uor_r4` constructor, and the
#    ENGINE_UOR_R4 id string are fine — they name no upstream type.)
LEAK="$(grep -RInE 'uor_r4_(api|core|graph|model|proof|router)' crates/ --include='*.rs' -l \
    | grep -v '^crates/hologram-ai-r4/' || true)"
[ -z "$LEAK" ] || fail "uor-r4 references outside hologram-ai-r4: $LEAK"

# 5. Networking/process execution only in the acquisition crate.
NET="$(grep -RInE 'std::net|reqwest|std::process::Command' crates/ --include='*.rs' -l \
    | grep -v '^crates/hologram-ai-huggingface/' || true)"
[ -z "$NET" ] || fail "network/process use outside hologram-ai-huggingface: $NET"

echo "arch-guards: green"

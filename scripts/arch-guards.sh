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
grep -RIn 'tless_store' crates/ --include='*.rs' \
    | grep -viE 'never|exclud|forbidden|guard' \
    && fail "tless_store reference in crate code"

# 4. uor-r4 types must not leak outside the adapter crate.
LEAK="$(grep -RIn 'uor_r4' crates/ --include='*.rs' -l | grep -v '^crates/hologram-ai-r4/' || true)"
[ -z "$LEAK" ] || fail "uor-r4 references outside hologram-ai-r4: $LEAK"

# 5. Networking/process execution only in the acquisition crate.
NET="$(grep -RInE 'std::net|reqwest|std::process::Command' crates/ --include='*.rs' -l \
    | grep -v '^crates/hologram-ai-huggingface/' || true)"
[ -z "$NET" ] || fail "network/process use outside hologram-ai-huggingface: $NET"

echo "arch-guards: green"

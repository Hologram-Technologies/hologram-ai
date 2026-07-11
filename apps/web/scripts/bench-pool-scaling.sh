#!/usr/bin/env bash
# Decode-GEMV thread-scaling benchmark for the wasm worker pool (ADR-0018).
#
# Runs the substrate's `wasm_threads_timing` harness under wasmtime
# (wasm32-wasip1-threads — std threads drive the SAME atomics fork-join queue the
# browser web workers do, so it is representative of browser scaling for the
# compute-bound decode GEMV). It times the int8 M=1 GEMV SERIAL vs POOLED
# (3 workers + main) at chat-scale shapes — Qwen2.5 0.5B / 1.5B / 7B MLP dims —
# with NO model download and NO OPFS limit, isolating the pool's speedup-vs-size.
#
# This is the honest counter to the ~1.0x on SmolLM2-135M: 135M's GEMV tiles are
# below the fork-join break-even; realistic chat models are not.
#
# Requires: a pinned nightly + the wasm32-wasip1-threads target + wasmtime
# (https://wasmtime.dev; set WASMTIME=/path/to/wasmtime).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NIGHTLY="${HAI_WASM_THREADS_NIGHTLY:-nightly-2026-07-09}"
# A timing bench is not the shipped artifact — if the pinned nightly is not
# installed, fall back to whatever `nightly` is present.
rustup run "$NIGHTLY" rustc --version >/dev/null 2>&1 || NIGHTLY="nightly"
WASMTIME="${WASMTIME:-$(command -v wasmtime 2>/dev/null || echo /tmp/wasmtime)}"
TARGET_DIR="${HAI_BENCH_TARGET_DIR:-/tmp/hai-pool-bench}"
[ -x "$WASMTIME" ] || { echo "wasmtime not found — install it or set WASMTIME=…" >&2; exit 1; }

# Locate the pinned substrate checkout from the RESOLVED commit in Cargo.lock
# (robust to a `rev =` or `tag =` pin).
REV="$(sed -n 's|.*hologram.git?[^#]*#\([0-9a-f]\{40\}\).*|\1|p' "$ROOT/Cargo.lock" | head -1)"
SHORT="${REV:0:7}"
SRC="$(ls -d /usr/local/cargo/git/checkouts/hologram-*/"$SHORT" 2>/dev/null | head -1)"
[ -n "$SRC" ] && [ -d "$SRC" ] || { echo "substrate checkout for rev $SHORT not found" >&2; exit 1; }

# The checkout is read-only; copy it so cargo can build in place (target → /tmp).
WORK="${TARGET_DIR}/src"
rm -rf "$WORK"; mkdir -p "$WORK"; cp -r "$SRC/." "$WORK/"; chmod -R u+w "$WORK"

# Inject our per-model metrics benchmark as a hologram-backend example (it drives
# the substrate's pub `matmul_i8_pc_omajor` + `wasm_pool`). Committed here, so the
# benchmark code is ours + reproducible; it builds against the pinned substrate.
cp "$ROOT/apps/web/scripts/pool-bench.rs" "$WORK/crates/hologram-backend/examples/pool_bench.rs"

echo "[bench] building pool_bench (rev $SHORT) for wasm32-wasip1-threads"
( cd "$WORK/crates/hologram-backend"
  CARGO_TARGET_DIR="$TARGET_DIR/target" RUSTFLAGS="-Ctarget-feature=+simd128" \
    rustup run "$NIGHTLY" cargo build --release \
    --example pool_bench --target wasm32-wasip1-threads \
    --no-default-features --features cpu,std,wasm-threads )

WASM="$TARGET_DIR/target/wasm32-wasip1-threads/release/examples/pool_bench.wasm"
POOL_WORKERS="${POOL_WORKERS:-3}"  # 3 workers + main = 4 participants (codespace's 4 phys cores)
echo "[bench] wasmtime run (POOL_WORKERS=$POOL_WORKERS; real chat-model configs)"
echo
POOL_WORKERS="$POOL_WORKERS" "$WASMTIME" run -W threads=y -S threads --env POOL_WORKERS="$POOL_WORKERS" "$WASM"

#!/usr/bin/env bash
# Build the MULTI-THREADED browser wasm — the substrate embedder worker pool
# over a shared linear memory (ADR-0018). This is the isolated-context artifact;
# `pnpm wasm` still builds the single-threaded `+simd128` fallback that ships
# alongside it and is loaded when the page is not cross-origin-isolated.
#
# Why this can't be `wasm-pack`: the threaded build needs nightly `-Z build-std`
# (to rebuild `std` with atomics) plus a precise set of linker exports, which
# wasm-pack does not drive. So it is a raw `cargo build` + the version-matched
# `wasm-bindgen` CLI — exactly the pipeline the ADR-0018 spike validated.
#
# The build differs from the single-threaded one in three ways:
#   1. `--features wasm-threads` → the no_std host-shell stack, so
#      `hologram-backend` compiles its `Atomics.wait` pool futex (not busy-spin).
#   2. `+atomics,+bulk-memory,+mutable-globals` + a shared, imported memory
#      (`--shared-memory --import-memory --max-memory=4GiB`).
#   3. The TLS/heap globals wasm-bindgen's thread transform needs, exported
#      explicitly (manual `--shared-memory` suppresses lld's default exports).
#
# The pool's host futex (`hologram_host_wait32`/`notify`) is satisfied natively
# by `hologram-ai-wasm::wasm_futex` (nightly wasm atomic intrinsics), so the
# artifact imports only `env.memory` — no JS futex shim needed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="$ROOT/apps/web/src/wasm-threads"
# Pin a nightly for reproducibility (gate == CI == deploy; see ADR-0018 and the
# dark-gates memory). Override with HAI_WASM_THREADS_NIGHTLY if it must advance.
NIGHTLY="${HAI_WASM_THREADS_NIGHTLY:-nightly-2026-07-09}"
TARGET_DIR="${HAI_WASM_THREADS_TARGET_DIR:-/tmp/hai-target-threads}"

MAX_MEMORY=$((4 * 1024 * 1024 * 1024)) # 4 GiB — the wasm32 address ceiling
export RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals,+simd128 \
  -C link-arg=--shared-memory -C link-arg=--max-memory=${MAX_MEMORY} -C link-arg=--import-memory \
  -C link-arg=--export=__heap_base -C link-arg=--export=__wasm_init_tls \
  -C link-arg=--export=__tls_base -C link-arg=--export=__tls_size -C link-arg=--export=__tls_align"

echo "[wasm-threads] cargo +${NIGHTLY} build-std (atomics + shared memory)"
CARGO_TARGET_DIR="$TARGET_DIR" rustup run "$NIGHTLY" cargo build \
  -p hologram-ai-wasm --target wasm32-unknown-unknown \
  --features wasm-threads -Z build-std=std,panic_abort --release

RAW="$TARGET_DIR/wasm32-unknown-unknown/release/hologram_ai_wasm.wasm"

# Resolve a wasm-bindgen CLI whose version matches the `wasm-bindgen` crate in
# Cargo.lock (a mismatch is a hard wasm-bindgen error). Prefer PATH, then the
# wasm-pack download cache, else instruct to install.
WANT="$(sed -n '/^name = "wasm-bindgen"$/{n;s/version = "\(.*\)"/\1/p;}' "$ROOT/Cargo.lock" | head -1)"
pick_wb() {
  local cand
  for cand in "$(command -v wasm-bindgen 2>/dev/null || true)" \
              "$HOME"/.cache/.wasm-pack/wasm-bindgen-*/wasm-bindgen; do
    [ -x "$cand" ] || continue
    if "$cand" --version 2>/dev/null | grep -q " ${WANT}\$"; then echo "$cand"; return 0; fi
  done
  return 1
}
WB="$(pick_wb || true)"
if [ -z "$WB" ]; then
  echo "[wasm-threads] wasm-bindgen ${WANT} not found; installing…"
  cargo install wasm-bindgen-cli --version "=${WANT}" --locked
  WB="$(command -v wasm-bindgen)"
fi

echo "[wasm-threads] wasm-bindgen ($("$WB" --version)) --target web -> $OUT"
rm -rf "$OUT"; mkdir -p "$OUT"
"$WB" "$RAW" --out-dir "$OUT" --target web

# Optimize, as wasm-pack does for the single-threaded build — otherwise the
# threaded wasm ships UNoptimised and could be SLOWER than the optimized
# single-threaded fallback (a shipping regression). The `--enable-*` flags keep
# wasm-opt from rejecting/stripping the shared-memory atomics + simd features.
BG="$OUT/hologram_ai_wasm_bg.wasm"
WASM_OPT="$(command -v wasm-opt 2>/dev/null || ls -t "$HOME"/.cache/.wasm-pack/wasm-opt-*/bin/wasm-opt 2>/dev/null | head -1 || true)"
if [ -n "$WASM_OPT" ]; then
  echo "[wasm-threads] wasm-opt -O (threads + simd preserved)"
  "$WASM_OPT" -O --enable-threads --enable-bulk-memory --enable-mutable-globals \
    --enable-simd "$BG" -o "$BG.opt"
  mv "$BG.opt" "$BG"
else
  echo "[wasm-threads] WARNING: wasm-opt not found — artifact ships UNoptimised"
fi

# Provenance: prove the artifact is actually threaded (a green build that
# silently produced a non-shared memory would be a dark gate — ADR-0018).
node -e '
const fs=require("fs");
const m=new WebAssembly.Module(fs.readFileSync(process.argv[1]));
const imp=WebAssembly.Module.imports(m), exp=WebAssembly.Module.exports(m);
const envFutex=imp.filter(i=>i.module==="env"&&/hologram_host/.test(i.name));
const pool=exp.filter(e=>/hologram_worker_run|hologram_pool_/.test(e.name)).map(e=>e.name);
if(pool.length!==3){console.error("FAIL: expected 3 pool exports, got",pool);process.exit(1);}
if(envFutex.length){console.error("FAIL: unexpected env futex imports",envFutex.map(i=>i.name));process.exit(1);}
console.log("[wasm-threads] verified: pool exports ["+pool.join(", ")+"], no env futex imports");
' "$OUT/hologram_ai_wasm_bg.wasm"

echo "[wasm-threads] done."

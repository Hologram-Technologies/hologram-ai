# hologram-ai-desktop

Tauri 2 desktop app for downloading, compiling, and chatting with ONNX models
through the `hologram-ai` CLI.

Three screens:

- **Chat** — pick a `.holo` archive, send a prompt, watch the response stream.
- **Models** — curated catalogue of known-working models; each row tracks
  download/compile state and exposes a single context-aware action button.
- **Logs** — live tail of in-process tracing + every line of CLI subprocess output.

## Architecture

The app is a thin shell over the `hologram-ai` CLI binary. It does **not** link
the runtime in-process — instead it spawns the CLI as a subprocess and streams
stdout/stderr to the UI as Tauri events.

In **release builds** the CLI ships as a Tauri sidecar (declared in
`tauri.conf.json` under `bundle.externalBin`) so the resulting `.dmg` is
self-contained: a user double-clicks and it works. In **dev builds** the app
falls back to `target/release/hologram-ai` or `target/debug/hologram-ai`.

## Prerequisites (dev)

- `pnpm`
- A built CLI under `target/`:
  ```
  cargo build --release -p hologram-ai
  ```
  Override the lookup with `HOLOGRAM_AI_BIN=/path/to/hologram-ai`.

## Run (dev)

```
cd apps/desktop
pnpm install
pnpm tauri dev
```

## Building a packaged `.app` / `.dmg` locally

```
# Stage the sidecar for the current host arch
TRIPLE=$(rustc -vV | sed -n 's|host: ||p')
cargo build --release -p hologram-ai
mkdir -p apps/desktop/src-tauri/binaries
cp target/release/hologram-ai \
   apps/desktop/src-tauri/binaries/hologram-ai-$TRIPLE

cd apps/desktop
pnpm install
pnpm tauri build
# Output: apps/desktop/src-tauri/target/release/bundle/{dmg,macos}/...
```

For a universal (arm64 + x86_64) build, run two `cargo build --target ...` and
stage both sidecars, then `pnpm tauri build -- --target universal-apple-darwin`.

## Releasing via GitHub

Pushing a tag matching `desktop-v*` (e.g. `desktop-v0.1.0`) triggers
`.github/workflows/release-desktop.yml`, which:

1. Builds `hologram-ai` for both macOS arches
2. Stages both as Tauri sidecars
3. Runs `tauri build --target universal-apple-darwin`
4. Publishes a draft GitHub Release with the `.dmg` and `.app.tar.gz`

To cut a release:

```
git tag desktop-v0.1.0
git push origin desktop-v0.1.0
```

The workflow also accepts manual `workflow_dispatch` with an optional `tag`
input (leave blank to build artifacts only, without publishing a release).

### Code signing (optional)

Without Apple Developer credentials the resulting `.dmg` is unsigned. macOS
Gatekeeper will block first launch — users right-click → **Open** to confirm,
or run:

```
xattr -dr com.apple.quarantine /Applications/hologram-ai-desktop.app
```

To enable signing + notarization, set these repo secrets and re-run the
workflow:

- `APPLE_CERTIFICATE` (base64 .p12)
- `APPLE_CERTIFICATE_PASSWORD`
- `APPLE_SIGNING_IDENTITY` (e.g. `Developer ID Application: Foo (TEAMID)`)
- `APPLE_ID`, `APPLE_PASSWORD` (app-specific), `APPLE_TEAM_ID`

## Conventions

- Models live under `<workspace>/models/<model-name>/` (matches the CLI default)
- Compiled archives live under `<workspace>/output/` or alongside the model
- Curated catalogue: `apps/desktop/src-tauri/src/known_models.rs`

## Future work

- Linux + Windows packaging (blocked on making `accelerate` feature conditional)
- WebGPU + WASM frontend (so the same UI runs in a browser tab)
- In-process inference once the runtime API stabilises
- Multi-modal screens (image-gen, audio-gen) once those pipelines are stable

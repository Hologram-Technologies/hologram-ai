# Development Guide — hologram-ai

## Prerequisites

- Rust (stable toolchain — see `rust-toolchain.toml` if present)
- `cargo clippy`, `cargo fmt`
- `protoc` (Protocol Buffers compiler) — required for ONNX protobuf parsing
- `holoarch` CLI — for architecture doc synchronization (`cargo install holoarch`)

---

## Building

```bash
cargo build
cargo build --release
```

For workspace-wide builds including all importer crates:

```bash
cargo build --workspace
```

The GGUF importer requires no external dependencies. The ONNX importer generates Rust bindings from `.proto` files during build; ensure `protoc` is in your PATH.

---

## Testing

```bash
cargo test
cargo test --workspace
```

Integration tests require model fixtures located in `tests/fixtures/`. To download test models:

```bash
cargo run --bin fetch-test-models
```

For validation against reference runtimes (llama.cpp, ONNX Runtime), set the `HOLOGRAM_AI_VALIDATE` environment variable:

```bash
HOLOGRAM_AI_VALIDATE=1 cargo test --test integration
```

---

## Linting and Formatting

```bash
cargo clippy -- -D warnings
cargo fmt --check
```

CI enforces both. Fix all warnings before opening a PR.

---

## Workflow

1. Create a branch from `main`.
2. Make changes; run tests and Clippy.
3. Open a PR with a clear description.
4. PR requires passing CI.

Before implementing significant functionality, sync architecture docs:

```bash
holoarch status
holoarch pull
```

Read `specs/docs/` for architectural context. Files under `specs/docs/` are read-only and managed by `holoarch pull`.

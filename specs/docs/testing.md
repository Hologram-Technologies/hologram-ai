# Testing Strategy — hologram-ai

## Unit Tests

Unit tests live in `#[cfg(test)]` modules in the same file as the code they test.

```rust
#[cfg(test)]
mod tests {
use super::*;

#[test]
fn it_works() { ... }
}
```

---

## Integration Tests

Integration tests live in `tests/`. Each file in `tests/` is a separate
test binary that can import the crate as an external user would.

### Fixture Setup

Test fixtures are organized under `tests/fixtures/`:

- `tests/fixtures/models/` — small GGUF and ONNX model files for import tests
- `tests/fixtures/graphs/` — serialized `AiGraph` snapshots for regression tests
- `tests/fixtures/expected/` — golden output tensors for numerical validation

The `tests/common/mod.rs` module provides shared test helpers:

```rust
use common::{load_test_model, compare_tensors, TestContext};
```

`TestContext` handles temporary directory creation, model loading, and cleanup.

---

## Test Conventions

- Test names are descriptive: `hologram-ai_<behavior>_<condition>`.
- Use `tempfile::tempdir()` for tests that need the filesystem.
- Do not write to shared state; tests must be order-independent.
- Each test covers one behavior.

### Project-Specific Conventions

- **Importer tests** validate that format-specific logic terminates at the importer boundary (ADR-0003). After import, tests verify the `AiGraph` structure without inspecting format-specific internals.
- **Quantization tests** verify dequantization correctness for each supported scheme (Q4_0, Q4_K_M, Q6_K). Compare outputs against reference implementations with tolerance ≤ 1e-5.
- **Numerical validation** uses llama.cpp as the reference backend for GGUF models. Test helper `compare_with_reference()` loads the same model in llama.cpp and compares forward pass outputs.
- **Graph structure tests** use snapshot testing via `insta` crate to catch unintended IR changes.
- **KV-cache tests** validate multi-turn inference by comparing intermediate cache states between turns.
- Mark slow tests with `#[ignore]` and run them explicitly in CI with `cargo test -- --ignored`.

---

## Running Tests

```bash
cargo test                   # all tests
cargo test --test <name>     # single integration test file
cargo test <pattern>         # filter by test name
```

# Release Process — hologram-ai

## Versioning

This project follows [Semantic Versioning](https://semver.org):
`MAJOR.MINOR.PATCH`.

- **MAJOR**: breaking API or behavior change.
- **MINOR**: new functionality, backward-compatible.
- **PATCH**: bug fixes, no API change.

During the MVP and early development phases (0.x.y), the API is considered unstable. Minor version bumps may contain breaking changes until 1.0.0 is reached.

---

## Release Checklist

- [ ] All tests pass (`cargo test --workspace`)
- [ ] No Clippy warnings (`cargo clippy -- -D warnings`)
- [ ] `CHANGELOG.md` updated (if applicable)
- [ ] Version bumped in `Cargo.toml`
- [ ] PR merged to `main`
- [ ] Release tag created: `v<version>`

Additional hologram-ai release steps:

- [ ] Verify GGUF import tests pass against reference models
- [ ] Run integration tests comparing outputs against llama.cpp reference
- [ ] Ensure `hologram` dependency version is compatible and pinned
- [ ] Update CLI help text if command interfaces changed
- [ ] Validate `.holo` archive format compatibility with previous releases

---

## Publishing

Releases are published via the following channels:

1. **Crates.io** — Library crates are published to crates.io in dependency order:
   - `hologram-ai-ir`
   - `hologram-ai-quant`
   - `hologram-ai-gguf`
   - `hologram-ai-onnx`
   - `hologram-ai-opt`
   - `hologram-ai-lower`
   - `hologram-ai-session`
   - `hologram-ai` (umbrella crate)

2. **Binary releases** — The `hologram-ai` CLI binary is built for supported platforms (x86_64-linux, aarch64-linux, x86_64-darwin, aarch64-darwin) and attached to GitHub releases.

3. **Release automation** — Tagged releases (`v*`) trigger CI workflows that run the full test suite, build release binaries, and publish to crates.io (requires `CARGO_REGISTRY_TOKEN` secret).

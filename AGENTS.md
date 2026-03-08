# AGENTS.md

This document provides guidance for automated agents operating in **`hologram-ai`**.

---

## Repository Purpose

`hologram-ai` is a **library** repository in the ecosystem.

Standards version: `2026.03`

---

## Repository Structure

```
specs/
  docs/         — project documentation
  adrs/         — architecture decision records
```

---

## Rules for Agents

1. Follow the architecture standards defined in the architecture repo
2. Do not modify files outside this repository unless explicitly instructed
3. Run `cargo clippy -- -D warnings` before committing Rust changes
4. Use a consistent naming prefix for all crate names

---

<!-- ARCHON:MANAGED:BEGIN -->
## Ecosystem Rules

These rules apply to all repositories in the Hologram ecosystem.

### Naming
- Use the `hologram-` prefix for all crate names (never `holo-`)
- Follow kebab-case for crate and repo names

### Code Quality
- Run `cargo clippy -- -D warnings` before committing Rust changes
- Run `cargo fmt --check` before committing Rust changes
- All public APIs must have documentation comments
- No `unwrap()` in library code — use proper error handling
- Use traits at API boundaries; use macros to eliminate boilerplate
- Functions with >3 parameters must use the builder pattern
- Use `thiserror` for library errors; `anyhow` only in binaries
- See ADR-0007 for the full set of Rust development standards

### Architecture
- Follow ADR decisions from `hologram-architecture`
- Declare contracts in `hologram.repo.yaml`
- Do not introduce cross-repo dependencies without an ADR

### Documentation
- Keep `specs/docs/architecture.md` up to date with structural changes
- Update `AGENTS.md` when adding new conventions or rules
<!-- ARCHON:MANAGED:END -->


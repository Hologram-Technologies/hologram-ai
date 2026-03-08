# Architecture Governance & Documentation Rules

This repository participates in a larger architecture ecosystem defined in the `hologram-architecture` project.

AI agents and contributors must treat architectural decisions recorded there as **authoritative design constraints**.

## Architecture Source of Truth

The canonical architecture decisions live in the `hologram-architecture` repository.

In particular:

- `specs/adrs/` contains **Architecture Decision Records (ADRs)** that define system-level decisions.
- `specs/projects/` contains project architecture plans.
- `specs/prompts/` contains implementation prompts used to scaffold repositories.
- `specs/docs/` contains durable documentation referenced by implementation repositories.

If a change in this repository conflicts with an accepted ADR, **the ADR takes precedence** unless it is explicitly updated.

Agents must not silently violate or bypass architecture decisions.

---

## Required Documentation Reading

Before performing significant work, agents MUST read relevant documentation from:

- `specs/docs/architecture.md` — system architecture overview
- `specs/docs/upstream-architecture.md` — hologram base crate architecture
- `specs/docs/development.md` — local development workflow
- `specs/docs/adrs/` — Architecture Decision Records

---

## Code Quality Rules

- **Zero clippy warnings.** Run `cargo clippy --workspace -- -D warnings` before committing. All warnings must be fixed, not suppressed with `#[allow(...)]` unless there is a documented reason.
- **Zero compiler warnings.** Unused imports, dead code, and similar warnings must be cleaned up.
- **Format with `cargo fmt --all`** before committing.
- **All tests must pass.** Run `cargo test --workspace` before committing.
- Use `just ci` to run the full CI pipeline locally (format check + clippy + tests).

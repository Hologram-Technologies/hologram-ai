# CLAUDE.md

This file provides context for Claude Code when working in **`hologram-ai`**.

## Project Overview

- **Name**: hologram-ai
- **Role**: library
- **Type**: rust-workspace
- **Standards Version**: 2026.03

## Architecture

This repository follows the ecosystem's architecture standards.
See `specs/docs/architecture.md` for project-specific architecture details.

## Development

- Run tests: `cargo test`
- Check lints: `cargo clippy -- -D warnings`
- Format code: `cargo fmt`

## Conventions

- Use a consistent naming prefix for all crate names
- Follow architecture decisions from the architecture repo
- Document significant decisions as ADRs in `specs/adrs/`

<!-- ARCHON:MANAGED:BEGIN -->
## Ecosystem Context

This repository is part of the **Hologram** ecosystem — a multi-repo project governed by shared architecture decisions.

### Key files
- `hologram.repo.yaml` — this repo's role, contracts, and standards version
- `AGENTS.md` — guidance for AI agents (includes ecosystem-wide rules)
- `specs/docs/architecture.md` — project-specific architecture documentation

### Standards
- Standards version is declared in `hologram.repo.yaml`
- Run `archon verify` to check conformance
- Run `archon sync` to pull latest managed content from the architecture repo

### Conventions
- Use `hologram-` prefix for crate names
- Follow ADR decisions (see `hologram-architecture/specs/adrs/`)
- Declare inter-repo dependencies as contracts in `hologram.repo.yaml`
<!-- ARCHON:MANAGED:END -->

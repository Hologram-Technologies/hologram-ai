# hologram-ai — Claude Code Instructions

## Architecture Context

This project is governed by the `hologram-architecture` repository.
Architecture decisions and planning docs are maintained there and synced
here via `holoarch`.

## Before Starting Work

1. Check if architecture docs are current:
```bash
holoarch status
```
2. Pull the latest if needed:
```bash
holoarch pull
```
3. Read `specs/docs/` before implementing significant functionality.

## Key Docs

| Topic | Path |
|-------|------|
| Architecture overview | `specs/docs/README.md` |
| Crate layout | `specs/docs/crate-layout.md` |
| Roadmap | `specs/docs/roadmap.md` |
| ADRs | `specs/docs/adrs/` |
| Implementation prompts | `specs/docs/prompts/` |
| Coding rules | `AGENTS.md` (root) |

<!-- HOLOARCH:MANAGED:BEGIN -->
## Relationship to hologram-architecture

This project is part of the Hologram ecosystem. Architecture decisions,
ADRs, and planning artifacts are maintained in `hologram-architecture`.

Before implementing significant functionality:

1. Read `specs/docs/architecture.md` and `specs/docs/upstream-architecture.md`.
2. Check `specs/docs/development.md` for the local development workflow.
3. Pull updated architecture docs with:
```bash
holoarch pull
```

## Important Commands

```bash
holoarch check       # validate repository conformance
holoarch pull        # pull latest docs + refresh managed sections
holoarch doc <name>  # generate a new doc template in specs/docs/
```

## Cross-Repository Isolation

Do NOT modify `hologram-architecture` or any sibling repository from this project.
Architecture flows one way: from the architecture repo into subprojects via
`holoarch pull`. Files under `specs/docs/` are read-only and will be overwritten.

If you need changes in another repository, write a prompt or spec describing the
required change and save it to `specs/plans/`. Never make the change directly.

_This section is managed by `holoarch pull`. Repo: hologram-ai_
<!-- HOLOARCH:MANAGED:END -->

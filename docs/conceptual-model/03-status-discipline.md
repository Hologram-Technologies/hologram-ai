# 03 — Status discipline

Every dictionary row carries exactly one status. A status is not a label of confidence;
it is a **contract about what the V&V suite is allowed to assert** about that row. The
honesty meta-gate enforces it mechanically (`just honesty`).

| Status | Meaning | What V&V may assert | Tier |
|---|---|---|---|
| `verified` | Established: an external oracle artifact reproduced, a live authority matched, or a substrate witness green. | A green, gating check that the implementation reproduces the sourced fact. | gating `suite` |
| `build` | A precisely-scoped construction on verified pieces. | That the construction satisfies its structural invariants and deterministic references — never that it is externally sourced. | gating `suite` |
| `open` | A genuine unknown. | Only *measurements*. The claim is reported, never asserted true. | non-gating `target` |

This is the [UOR-Atlas-UTQC] vocabulary (`some-true`/`build`/`open`) mapped to this
domain: `verified` plays `some-true` (the sourced fact), `build` and `open` are
unchanged.

## Tiers

- `suite` — implemented, gating, green. The feature's scenarios run in CI and fail the
  build on regression.
- `target` — defined behavior, not yet established. The feature exists (the definition
  is real), its probe runs non-gating, and its results are reported. A `target` row is
  the only permitted representation of unfinished work — never a TODO in source, never a
  skipped step in a gating suite.

## Forbidden assertions

The honesty gate fails CI if:

- a dictionary row has no feature file, or a feature file maps to no row (orphans);
- a gating suite contains a skipped, pending, or undefined step;
- a feature asserts, as established, a claim whose row is `open` — e.g. **universal
  architecture coverage** ("arbitrary" means *parametric over the family registry and
  the model's own configuration*, not "every architecture on the Hub"), or asymptotic
  scaling claims beyond the measured `decode-elision` witness;
- a `build` row's scenarios cite an external authority as their basis (builds validate
  against invariants and committed references, not borrowed authority).

[UOR-Atlas-UTQC]: https://github.com/afflom/UOR-Atlas-UTQC

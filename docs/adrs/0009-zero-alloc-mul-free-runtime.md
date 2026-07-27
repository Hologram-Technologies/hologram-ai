# ADR-0009: Zero-allocation and multiplication-free runtime guarantees

Status: Accepted (rewrite baseline)
Date: 2026-07-24

## Context

uor-r4's deployed inference contract is: integer kernel only (no multiply,
divide, or float in the deployed path), CPU-only, and steady-state
allocation-free. hologram-ai wraps that engine; a wrapper that allocates or
reintroduces floats would silently void the contract.

## Decision

- Compilation, downloading, archive construction, initialization, and SDK
  marshaling may allocate. After model/session initialization, the core
  per-step inference path operates through caller-owned or preallocated
  buffers and performs **zero heap allocations**.
- The allocation-free low-level APIs (`predict_next_into`,
  `generate_into`) are the same engine path used by the allocating
  convenience methods (Rust strings, Python/TS values) at the outer
  boundary.
- An allocation census (counting global allocator in tests) asserts zero
  allocations for `predict_next_into` and each steady-state
  `generate_into` step after initialization, and that caller buffers are
  reused without hidden growth.
- hologram-ai does not invent a competing arithmetic contract: it reuses
  uor-r4's normative contract surface (`inference_contract`,
  invariant-ownership matrix, format/runtime boundaries) and adds only
  wrapper-level checks (dependency audit, no float/mul in adapter decision
  code).
- Abstention and status decisions come from uor-r4's deterministic policy;
  no sampling, temperature, or floating-point probability enters the
  decision path.

## Consequences

- Performance and determinism claims are test-backed on every commit.
- Any future engine swap must satisfy the same census to merge.

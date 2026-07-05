@row:deterministic-compile @stage:S2 @status:build @executor:rust @lane:default
Feature: Deterministic k-form compilation
  Content addressing is the system's core premise: a model must have a stable
  κ. The parametric k-form compile therefore must be a pure function of its
  inputs — the same config.json plus the same tensor manifest must compile to a
  byte-identical archive on every call, so its κ (and thus κ-dedup) is stable
  across processes and platforms. Graph construction emits nodes in a fixed
  order and the topological schedule that fixes the archive's node layout is a
  pure function of that order — never of a HashMap iteration seed. The invariant
  holds for both the monolithic compile and the staged (windowed) partition,
  validated against the arbitrary handshake-tiny use-case (model/usecases.toml).

  Scenario: the monolithic compile is byte-identical across repeated calls
    Given the handshake-tiny config and its Llama k-form manifest
    When the manifest is compiled to a monolithic k-form archive several times
    Then every monolithic archive is byte-identical, a single stable κ

  Scenario: the staged partition is byte-identical across repeated calls
    Given the handshake-tiny config and its Llama k-form manifest
    When the manifest is compiled to staged k-form archives several times
    Then every stage archive is byte-identical to its first compile, a single stable κ per stage

@row:total-algebraic-path @stage:S3 @status:open @executor:rust @lane:default @target
Feature: Total algebraic path, measured
  The normative posture (04-resource-model.md, Totality): every graph
  operation lowers to the quantum hierarchy; the float reference exists at
  gate time only; a runtime float escape forks semantics and reintroduces the
  cost structure the model eliminates. The measured present contradicts the
  posture: the pinned substrate dispatches float-dtype kernel calls to native
  IEEE-754 kernels first, so this repo's f32/bf16 decoder workloads run the
  float path today. The substrate is a read-only upstream dependency — this
  row holds the frontier as a NUMBER: the probe walks the fixture decoder's
  compiled graph and reports what fraction of its kernel work is float-dtyped
  (and would therefore take the substrate's runtime float dispatch). The row
  flips to build only when that fraction reaches zero and gate-time parity
  with the retired reference is witnessed per (op, tier).

  Scenario: the float-dispatch fraction of the compiled plan is measured
    Given the deterministic tiny decoder fixture with its weights in a κ-store
    When the fixture decoder's lowered kernel dtypes are tallied
    Then the float-dispatched fraction is reported, never asserted

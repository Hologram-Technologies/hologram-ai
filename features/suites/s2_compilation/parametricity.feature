@row:parametricity @stage:S2 @status:build @executor:rust @lane:default
Feature: Parametricity — an arbitrary instance runs the same pipeline
  No canonical-instance constant leaks into generic code: the arbitrary
  handshake-tiny use-case (model/usecases.toml) instantiates the identical
  pipeline end-to-end — parametric graph from its own config, weightless
  k-form compile with External κ parameters, materialization from a κ-store,
  and execution — with deterministic seeded weights standing in for a
  published checkpoint. The validation basis is determinism: two independent
  materialized sessions must execute to byte-identical output.

  Scenario: handshake-tiny runs build, compile, materialize, and execute
    Given the handshake-tiny use-case from the model registry
    And deterministic seeded weights for its manifest in a κ-store
    When the manifest is compiled without weights and materialized against the store
    Then the materialized session executes a forward pass
    And a second materialized session executes to byte-identical output

@row:streamed-weightless-compile @stage:S2 @status:build @executor:rust @lane:default
Feature: Streamed weightless compilation
  Compilation from the tensor manifest alone — names, shapes, dtypes, κ —
  emits a k-form .holo whose kappa_map binds every weight constant to its κ.
  No weight bytes are consumed at any point. The manifest is streamed from
  the live HuggingFace Hub metadata of a published model (the hf-hub
  authority supplies the inputs; the assertion is the structural invariant
  that the emitted κ-map is complete and duplicate-free).

  Scenario: a live manifest compiles to a k-form archive binding every weight
    Given the streamed metadata of "TinyLlama/TinyLlama-1.1B-Chat-v1.0" from the Hub
    When the manifest is compiled without weights
    Then the archive carries a kappa_map
    And the kappa_map names every manifest weight tensor exactly once

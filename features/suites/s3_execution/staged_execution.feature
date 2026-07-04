@row:staged-execution @stage:S3 @status:build @executor:rust @lane:default
Feature: Staged (windowed) execution over k
  Windowed execution over k equals monolithic execution: the parametric
  decoder is partitioned into stage archives — embedding, decoder-layer
  blocks, head — whose κ-maps cover the model's tensors exactly (a tied
  embedding κ is shared between the embedding and head stages: one κ-store
  blob, two stage bindings). The staged pipeline materializes, executes, and
  releases one stage at a time, reproduces the monolithic logits
  byte-for-byte (same kernels in the same per-layer order — the head-stage
  boundary carries the fused final-norm operands so no kernel differs), and
  bounds peak weight residency by the largest stage — the window — never the
  model. The classical whole-model residency assumption is thereby removed
  structurally, not managed by policy.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: the stage κ-maps partition the monolithic κ-map exactly
    When the fixture is compiled monolithically and as one-layer stages
    Then the union of the stage κ-maps equals the monolithic κ-map's tensor set
    And each weight κ appears in exactly the stages that consume it
    And a tied fixture shares the embedding κ between the embedding and head stages

  Scenario: staged execution reproduces the monolithic logits exactly
    When the same token window is executed monolithically and through the staged runner
    Then the staged logits are byte-identical to the monolithic logits

  Scenario: peak weight residency is bounded by the window, never the model
    When the same token window is executed monolithically and through the staged runner
    Then the peak resident weight bytes are at most the largest stage's weight bytes
    And the largest stage's weight bytes are strictly less than the model's total weight bytes

  Scenario: generation over the staged runner equals the monolithic completion
    When the same greedy completion is generated through the staged runner and the monolithic session
    Then both completions are identical and non-empty

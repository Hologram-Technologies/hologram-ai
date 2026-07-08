@row:stage-residency-cache @stage:S3 @status:build @executor:rust @lane:default
Feature: κ-store bandwidth is paid per window, never per token
  Strict one-stage windowing rematerializes every stage for every forward
  pass: a full model of κ-store traffic per generated token. That bound is a
  guarantee for models that cannot fit — not a tax on models that can. The
  residency budget is an environment MEASUREMENT (the structural heap
  ceiling minus what is already claimed, halved for the materialization
  transient — the download planner's own convention): materialized stage
  sessions stay resident across passes while they fit, so each stage's
  weights move once per window; a model past the headroom falls back to the
  strict window, never refused. Content-addressed elision then acts across
  tokens inside the retained sessions instead of dying with them.

  Background:
    Given the deterministic tiny decoder fixture with its weights in a κ-store

  Scenario: within the budget, each stage materializes once per window
    When a completion is generated with a residency budget that holds the whole model
    Then each stage materialized exactly once across the whole generation

  Scenario: a zero budget is exactly the strict one-stage window
    When a completion is generated with a zero residency budget
    Then every forward pass rematerialized every stage
    And the strict window's peak residency stays within one stage

  Scenario: the resident set never exceeds the budget
    When a completion is generated with a residency budget of two stages
    Then the peak resident weight bytes never exceed the budget or the single-stage floor

  Scenario: cached and strict execution produce the same completion
    When the same greedy completion is generated with and without a residency budget
    Then both completions are identical and non-empty

  Scenario: a hard address ceiling charges the true footprint, not the weight
    When a completion is generated under a hard address ceiling that holds the whole model's weights
    Then the address-ceiling run rematerializes more than the weight-cache run at the same budget
    And both ceiling and weight-cache completions are identical and non-empty

  Scenario: the resident set survives across chat turns
    When two completions are generated over one warm session within the budget
    Then the second completion adds no stage materializations

  Scenario: admission asks the environment with the model's own transient margin
    When a completion is generated under a margin-recording admission probe
    Then every admission carried the largest stage's transient bound as its margin
    And a probe that refuses admission yields strict windowing

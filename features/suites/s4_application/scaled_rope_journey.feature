@row:scaled-rope-journey @stage:S4 @status:build @executor:browser
Feature: A scaled-rope checkpoint runs the whole journey
  `rope_scaling` is an implemented frequency law, not a refusal: a checkpoint
  carrying llama3 scaling downloads, compiles, and STREAMS CHAT through the
  same journey as the plain fixture — the engine synthesizes the scaled tables
  at the realized positions (the law itself is pinned bit-level by
  `rope_scaling_reference.rs`; this scenario pins the browser wiring).

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: the llama3-scaled fixture downloads and chats
    When the llama3-scaled fixture model is downloaded
    And the user chats on the llama3-scaled fixture model
    Then the llama3-scaled turn streams a non-empty completion

@row:memory-guard @stage:S1 @status:build @executor:browser
Feature: The resource guard — pure projection, never refusal
  The guard surfaces the resource picture before transfer: κ-store need, the
  MEASURED local headroom (navigator.storage.estimate()), the resulting cache
  coverage, the execution window, and the stage plan. The κ-store is a cache
  over recorded provenance, so no resource figure refuses the journey.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: the projection is surfaced before transfer
    When the fixture model is downloaded
    Then the journey proceeds past the resource guard with figures surfaced

  Scenario: a model beyond local headroom is not refused
    When downloading a model whose κ-store need exceeds the measured local headroom
    Then the resource projection reports partial cache coverage
    And the journey is not refused at the guard

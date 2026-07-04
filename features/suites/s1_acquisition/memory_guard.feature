@row:memory-guard @stage:S1 @status:build @executor:browser
Feature: Parametric memory guard
  Before any transfer, a resource estimate derived from the model's own
  config.json and manifest sizes gates the journey against the environment
  budget. The estimate is a function of the model's parameters — never a
  hard-coded per-model constant.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: a model within budget proceeds
    When the fixture model is downloaded
    Then the journey proceeds past the memory guard

  Scenario: a model exceeding the environment budget is rejected before transfer
    When downloading a model whose config-derived estimate exceeds the environment budget
    Then the journey is rejected at the memory guard with the estimate
    And no shard bytes were transferred

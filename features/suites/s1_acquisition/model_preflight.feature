@row:model-preflight @stage:S1 @status:build @executor:browser
Feature: Model preflight — validate before any shard byte moves
  The downloader is intelligent: config.json must name a supported architecture
  family with its required keys, and the parametric graph must build from the
  config plus the header-only tensor manifest (ranged requests — kilobytes),
  BEFORE any weight shard is transferred. Rejection is loud and names the
  reason; a config that cannot produce a resource estimate is a preflight
  failure, never a silent pass.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: an unsupported architecture family is rejected before transfer
    When downloading a model whose config names an unsupported family
    Then the journey is rejected at preflight naming the family
    And no shard bytes were transferred for the rejected model

  Scenario: a malformed config is rejected before transfer
    When downloading a model whose config lacks the required keys
    Then the journey is rejected at preflight naming the missing key
    And no shard bytes were transferred for the rejected model

  Scenario: the fixture model passes preflight
    When the fixture model is downloaded
    Then the preflight validated the model before the first shard byte

@row:model-preflight @stage:S1 @status:build @executor:browser
Feature: Model preflight — validate before any shard byte moves
  The downloader is intelligent: config.json must supply the parametric decoder
  schema — either a recognized family, or an unrecognized architecture whose
  config and header-only tensor manifest (ranged requests — kilobytes) match
  the generic decoder recipe — and the parametric graph must build from that
  config plus the manifest BEFORE any weight shard is transferred. There is no
  name allowlist: an unknown architecture is DERIVED from its manifest, but a
  config outside the decoder schema (GPT-2's learned positions and Conv1D
  attention, say) is rejected on config alone, naming the architecture.
  Rejection is loud and names the reason; a config that cannot produce a
  resource estimate is a preflight failure, never a silent pass.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: an architecture outside the decoder schema is rejected before transfer
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

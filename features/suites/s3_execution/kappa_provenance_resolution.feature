@row:kappa-provenance-resolution @stage:S3 @status:build @executor:browser
Feature: κ-provenance resolution — the local store is a cache, not a mirror
  Every tensor's κ is recorded with its revision-pinned source (URL + byte
  range) at streaming time. A κ absent from the local OPFS cache resolves from
  that provenance and must re-hash to its κ — the same integrity check as
  local content — so the journey completes even with an EMPTY local cache.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: the handshake completes with an empty local cache
    Given a zero local cache budget
    When the fixture model is downloaded
    Then the local κ-store holds no fixture tensors
    And the model directory records κ provenance for every manifest tensor
    When the user sends handshake message 1
    Then assistant turn 1 streams a non-empty completion
    And the completion matches reference turn 1

  Scenario: a corrupted cache entry recovers through recorded provenance
    When the fixture model is downloaded
    And one cached fixture tensor is corrupted in the κ-store
    When the user sends handshake message 1
    Then assistant turn 1 streams a non-empty completion
    And the completion matches reference turn 1
    And the corrupted κ-store entry has evaporated

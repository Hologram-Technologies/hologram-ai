@row:memory-guard @stage:S1 @status:build @executor:browser
Feature: The resource guard — quota, never size
  The guard rejects only genuine resource shortfall: the κ-store bytes the
  model actually needs versus the MEASURED OPFS quota
  (navigator.storage.estimate()). Model size is never a rejection criterion —
  execution is windowed over k — and the projected window/storage figures are
  surfaced as information before transfer.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: a model within the storage quota proceeds with figures surfaced
    When the fixture model is downloaded
    Then the journey proceeds past the resource guard with figures surfaced

  Scenario: genuine storage shortfall is rejected before transfer naming both figures
    When downloading a model whose κ-store requirement exceeds the measured storage quota
    Then the journey is rejected naming the requirement and the quota
    And no shard bytes were transferred

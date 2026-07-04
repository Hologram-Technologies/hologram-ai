@row:streamed-download @stage:S1 @status:build @executor:browser
Feature: Streamed download into the OPFS κ-store
  The download worker streams safetensors shards tensor-by-tensor: each tensor
  is incrementally κ-hashed and persisted as `tensors/{κ}.bin`; peak transient
  memory is bounded by one tensor, never a shard.

  Background:
    Given the app is open in the browser against the hermetic model server

  Scenario: shards stream into content-addressed tensor blobs
    When the fixture model is downloaded
    Then every tensor in the fixture manifest is persisted under its κ in OPFS
    And each persisted blob re-hashes to its κ
    And identical tensors are stored once

  Scenario: download failures surface loudly
    When downloading a repository that does not exist
    Then the journey fails at the download stage naming the repository

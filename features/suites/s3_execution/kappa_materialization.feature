@row:kappa-materialization @stage:S3 @status:build @executor:rust @lane:default
Feature: κ-materialization
  A k-form archive plus a κ-store materialize into an executable session:
  every resolve is content-verified — the resolved bytes must re-hash to
  their κ, the same blake3 labeling the verified addressing rows anchor —
  and a missing or corrupt κ aborts with the label. The construction is
  validated against its deterministic reference: the same graph compiled
  from inline weights must execute byte-identically to the materialized
  k-form.

  Background:
    Given a matmul graph whose weight is available as bytes

  Scenario: materialized execution equals inline compilation byte-for-byte
    Given a κ-store holding the weight under its κ
    When the k-form archive is materialized and executed next to the inline archive
    Then the k-form archive declares exactly the weight's κ as its one requirement
    And both executions produce byte-identical non-trivial output

  Scenario: a missing κ aborts naming the label
    Given an empty κ-store directory
    When materialization of the k-form archive is attempted
    Then materialization fails naming the weight's κ

  Scenario: corrupt store content fails the integrity check
    Given a κ-store holding corrupt bytes under the weight's κ
    When materialization of the k-form archive is attempted
    Then materialization fails the integrity check naming the expected κ

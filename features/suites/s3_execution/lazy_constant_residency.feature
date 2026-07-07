@row:lazy-constant-residency @stage:S3 @status:build @executor:rust @lane:default
Feature: The weight tier pages against a residency budget
  Everything below the stage-archive tier already pages; the one tier the
  arena could not reach was constant residency — hologram pinned every model
  weight resident at load, so a model whose weight set exceeds the window
  would not fit however finely staged. The pager closes it: a k-form archive
  loads against a κ-store provider under a residency budget, its whole-κ
  weight constants carried as by_reference fingerprints that page in on first
  use and evict cold under the budget. The fingerprint IS the κ's content
  digest, so the slot's label and every derivation key equal the
  fully-resident path's — residency is orthogonal to identity. Verification
  stays at the trust boundary, once per κ per session. The arena is a bounded
  window over the provider, not a full copy.

  Scenario: a paged load bounds weight residency and stays bit-identical
    Given the staged decoder fixture compiled monolithically over a κ-store
    When it is loaded paged under a budget below its distinct weight set and decoded
    Then peak resident weight bytes stay under the budget and the logits are byte-identical to the fully-resident load

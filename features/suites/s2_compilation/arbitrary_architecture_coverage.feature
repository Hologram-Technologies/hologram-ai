@row:arbitrary-architecture-coverage @stage:S2 @status:open @executor:rust @lane:default @target
Feature: Architecture-family coverage of the parametric recipe, measured
  How much of the HuggingFace Hub's architecture space the parametric decoder
  recipe covers is a genuine unknown: it is measured here, never asserted
  universal. Coverage is not name-gating — an unrecognized architecture is
  DERIVED from its tensor manifest — so the honest question is faithfulness:
  given each family's OWN characteristic tensor layout, does the generic
  gated-SwiGLU decoder recipe represent it, or reject it loud? A family whose
  real tensors match the recipe (Llama, Mistral, Qwen2, Phi3) builds; a family
  the recipe cannot faithfully represent — per-head qk-norm (Qwen3), GeGLU
  (Gemma2), sparse MoE (Mixtral), Conv1D attention (GPT-2), a fused
  query_key_value block (GPT-NeoX), a bidirectional encoder (BERT) — is
  rejected rather than silently mis-built. The probe measures this frontier;
  correctness first, coverage second.

  Scenario: each family builds iff the recipe faithfully represents its layout
    Given a fixed list of common HuggingFace architecture families
    When each family is probed against the parametric registry
    Then the supported and unsupported counts are reported for every probed family

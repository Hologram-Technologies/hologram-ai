@row:bounded-embedding @stage:S3 @status:build @executor:rust @lane:default
Feature: Bounded by per-row/per-tile work, never whole-matrix F32
  A model's execution is bounded by the work it cannot avoid — the token rows
  the embedding selects, the tiles the matmul streams — never by widening a
  whole weight matrix to F32. The embedding is the confirmed case: casting the
  entire [vocab, hidden] table to F32 before selecting rows materializes a
  vocab·hidden·4 byte tensor (past the 32-bit wasm heap for a large-vocabulary
  model — the RuntimeError: unreachable trap). Row selection is dtype-agnostic,
  so the table stays at its native storage dtype, the gather yields native rows,
  and only the gathered [batch, seq, hidden] result is widened. The fused
  Phi3-family decoder (qkv_proj / gate_up_proj carved by compile-time Slice)
  executes monolithically and staged to the same logits, so the memory-bounding
  fixes never change the numbers.

  Scenario: a large-vocabulary embedding never materializes a whole-vocab F32 table
    Given a large-vocabulary decoder with a narrow-dtype embedding table
    When its embedding stage is compiled
    Then the embedding stage compiles with no whole [vocab, hidden] F32 tensor
    And the embedding table is gathered at its native dtype and only the gathered rows are widened

  Scenario: a fused Phi3-family model runs monolithic and staged to identical logits
    Given the tiny fused Phi3-family decoder fixture with its weights in a κ-store
    When the fused fixture is executed monolithically and through the staged runner
    Then the fused staged logits are byte-identical to the fused monolithic logits

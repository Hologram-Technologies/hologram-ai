@row:speculative-decode @stage:S3 @status:build @executor:rust @lane:default
Feature: Speculative decode batches a drafted continuation without changing the output
  A decode step is the substrate matmul's worst shape (M = 1): the
  `decode_shape` bench measures an M = K pass at a fraction of the wall-clock
  of K single steps. Speculative decode escapes M = 1 without a draft model.
  The next tokens are DRAFTED from the realized sequence's own recurrence — the
  tokens that followed the current suffix's most recent earlier occurrence, a
  zero-weight lookup — and VERIFIED in one M = K pass whose head emits logits at
  every drafted position. Only the longest prefix the model would ITSELF
  greedily produce is accepted; its K/V is spliced from that same pass and one
  correcting bonus token is committed. So the greedy completion is byte-identical
  to single-position decode — nothing unverified is ever emitted — while a
  recurring stretch commits several tokens per two passes, dropping the
  forward-pass count below the token count. No recurrence is a plain step, never
  worse. Validated against the plain step-decode oracle over the fixture, never a
  canonical constant.

  Scenario: speculative decode reproduces plain step decode byte for byte
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When the fixture is decoded once by plain steps and once by speculative decode
    Then both runs emit the identical tokens

  Scenario: a recurring stretch commits in fewer forward passes than tokens
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When a recurring fixture continuation is decoded by speculative decode
    Then it emits every token in fewer forward passes than tokens

@row:speculative-decode @stage:S3 @status:build @executor:rust @lane:default
Feature: Speculative decode batches a drafted continuation without changing the output
  A decode step is the substrate matmul's worst shape (M = 1): the
  `decode_shape` bench measures an M = K pass at a fraction of the wall-clock
  of K single steps. Speculative decode escapes M = 1 by DRAFTING the next
  tokens and VERIFYING them in one M = K pass whose head emits logits at every
  drafted position. The draft SOURCE is parametric — the verify/accept loop is
  drafter-agnostic — so either a zero-weight PROMPT-LOOKUP (the tokens that
  followed the current suffix's most recent earlier occurrence) or a small
  DRAFT MODEL (a second decode session proposing the continuation from its own
  cheaper forward) plugs into the same loop, changing only the acceptance rate,
  never the output. Only the longest prefix the model would ITSELF
  produce under the SAMPLER is accepted; its K/V is spliced from that same pass
  and one correcting bonus token is committed. The accept rule is the caller's
  own next-token rule — greedy argmax, or a per-ABSOLUTE-position sample — the
  same rule plain decode applies. Because that rule is a pure function of
  (logits, position), not of the decode PATH, the completion is byte-identical
  to single-position decode AT ANY TEMPERATURE, not only greedy: nothing
  unverified is ever emitted, and a sampled run reproduces plain sampled decode
  token-for-token given the same seed. A recurring stretch still commits several
  tokens per two passes, dropping the forward-pass count below the token count.
  No recurrence is a plain step, never worse. Speculation stays STRICTLY within
  the carried bucket: a verified batch splices its accepted K/V into fixed bucket
  rows, so the moment the next batch would reach the bucket the drafter retires —
  freeing the verify runner — and plain steps regrow the bucket, never a splice
  past its rows nor a verify plan co-resident with the wider bucket's build.
  Validated against the plain step-decode oracle over the fixture, never a
  canonical constant.

  Scenario: speculative decode reproduces plain step decode byte for byte
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When the fixture is decoded once by plain steps and once by speculative decode
    Then both runs emit the identical tokens

  Scenario: sampled speculative decode reproduces plain sampled decode byte for byte
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When the fixture is decoded once by plain sampled steps and once by speculative sampled decode at temperature 0.8
    Then both runs emit the identical tokens

  Scenario: a draft model reproduces plain decode byte for byte under partial acceptance
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When the fixture is decoded once plainly and once by speculative decode with a draft model at temperature 0.8
    Then both runs emit the identical tokens

  Scenario: a recurring stretch commits in fewer forward passes than tokens
    Given a decode session and a verify runner over the staged fixture with a bucket of 64 rows
    When a recurring fixture continuation is decoded by speculative decode
    Then it emits every token in fewer forward passes than tokens

  Scenario: speculative decode retires at a bucket boundary and reproduces plain decode
    Given a decode session and a verify runner over the staged fixture with a bucket of 8 rows
    When the fixture is decoded across a bucket boundary by plain steps and by speculative decode
    Then both runs emit the identical tokens

  Scenario: the verify head is parametric over any vocabulary
    When a large-vocabulary draft is verified by the whole head and the chunked staged head
    Then the chunked head reproduces the whole head at every position

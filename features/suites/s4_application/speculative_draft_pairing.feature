@row:speculative-draft-pairing @stage:S4 @status:build @executor:browser
Feature: A catalogue-paired draft model drives the browser drafter
  Speculative decode (row `speculative-decode`) is drafter-AGNOSTIC — the
  verify/accept/commit loop emits only the TARGET's own tokens, so a drafter
  changes the acceptance rate, never the output. The zero-weight prompt-lookup
  drafter ships by default; this row wires the OTHER drafter into the browser: a
  small paired DRAFT MODEL (`ModelDrafter`), the general throughput lever for
  novel text where prompt-lookup finds no recurrence.

  The pairing is DATA, never code (the anti-hardcode law): a catalogue entry
  names its `draftModel` (an hfId), and downloading the target downloads the
  paired draft the same parametric way — its own staged compile, κ-store, and
  quant tier. At generation the worker builds a SECOND decode session from the
  draft's dir and `attach_draft`s it to the target; the target's `generate` then
  drafts from the paired model instead of prompt-lookup. Both sessions are
  reused warm across turns (the draft session is reclaimed by `into_session`, no
  rebuild).

  Two invariants make the pairing safe rather than merely fast:

  - RESIDENCY. A paired draft is a SECOND growable session. Two residency
    budgets over the one wasm 4 GiB address space would over-commit (the
    `RuntimeError: unreachable` allocation abort). So the target and draft SHARE
    ONE residency ledger (`share_residency_with`) — admission charges their
    COMBINED footprint, extending the one-ledger law that already binds a decode
    turn's step/seeder/verify runners (row `lazy-constant-residency`) across the
    model pair. Neither model over-commits; a pair that does not fit windows,
    never crashes.
  - COMPATIBILITY. The draft consumes the TARGET's token ids (it never tokenizes
    text itself) and carries the target's realized sequence, so it must both
    cover the target's VOCABULARY (every target id indexes the draft's embedding)
    and match its CONTEXT (a shorter-context draft would abort its own forward the
    moment the sequence crossed its window). The pairing is admitted only when
    both hold; an incompatible or absent draft is REFUSED and the drafter falls
    back to prompt-lookup — the journey never dead-ends, and because the output is
    the target's regardless, correctness is unconditional.

  Byte-for-byte equality with plain decode under a real draft model is witnessed
  natively (row `speculative-decode`, "a draft model reproduces plain decode
  byte for byte under partial acceptance"); the browser witness is that the
  paired draft is downloaded, attached, and drives a live warm journey. The
  fixture is paired with itself (same tokenizer/vocab — a guaranteed-compatible
  draft) so the witness needs no second fixture model; a distinct draft is the
  identical wiring with a different dir.

  Background:
    Given the app is open in the browser against the hermetic model server
    And a forced single-layer execution window
    And speculative decode is enabled
    And the fixture model is paired with itself as its own draft
    When the fixture model is downloaded

  Scenario: the paired draft is downloaded with its target
    Then the paired draft model is present in local storage

  Scenario: speculative decode drafts from the paired model across a warm journey
    When the user sends handshake message 1
    And the user sends handshake message 2
    Then the drafter reports the paired draft model attached
    And the second turn reports a warm session
    And assistant turn 2 streams a non-empty completion

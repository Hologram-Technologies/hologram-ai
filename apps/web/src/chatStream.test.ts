// The chat generation store's defining properties (dictionary rows `warm-turn`,
// chat streaming): a generation in flight accumulates tokens INDEPENDENT of any
// React subscriber, so leaving the /chat route mid-generation (which unmounts
// `Chat` and removes its subscriber) no longer strands the bubble or loses the
// completed turn — and the WIRE carries DELTAS (each `chat://line` stdout event
// is one incremental chunk, per `StreamMessage` in generate.worker.ts) while
// the store's consumers keep reading the full cumulative text. Byte-identity is
// the law: the accumulated text equals the join of the deltas, always.
import { describe, expect, it, vi } from "vitest";

// The store subscribes to `chat://line` / `chat://status` at module load; mock
// `./ipc` so importing it needs no wasm/worker, and capture the callbacks so we
// can drive the stream deterministically. `vi.hoisted` shares the registry with
// the hoisted `vi.mock` factory.
const { handlers } = vi.hoisted(() => ({
  handlers: {} as Record<string, (l: { stream?: string; line: string }) => void>,
}));
vi.mock("./ipc", () => ({
  onProcessLine: (evt: string, cb: (l: { stream?: string; line: string }) => void) => {
    handlers[evt] = cb;
    return Promise.resolve(() => {});
  },
}));

import { chatStream } from "./chatStream";

describe("chatStream — route-independent generation store over a DELTA wire", () => {
  it("accumulates deltas in order with no subscriber (Chat unmounted) and the completed turn survives", () => {
    chatStream.begin("model-x");
    expect(chatStream.getSnapshot().generating).toBe(true);
    expect(chatStream.getSnapshot().archive).toBe("model-x");

    // No subscriber attached — the Chat route is "unmounted". Each event is a
    // DELTA; the store still accumulates (getSnapshot reads live state).
    const deltas = ["The", " quick", " brown", " 🦊"];
    handlers["chat://line"]({ stream: "stdout", line: deltas[0] });
    expect(chatStream.getSnapshot().text).toBe("The");
    handlers["chat://line"]({ stream: "stdout", line: deltas[1] });
    expect(chatStream.getSnapshot().text).toBe("The quick");

    // stderr lines are not part of the completion text.
    handlers["chat://line"]({ stream: "stderr", line: "a warning" });
    expect(chatStream.getSnapshot().text).toBe("The quick");

    // A subscriber attached MID-STREAM (Chat remounted) is notified and reads
    // the FULL accumulated text, not just the deltas that arrive after it —
    // the in-flight turn is not lost across the route change.
    let notified = 0;
    const unsub = chatStream.subscribe(() => {
      notified += 1;
    });
    expect(chatStream.getSnapshot().text).toBe("The quick");
    handlers["chat://line"]({ stream: "stdout", line: deltas[2] });
    handlers["chat://line"]({ stream: "stdout", line: deltas[3] });
    expect(chatStream.getSnapshot().text).toBe("The quick brown 🦊");

    chatStream.end();
    expect(notified).toBeGreaterThan(0); // end() notifies immediately
    expect(chatStream.getSnapshot().generating).toBe(false);
    // Byte-identity: the completed turn's text IS the join of the deltas.
    expect(chatStream.getSnapshot().text).toBe(deltas.join(""));
    unsub();
  });

  it("status narration shows before the first token and clears once a delta arrives; begin() clears the prior turn", () => {
    chatStream.begin("model-y");
    expect(chatStream.getSnapshot().text).toBe(""); // the prior turn's text is gone
    handlers["chat://status"]({ stream: "stdout", line: "compiling a 64-token window" });
    expect(chatStream.getSnapshot().status).toBe("compiling a 64-token window");
    expect(chatStream.getSnapshot().text).toBe("");
    // An EMPTY delta (the turn-opening line) accumulates nothing and does not
    // clear the narration.
    handlers["chat://line"]({ stream: "stdout", line: "" });
    expect(chatStream.getSnapshot().status).toBe("compiling a 64-token window");
    // The first non-empty delta clears the startup narration.
    handlers["chat://line"]({ stream: "stdout", line: "Hi" });
    expect(chatStream.getSnapshot().status).toBe("");
    expect(chatStream.getSnapshot().text).toBe("Hi");
    handlers["chat://line"]({ stream: "stdout", line: " there" });
    expect(chatStream.getSnapshot().text).toBe("Hi there");
    chatStream.end();
  });

  it("fails loud on a non-string delta (a stale full-string producer), never rendering 'undefined'", () => {
    chatStream.begin("model-z");
    expect(() =>
      handlers["chat://line"]({ stream: "stdout", line: undefined as unknown as string }),
    ).toThrow(/protocol mismatch/);
    expect(chatStream.getSnapshot().text).toBe("");
    chatStream.end();
  });
});

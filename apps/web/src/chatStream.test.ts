// The chat generation store's defining property (dictionary rows `warm-turn`,
// chat streaming): a generation in flight accumulates tokens INDEPENDENT of any
// React subscriber, so leaving the /chat route mid-generation (which unmounts
// `Chat` and removes its subscriber) no longer strands the bubble or loses the
// completed turn. This unit test drives the store with NO subscriber attached
// (the unmounted case) and asserts it still holds the stream, then a
// late-attached subscriber (the remounted case) reads the current snapshot.
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

describe("chatStream — route-independent generation store", () => {
  it("accumulates the stream with no subscriber (Chat unmounted) and the completed turn survives", () => {
    chatStream.begin("model-x");
    expect(chatStream.getSnapshot().generating).toBe(true);
    expect(chatStream.getSnapshot().archive).toBe("model-x");

    // No subscriber attached — the Chat route is "unmounted". Cumulative
    // SNAPSHOTS still land in the store (getSnapshot reads live state).
    handlers["chat://line"]({ stream: "stdout", line: "The" });
    expect(chatStream.getSnapshot().text).toBe("The");
    handlers["chat://line"]({ stream: "stdout", line: "The quick" });
    expect(chatStream.getSnapshot().text).toBe("The quick");

    // stderr lines are not part of the completion text.
    handlers["chat://line"]({ stream: "stderr", line: "a warning" });
    expect(chatStream.getSnapshot().text).toBe("The quick");

    // A subscriber attached LATER (Chat remounted) is notified and reads the
    // current snapshot — the in-flight turn is not lost across the route change.
    let notified = 0;
    const unsub = chatStream.subscribe(() => {
      notified += 1;
    });
    chatStream.end();
    expect(notified).toBeGreaterThan(0); // end() notifies immediately
    expect(chatStream.getSnapshot().generating).toBe(false);
    expect(chatStream.getSnapshot().text).toBe("The quick"); // the completed turn survived
    unsub();
  });

  it("status narration shows before the first token and clears once text arrives", () => {
    chatStream.begin("model-y");
    handlers["chat://status"]({ stream: "stdout", line: "compiling a 64-token window" });
    expect(chatStream.getSnapshot().status).toBe("compiling a 64-token window");
    expect(chatStream.getSnapshot().text).toBe("");
    // The first non-empty token clears the startup narration.
    handlers["chat://line"]({ stream: "stdout", line: "Hi" });
    expect(chatStream.getSnapshot().status).toBe("");
    expect(chatStream.getSnapshot().text).toBe("Hi");
    chatStream.end();
  });
});

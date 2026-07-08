// A module-level store for the IN-FLIGHT chat generation (token stream +
// startup narration), decoupled from any React route.
//
// The token stream and the log console share one Rust→JS channel (`chat://line`
// via ipc's `emitLine`), but they diverged in the UI: `addLog` captured every
// snapshot unconditionally (the Logs page updated live), while the assistant
// bubble only updated through a subscription that the routed `Chat` component
// set up on mount and tore down on unmount. So watching the Logs page — the
// natural way to watch a long generation — unmounted `Chat`, and its tokens
// reached the log but never the bubble; the completed turn was lost on return.
//
// This store subscribes to the stream ONCE at module load and never tears down,
// mirroring how the log buffer already captures it. It holds the current
// generation's cumulative text + status regardless of which route is mounted,
// so `Chat` renders the live bubble whenever it is on screen, and the completed
// turn is committed (and persisted) by the sender independent of mount state.

import { onProcessLine } from "./ipc";

export interface ChatStreamState {
  /** The archive whose generation is in flight (`null` when idle). */
  archive: string | null;
  /** Cumulative decoded text of the in-flight completion (a SNAPSHOT per event,
   * so consumers REPLACE, never append). */
  text: string;
  /** The startup narration (window compile / stage materialization) shown
   * before the first token, so a large model's honest startup is visible. */
  status: string;
  /** True between {@link begin} and {@link end}. */
  generating: boolean;
}

let state: ChatStreamState = { archive: null, text: "", status: "", generating: false };
const listeners = new Set<() => void>();
let notifyScheduled = false;

function notify() {
  listeners.forEach((l) => l());
}

// Coalesce the per-token notifications to one per animation frame: a fast model
// emits many snapshots a second, and re-rendering the whole transcript on each
// would jank. The state itself is always current (getSnapshot reads it live, so
// the final commit is never stale); only the RENDER cadence is throttled.
function scheduleNotify() {
  if (notifyScheduled) return;
  notifyScheduled = true;
  const raf =
    typeof requestAnimationFrame === "function"
      ? requestAnimationFrame
      : (f: () => void) => setTimeout(f, 16);
  raf(() => {
    notifyScheduled = false;
    notify();
  });
}

function set(next: Partial<ChatStreamState>, immediate: boolean) {
  // A fresh object identity each change, so `useSyncExternalStore` re-renders.
  state = { ...state, ...next };
  if (immediate) notify();
  else scheduleNotify();
}

// Subscribe ONCE, at module load — never torn down, so token deltas accumulate
// whichever route is mounted. `onProcessLine` resolves its unlisten
// asynchronously; we never unlisten (the store lives for the app's lifetime).
void onProcessLine("chat://line", (l) => {
  if (l.stream === "stderr") return;
  set({ text: l.line, ...(l.line ? { status: "" } : {}) }, false);
});
void onProcessLine("chat://status", (l) => {
  if (l.stream === "stderr") return;
  set({ status: l.line }, false);
});

export const chatStream = {
  /** Mark a new generation in flight for `archive`, clearing prior text.
   * Immediate (not frame-coalesced): the transition must not race a stale
   * trailing bubble. */
  begin(archive: string) {
    set({ archive, text: "", status: "", generating: true }, true);
  },
  /** Mark the in-flight generation finished (the text stays readable for the
   * sender's final commit; a later {@link begin} clears it). Immediate so the
   * in-flight bubble clears in the same tick the committed turn is rendered —
   * no duplicate-bubble flash. */
  end() {
    set({ generating: false }, true);
  },
  subscribe(cb: () => void): () => void {
    listeners.add(cb);
    return () => {
      listeners.delete(cb);
    };
  },
  getSnapshot(): ChatStreamState {
    return state;
  },
};

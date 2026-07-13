// Turning an opaque worker failure into a diagnosable message.
//
// A Web Worker that dies without reaching its own error handlers delivers a
// bare `ErrorEvent` to the spawning thread's `onerror`. Stringifying that event
// yields "[object Event]" — which is exactly what the deployed instance showed
// ("generate worker failed: [object Event]"), telling the user nothing. The two
// real causes are: the worker ran out of memory (the wasm 4 GiB address-space
// ceiling, or the tab's own budget) and was killed, or its module failed to
// load. This says which, from whatever detail the event carries.

/** The diagnostic detail of a worker `error` event — its message + location if
 * present, else a plain-language statement of what a detail-less death means. */
export function describeWorkerErrorEvent(ev: {
  message?: unknown;
  filename?: unknown;
  lineno?: unknown;
}): string {
  const message = typeof ev.message === "string" ? ev.message : "";
  if (message) {
    const at =
      typeof ev.filename === "string" && ev.filename
        ? ` (${ev.filename}${typeof ev.lineno === "number" ? `:${ev.lineno}` : ""})`
        : "";
    return `${message}${at}`;
  }
  // No message: a worker the browser terminated before it could report. In
  // this app that is overwhelmingly memory — a large model whose resident set
  // crossed the tab's wasm ceiling — or a failed module load.
  return "the worker died without an error message — most likely out of memory (a large model exceeding the browser tab's wasm memory), or its script failed to load";
}

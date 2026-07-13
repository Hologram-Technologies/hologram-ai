import { describe, it, expect } from "vitest";
import { describeWorkerErrorEvent } from "./workerError";

describe("describeWorkerErrorEvent", () => {
  it("never returns the opaque '[object Event]' — a detail-less death names memory as the likely cause", () => {
    // The deployed regression: a worker killed by the tab surfaces a bare
    // event whose `message` is empty. The old code stringified it to
    // "[object Event]". The diagnostic must name what that means.
    const detail = describeWorkerErrorEvent({});
    expect(detail).not.toContain("[object Event]");
    expect(detail).toMatch(/out of memory/i);
    expect(detail).toMatch(/script failed to load/i);
  });

  it("surfaces a real error message and its location verbatim", () => {
    const detail = describeWorkerErrorEvent({
      message: "RuntimeError: unreachable",
      filename: "generate.worker.ts",
      lineno: 42,
    });
    expect(detail).toBe("RuntimeError: unreachable (generate.worker.ts:42)");
  });

  it("surfaces a message without a location when the filename is absent", () => {
    expect(describeWorkerErrorEvent({ message: "boom" })).toBe("boom");
  });

  it("treats a non-string message as no message (the detail-less path)", () => {
    // A bare Event's `message` is often undefined, not "".
    const detail = describeWorkerErrorEvent({ message: undefined });
    expect(detail).toMatch(/out of memory/i);
  });
});

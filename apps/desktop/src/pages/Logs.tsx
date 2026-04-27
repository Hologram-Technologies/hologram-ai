import { useEffect, useRef, useState } from "react";
import { LogEntry, clearLogs, recentLogs } from "../ipc";

export function Logs() {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [autoscroll, setAutoscroll] = useState(true);
  const sinceRef = useRef(0);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    async function tick() {
      try {
        const res = await recentLogs(sinceRef.current);
        if (cancelled) return;
        if (res.entries.length > 0) {
          sinceRef.current = res.nextIndex;
          setEntries((prev) => {
            const merged = [...prev, ...res.entries];
            // Cap UI list to keep the DOM responsive.
            return merged.length > 4096 ? merged.slice(-4096) : merged;
          });
        }
      } catch (e) {
        console.error(e);
      }
    }
    tick();
    const handle = setInterval(tick, 500);
    return () => {
      cancelled = true;
      clearInterval(handle);
    };
  }, []);

  useEffect(() => {
    if (autoscroll) {
      bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight });
    }
  }, [entries, autoscroll]);

  async function onClear() {
    await clearLogs();
    sinceRef.current = 0;
    setEntries([]);
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Logs</h1>
        <div className="row">
          <label className="row" style={{ fontSize: 12, color: "var(--fg-dim)" }}>
            <input
              type="checkbox"
              checked={autoscroll}
              onChange={(e) => setAutoscroll(e.target.checked)}
            />
            autoscroll
          </label>
          <button onClick={onClear}>Clear</button>
        </div>
      </div>
      <div className="page-body" ref={bodyRef}>
        {entries.length === 0 ? (
          <div className="empty">No log entries yet.</div>
        ) : (
          <div className="logs">
            {entries.map((e, i) => (
              <div key={i} className={`log-entry ${e.level}`}>
                <span className="ts">
                  {new Date(e.timestampMs).toLocaleTimeString(undefined, {
                    hour12: false,
                  })}
                </span>
                <span className="lvl">{e.level}</span>
                <span>{e.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

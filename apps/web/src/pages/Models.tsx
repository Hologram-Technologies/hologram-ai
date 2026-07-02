import { useEffect, useState } from "react";
import {
  KnownModelStatus,
  WorkspacePaths,
  compileKnownModel,
  downloadKnownModel,
  listKnownModels,
  addCustomModel,
  onProcessLine,
  workspacePaths,
} from "../ipc";

type Busy = { id: string; phase: "downloading" | "compiling" } | null;

export function Models() {
  const [paths, setPaths] = useState<WorkspacePaths | null>(null);
  const [models, setModels] = useState<KnownModelStatus[]>([]);
  const [busy, setBusy] = useState<Busy>(null);
  const [tail, setTail] = useState<string[]>([]);



  async function refresh() {
    const [p, m] = await Promise.all([workspacePaths(), listKnownModels()]);
    setPaths(p);
    setModels(m);
  }

  useEffect(() => {
    refresh().catch(console.error);
    const subs = [
      onProcessLine("models://download-line", (l) =>
        setTail((t) => [...t.slice(-200), l.line]),
      ),
      onProcessLine("models://compile-line", (l) =>
        setTail((t) => [...t.slice(-200), l.line]),
      ),
    ];
    return () => {
      subs.forEach((p) => p.then((un) => un()));
    };
  }, []);

  async function onDownload(id: string) {
    setBusy({ id, phase: "downloading" });
    setTail([]);
    try {
      await downloadKnownModel(id);
    } catch (e) {
      setTail((t) => [...t, `error: ${String(e)}`]);
    } finally {
      setBusy(null);
      refresh().catch(console.error);
    }
  }

  async function onCompile(id: string) {
    setBusy({ id, phase: "compiling" });
    setTail([]);
    try {
      await compileKnownModel(id);
    } catch (e) {
      setTail((t) => [...t, `error: ${String(e)}`]);
    } finally {
      setBusy(null);
      refresh().catch(console.error);
    }
  }

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  async function onSearch() {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    setTail([]);
    try {
      // Query HuggingFace catalog for ONNX models matching the query
      const url = `https://huggingface.co/api/models?search=${encodeURIComponent(searchQuery.trim())}&sort=downloads&direction=-1&limit=10`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
      const data = await res.json();
      setSearchResults(data);
    } catch (e) {
      setTail((t) => [...t, `error: ${String(e)}`]);
    } finally {
      setIsSearching(false);
    }
  }

  async function onAddAndDownload(hfId: string) {
    try {
      await addCustomModel(hfId);
      await refresh();
      // Use the local ID (which is the trailing part of hfId) to download
      const id = hfId.split("/").pop()?.toLowerCase() || hfId.toLowerCase();
      await onDownload(id);
    } catch (e) {
      setTail((t) => [...t, `error: ${String(e)}`]);
    }
  }

  function statusLabel(m: KnownModelStatus): string {
    if (m.compiledArchive) return "Ready";
    if (m.downloaded) return "Downloaded";
    return "Not downloaded";
  }

  function actionFor(m: KnownModelStatus) {
    const isBusy = busy !== null;
    const meBusy = busy?.id === m.id;
    if (m.compiledArchive) {
      return (
        <button onClick={() => onCompile(m.id)} disabled={isBusy}>
          {meBusy && busy?.phase === "compiling" ? "Recompiling…" : "Recompile"}
        </button>
      );
    }
    if (m.downloaded) {
      return (
        <button onClick={() => onCompile(m.id)} disabled={isBusy}>
          {meBusy && busy?.phase === "compiling" ? "Compiling…" : `Compile (${m.quantize})`}
        </button>
      );
    }
    return (
      <button onClick={() => onDownload(m.id)} disabled={isBusy}>
        {meBusy && busy?.phase === "downloading" ? "Downloading…" : "Download"}
      </button>
    );
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1>Models</h1>
        <button onClick={() => refresh()} disabled={busy !== null}>
          Refresh
        </button>
      </div>
      <div className="page-body">
        <p style={{ color: "var(--fg-dim)", marginTop: 0, fontSize: 13 }}>
          Discover and download models via the HuggingFace Catalog API.
          Models are stored in <code>{paths?.modelsDir ?? "models/"}</code> and compiled to a{" "}
          <code>.holo</code> archive for WebAssembly execution.
        </p>

        <div style={{ marginBottom: 16, display: "flex", gap: 8 }}>
          <input
            type="text"
            placeholder="Search HuggingFace (e.g. llama, qwen, phi)"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSearch()}
            style={{ flex: 1, padding: "6px 8px" }}
          />
          <button onClick={onSearch} disabled={!searchQuery.trim() || isSearching}>
            {isSearching ? "Searching..." : "Search Catalog"}
          </button>
        </div>

        {searchResults.length > 0 && (
          <div className="list" style={{ marginBottom: 32, border: "1px dashed var(--border)", background: "rgba(0,0,0,0.1)" }}>
            <div style={{ padding: "8px 12px", borderBottom: "1px solid var(--border)", fontSize: 12, fontWeight: "bold", color: "var(--fg-dim)" }}>
              Search Results (Top 10)
            </div>
            {searchResults.map((r) => (
              <div className="list-item" key={r.id} style={{ alignItems: "center" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <strong>{r.id}</strong>
                  <div className="meta">Downloads: {r.downloads} · Tags: {(r.tags || []).slice(0, 5).join(", ")}</div>
                </div>
                <button onClick={() => onAddAndDownload(r.id)} disabled={busy !== null}>
                  Add & Download
                </button>
              </div>
            ))}
          </div>
        )}

        <h2 style={{ fontSize: 16, marginTop: 32, marginBottom: 16 }}>My Models</h2>
        <div className="list">
          {models.map((m) => (
            <div
              className="list-item"
              key={m.id}
              style={{ alignItems: "flex-start" }}
            >
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
                  <strong>{m.displayName}</strong>
                  <span className="meta">
                    {m.size} · {m.modality}
                  </span>
                </div>
                <div className="meta">{m.description}</div>
                <div className="meta">
                  HF: <code>{m.hfId}</code> · ~{m.approxArchiveMb} MB archive ·{" "}
                  <span
                    style={{
                      color: m.compiledArchive
                        ? "var(--accent)"
                        : "var(--fg-dim)",
                    }}
                  >
                    {statusLabel(m)}
                  </span>
                </div>
              </div>
              <div>{actionFor(m)}</div>
            </div>
          ))}
        </div>

        {tail.length > 0 && (
          <section style={{ marginTop: 24 }}>
            <h3 style={{ fontSize: 13, color: "var(--fg-dim)" }}>Output</h3>
            <pre
              style={{
                background: "var(--bg-elev)",
                border: "1px solid var(--border)",
                borderRadius: 6,
                padding: 12,
                fontSize: 12,
                maxHeight: 240,
                overflow: "auto",
              }}
            >
              {tail.join("\n")}
            </pre>
          </section>
        )}
      </div>
    </div>
  );
}

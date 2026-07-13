import { useEffect, useRef, useState } from "react";
import {
  KnownModelStatus,
  WorkspacePaths,
  compileKnownModel,
  downloadKnownModel,
  listKnownModels,
  addCustomModel,
  onProcessLine,
  workspacePaths,
  extensionPresent,
  hfBase,
} from "../ipc";
import { validateModelConfig } from "../holo";
import {
  PROBE_CONCURRENCY,
  ProbeCache,
  type ProbeState,
  lacksSafetensorsExport,
  mapBounded,
  probeWithCache,
} from "../searchPreflight";

type Busy = { id: string; phase: "downloading" | "compiling" } | null;

/** One search row: the listing's metadata plus its derivability probe state
 * (row `supported-search`). Rows render immediately as `probing` and upgrade
 * in place — a probe never blocks the list. */
type SearchResult = {
  id: string;
  downloads: number;
  tags?: string[];
  probe: ProbeState;
};

// Probe verdicts are per repo+revision, session-scoped (Map + sessionStorage
// mirror) — a re-search re-fetches nothing. Module-level: survives remounts.
const probeCache = new ProbeCache();

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
      onProcessLine("models://download-progress", (l) =>
        setTail((t) => {
          const newTail = [...t];
          if (newTail.length > 0 && newTail[newTail.length - 1].startsWith("Downloading ")) {
            newTail[newTail.length - 1] = l.line;
          } else {
            newTail.push(l.line);
          }
          return newTail.slice(-200);
        }),
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
    setTail([]);
    try {
      setBusy({ id, phase: "downloading" });
      await downloadKnownModel(id);
      
      setBusy({ id, phase: "compiling" });
      await compileKnownModel(id);
    } catch (e) {
      setTail((t) => [...t, `error: ${String(e)}`]);
    } finally {
      setBusy(null);
      refresh().catch(console.error);
    }
  }



  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  // A newer search supersedes the previous one's in-flight probes: their late
  // outcomes must not be written onto the new result rows.
  const searchSeq = useRef(0);

  async function onSearch() {
    if (!searchQuery.trim()) return;
    const seq = ++searchSeq.current;
    setIsSearching(true);
    setTail([]);
    try {
      const q = encodeURIComponent(searchQuery.trim());
      // ONE listing request: `full=true` includes each repo's file list
      // (siblings), so the safetensors first pass costs no extra fetches.
      const res = await fetch(
        `${hfBase()}/api/models?search=${q}&sort=downloads&direction=-1&limit=20&full=true`,
      );
      if (!res.ok) throw new Error(`Search failed`);

      const data = await res.json();
      const unique = Array.from(new Map(data.map((item: any) => [item.id, item])).values()) as any[];
      unique.sort((a, b) => b.downloads - a.downloads);

      // Derivability discovery (row `supported-search`): the ONLY authority is
      // the SAME config-only preflight the download runs (validateModelConfig →
      // parametric::validate_config) — no architecture-name allowlist anywhere.
      // First pass (structural, free): a repo whose listing PROVES it has no
      // safetensors export can never start the journey; a listing without file
      // info stays in and the probe decides.
      const candidates = unique
        .filter((item: any) => !lacksSafetensorsExport(item.siblings))
        .slice(0, 15);

      // A model already downloaded locally passed this very preflight at
      // download time — selectable without re-probing.
      const local = new Set(
        models.filter((m) => m.downloaded || m.compiledArchive).map((m) => m.hfId),
      );

      // Render NOW; each row upgrades in place as its probe resolves.
      setSearchResults(
        candidates.map(
          (item: any): SearchResult => ({
            id: item.id,
            downloads: item.downloads,
            tags: item.tags,
            probe: local.has(item.id) ? { status: "derivable" } : { status: "probing" },
          }),
        ),
      );

      // Bounded, cached probes (per repo+revision; a hit costs no fetch). Each
      // config.json comes from the same resolve base the downloader uses. A
      // refusal is SHOWN with the preflight's reason — never silently hidden.
      const toProbe = candidates.filter((item: any) => !local.has(item.id));
      void mapBounded(toProbe, PROBE_CONCURRENCY, async (item: any) => {
        const outcome = await probeWithCache(
          item.id,
          item.sha,
          `${hfBase()}/${item.id}/resolve/main/config.json`,
          probeCache,
          validateModelConfig,
        );
        if (searchSeq.current !== seq) return; // superseded by a newer search
        setSearchResults((rs) => rs.map((r) => (r.id === item.id ? { ...r, probe: outcome } : r)));
      });
    } catch (e) {
      setTail((t) => [...t, `search error: ${String(e)}`]);
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
        <button disabled={true}>
          Ready
        </button>
      );
    }
    return (
      <button onClick={() => onDownload(m.id)} disabled={isBusy}>
        {meBusy ? (busy?.phase === "compiling" ? "Compiling…" : "Downloading…") : "Download"}
      </button>
    );
  }

  function renderModelItem(m: KnownModelStatus) {
    return (
      <div className="list-item" key={m.id} style={{ alignItems: "flex-start" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
            <strong>{m.displayName}</strong>
            <span className="meta">
              {m.size && m.size !== "?" ? `${m.size} · ` : ""}
              {m.modality}
            </span>
          </div>
          <div className="meta">{m.description}</div>
          <div className="meta">
            HF: <code>{m.hfId}</code> ·{" "}
            {m.approxArchiveMb > 0
              ? `~${m.approxArchiveMb} MB archive`
              : "archive size resolved on download"}{" "}
            ·{" "}
            <span style={{ color: m.compiledArchive ? "var(--accent)" : "var(--fg-dim)" }}>
              {statusLabel(m)}
            </span>
          </div>
        </div>
        <div>{actionFor(m)}</div>
      </div>
    );
  }

  // "My Models" = models the user actually has (added themselves, or downloaded/
  // compiled locally). Featured suggestions the user has NOT adopted stay in
  // their own section — clearing storage empties "My Models" instead of
  // re-showing the shipped catalogue as if it were the user's.
  const myModels = models.filter((m) => !m.featured || m.downloaded || m.compiledArchive);
  const featured = models.filter((m) => m.featured && !m.downloaded && !m.compiledArchive);

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

        {!extensionPresent() && (
          <div style={{ padding: "8px 12px", background: "var(--bg-hover)", borderLeft: "4px solid var(--accent)", marginBottom: 16, fontSize: 13, borderRadius: 4 }}>
            <strong>Extension (optional):</strong> public models download directly; to download gated models install the <a href={`${import.meta.env.BASE_URL}extension.zip`} download>Holospaces Egress Extension</a> (load unpacked in chrome://extensions).
          </div>
        )}

        {extensionPresent() && (
          <div style={{ padding: "8px 12px", background: "rgba(0, 200, 100, 0.1)", borderLeft: "4px solid rgb(0, 200, 100)", marginBottom: 16, fontSize: 13, borderRadius: 4 }}>
            <strong>Extension Active:</strong> Holospaces Egress Extension is loaded. You can download gated models.
          </div>
        )}



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
              Search Results
            </div>
            {searchResults.map((r) => (
              <div
                className="list-item"
                key={r.id}
                style={{ alignItems: "center", opacity: r.probe.status === "refused" ? 0.55 : 1 }}
              >
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <strong>{r.id}</strong>
                  <div className="meta">Downloads: {r.downloads} · Tags: {(r.tags || []).slice(0, 4).join(", ")}</div>
                  {r.probe.status === "probing" && (
                    <div className="meta">Preflight: checking derivability…</div>
                  )}
                  {r.probe.status === "refused" && (
                    // The preflight's refusal VERBATIM — an honest refusal is
                    // information, never hidden (row `supported-search`).
                    <div className="meta" style={{ color: "var(--fg-dim)" }}>
                      Not runnable: {r.probe.reason}
                    </div>
                  )}
                </div>
                {r.probe.status === "derivable" ? (
                  <button onClick={() => onAddAndDownload(r.id)} disabled={busy !== null}>
                    Add & Download
                  </button>
                ) : (
                  <button
                    disabled
                    title={r.probe.status === "refused" ? r.probe.reason : "preflight in progress"}
                  >
                    {r.probe.status === "probing" ? "Checking…" : "Not runnable"}
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        <h2 style={{ fontSize: 16, marginTop: 32, marginBottom: 16 }}>My Models</h2>
        {myModels.length === 0 ? (
          <p className="meta" style={{ marginTop: 0 }}>
            No models yet. Search HuggingFace above and add any supported model —
            it appears here once you download it. The app ships no default models.
          </p>
        ) : (
          <div className="list">{myModels.map(renderModelItem)}</div>
        )}

        {featured.length > 0 && (
          <>
            <h2 style={{ fontSize: 16, marginTop: 32, marginBottom: 8 }}>
              Featured — quick start
            </h2>
            <p className="meta" style={{ marginTop: 0, marginBottom: 16 }}>
              Curated starting points. These are suggestions, not stored on your
              device until you download one.
            </p>
            <div className="list">{featured.map(renderModelItem)}</div>
          </>
        )}

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

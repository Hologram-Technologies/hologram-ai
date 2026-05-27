import { useState } from "react";
import { describe, run, type ModelInfo, type Output } from "./holo.ts";

// Minimal playground over the real wasm pipeline (ADR-0017): load a compiled
// `.holo`, inspect its ports, and run a forward pass with zero/ones/N fill.
// This is the seam the reused desktop React components plug into; the full GUI
// grows on top as compile()/generate() land.
export function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [holo, setHolo] = useState<Uint8Array | null>(null);
  const [outputs, setOutputs] = useState<Output[] | null>(null);
  const [fill, setFill] = useState<string>("zeros");
  const [error, setError] = useState<string | null>(null);

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    setError(null);
    setOutputs(null);
    const file = e.target.files?.[0];
    if (!file) return;
    const bytes = new Uint8Array(await file.arrayBuffer());
    setHolo(bytes);
    try {
      setInfo(await describe(bytes));
    } catch (err) {
      setError(String(err));
      setInfo(null);
    }
  }

  async function onRun() {
    if (!holo) return;
    setError(null);
    const f = fill === "zeros" ? undefined : fill === "ones" ? 1 : Number(fill);
    try {
      setOutputs(await run(holo, [], f));
    } catch (err) {
      setError(String(err));
    }
  }

  return (
    <main style={{ fontFamily: "system-ui, sans-serif", maxWidth: 760, margin: "2rem auto", padding: "0 1rem" }}>
      <h1>hologram-ai · browser</h1>
      <p style={{ color: "#555" }}>
        The real compile/run pipeline, client-side via WebAssembly. Load a compiled <code>.holo</code> and run it.
      </p>

      <section>
        <input type="file" accept=".holo" onChange={onFile} />
      </section>

      {info && (
        <section>
          <h2>Ports</h2>
          <p>
            <strong>inputs:</strong>{" "}
            {info.inputs.map((p, i) => `[${i}] ${p.dtype_name}×${p.element_count}`).join(", ") || "none"}
            <br />
            <strong>outputs:</strong>{" "}
            {info.outputs.map((p, i) => `[${i}] ${p.dtype_name}×${p.element_count}`).join(", ") || "none"}
          </p>
          <label>
            fill{" "}
            <select value={fill} onChange={(e) => setFill(e.target.value)}>
              <option value="zeros">zeros</option>
              <option value="ones">ones</option>
              <option value="2">2</option>
            </select>
          </label>{" "}
          <button onClick={onRun}>Run forward pass</button>
        </section>
      )}

      {outputs && (
        <section>
          <h2>Outputs</h2>
          {outputs.map((o, i) => (
            <div key={i}>
              <code>
                output[{i}]: {o.dtype_name}×{o.element_count}
              </code>
              <pre style={{ background: "#f4f4f4", padding: "0.5rem", overflowX: "auto" }}>
                {o.decoded ? `[${o.values.slice(0, 64).join(", ")}${o.values.length > 64 ? ", …" : ""}]` : "(binary)"}
              </pre>
            </div>
          ))}
        </section>
      )}

      {error && <pre style={{ color: "#b00", whiteSpace: "pre-wrap" }}>{error}</pre>}
    </main>
  );
}

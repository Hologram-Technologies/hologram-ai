import { useEffect, useRef, useState } from "react";
import {
  CompiledArchive,
  cancelGeneration,
  generate,
  listCompiledArchives,
  onProcessLine,
} from "../ipc";

interface Message {
  role: "user" | "assistant";
  text: string;
}

export function Chat() {
  const [archives, setArchives] = useState<CompiledArchive[]>([]);
  const [archive, setArchive] = useState<string>("");
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(128);
  const [temperature, setTemperature] = useState(0.7);
  const [running, setRunning] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const streamingRef = useRef<string>("");

  useEffect(() => {
    listCompiledArchives().then((a) => {
      setArchives(a);
      if (a.length && !archive) setArchive(a[0].path);
    });
  }, []);

  // Subscribe to chat lines once.
  useEffect(() => {
    const unlisten = onProcessLine("chat://line", (l) => {
      // The CLI prints generated tokens to stdout (when streaming) and
      // diagnostics to stderr. For a v0 scaffold we treat every line as
      // assistant content; users can switch to the Logs tab for diagnostics.
      if (l.stream === "stderr") return;
      streamingRef.current += (streamingRef.current ? "\n" : "") + l.line;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant") {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", text: streamingRef.current };
          return updated;
        }
        return [...prev, { role: "assistant", text: streamingRef.current }];
      });
    });
    return () => {
      unlisten.then((un) => un());
    };
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function onSend() {
    if (!archive || !prompt.trim() || running) return;
    const userText = prompt.trim();
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setPrompt("");
    setRunning(true);
    streamingRef.current = "";
    try {
      await generate({
        archive,
        prompt: userText,
        maxTokens,
        temperature,
      });
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `error: ${String(e)}` },
      ]);
    } finally {
      setRunning(false);
    }
  }

  async function onCancel() {
    await cancelGeneration();
  }

  return (
    <div className="page chat">
      <div className="page-header">
        <h1>Chat</h1>
        <div className="row">
          <select
            value={archive}
            onChange={(e) => setArchive(e.target.value)}
            disabled={running}
            style={{ minWidth: 260 }}
          >
            {archives.length === 0 && <option value="">No compiled archives</option>}
            {archives.map((a) => (
              <option key={a.path} value={a.path}>
                {a.name}
              </option>
            ))}
          </select>
          <input
            type="number"
            value={maxTokens}
            min={1}
            max={4096}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ width: 80 }}
            disabled={running}
            title="max tokens"
          />
          <input
            type="number"
            value={temperature}
            step={0.1}
            min={0}
            max={2}
            onChange={(e) => setTemperature(Number(e.target.value))}
            style={{ width: 70 }}
            disabled={running}
            title="temperature"
          />
        </div>
      </div>

      <div className="transcript" ref={transcriptRef}>
        {messages.length === 0 ? (
          <div className="empty">Pick an archive and send a prompt.</div>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={`bubble ${m.role}`}>
              <div className="role">{m.role}</div>
              <div>{m.text || (running && i === messages.length - 1 ? "…" : "")}</div>
            </div>
          ))
        )}
      </div>

      <div className="composer">
        <textarea
          placeholder="Ask anything…"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              onSend();
            }
          }}
          disabled={running || !archive}
        />
        {running ? (
          <button onClick={onCancel}>Cancel</button>
        ) : (
          <button onClick={onSend} disabled={!archive || !prompt.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}

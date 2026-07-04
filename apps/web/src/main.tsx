import React, { useState, useEffect } from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Models } from "./pages/Models";
import { Chat } from "./pages/Chat";
import { Logs } from "./pages/Logs";
import { extensionPresent } from "./ipc";
import "katex/dist/katex.min.css";
import "./styles.css";

function Shell() {
  const [hasExtension, setHasExtension] = useState(extensionPresent());

  useEffect(() => {
    // In case the extension script runs slightly after React mounts
    if (hasExtension) return;
    const interval = setInterval(() => {
      if (extensionPresent()) {
        setHasExtension(true);
        clearInterval(interval);
      }
    }, 500);
    return () => clearInterval(interval);
  }, [hasExtension]);

  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">hologram chat</div>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/models">Models</NavLink>
        <NavLink to="/logs">Logs</NavLink>
      </nav>
      <main className="content">
        {!hasExtension && (
          <div style={{ background: "var(--bg-hover)", color: "var(--fg-dim)", padding: "8px", textAlign: "center", fontSize: 13 }}>
            Optional: the holospaces egress extension enables gated-model downloads.{" "}
            <a href={`${import.meta.env.BASE_URL}extension.zip`} download style={{ textDecoration: "underline" }}>
              extension.zip
            </a>
            {" "}(load unpacked in chrome://extensions)
          </div>
        )}
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/models" element={<Models />} />
          <Route path="/logs" element={<Logs />} />
        </Routes>
      </main>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <HashRouter>
      <Shell />
    </HashRouter>
  </React.StrictMode>,
);

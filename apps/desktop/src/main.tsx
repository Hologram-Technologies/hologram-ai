import React from "react";
import ReactDOM from "react-dom/client";
import { HashRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { Models } from "./pages/Models";
import { Chat } from "./pages/Chat";
import { Logs } from "./pages/Logs";
import "katex/dist/katex.min.css";
import "./styles.css";

function Shell() {
  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">hologram chat</div>
        <NavLink to="/chat">Chat</NavLink>
        <NavLink to="/models">Models</NavLink>
        <NavLink to="/logs">Logs</NavLink>
      </nav>
      <main className="content">
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

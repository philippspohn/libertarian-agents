import { useEffect, useState } from "react";
import { api } from "../api";

export default function FilesPanel({ env }: { env: string }) {
  const [path, setPath] = useState("");
  const [node, setNode] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.files(env, path).then(setNode).catch((e) => setError(String(e.message ?? e)));
  }, [env, path]);

  const parent = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";

  return (
    <div className="content">
      <div className="row" style={{ marginBottom: 10 }}>
        <button onClick={() => setPath("")} disabled={!path}>/</button>
        <button onClick={() => setPath(parent)} disabled={!path}>..</button>
        <span className="dim">/{path}</span>
        <div style={{ flex: 1 }} />
        <button onClick={() => api.files(env, path).then(setNode)}>refresh</button>
      </div>
      {error && <div style={{ color: "var(--red)" }}>{error}</div>}
      {node?.dir && (
        <div className="card tree">
          {node.entries.length === 0 && <div className="dim">empty</div>}
          {node.entries.map((e: any) => (
            <div key={e.path} className="f" onClick={() => setPath(e.path)}>
              {e.dir ? "▸ " : "  "}{e.name}{e.dir ? "/" : ""}
              {!e.dir && <span className="dim mono-sm"> {e.size} B</span>}
            </div>
          ))}
        </div>
      )}
      {node && !node.dir && (
        <div className="card">
          <div className="spread"><b>{node.path}</b><span className="dim mono-sm">{node.size} B</span></div>
          <pre className="out" style={{ maxHeight: "70vh" }}>{node.content}</pre>
        </div>
      )}
    </div>
  );
}

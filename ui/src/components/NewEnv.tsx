import { useState } from "react";
import { api } from "../api";

export default function NewEnv({ onClose, onCreated }: {
  onClose: () => void;
  onCreated: (name: string) => void | Promise<void>;
}) {
  const [name, setName] = useState("");
  const [sandbox, setSandbox] = useState("docker");
  const [image, setImage] = useState("python:3.12-slim");
  const [cap, setCap] = useState("");
  const [envFile, setEnvFile] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function create() {
    setBusy(true);
    setError(null);
    try {
      await api.createEnv(name, {
        sandbox,
        image,
        input_token_cap: cap ? Number(cap) : null,
      }, envFile);
      await onCreated(name);
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="card modal" onClick={(e) => e.stopPropagation()}>
        <div className="spread"><b>new environment</b><button onClick={onClose}>×</button></div>
        <div className="section-title">name</div>
        <input value={name} onChange={(e) => setName(e.target.value.toLowerCase())} placeholder="research-lab" autoFocus />
        <div className="grid2">
          <div><div className="section-title">sandbox</div><select value={sandbox} onChange={(e) => setSandbox(e.target.value)}>
            <option value="docker">docker (isolated)</option>
            <option value="macos">macOS (native GPU, write-confined)</option>
            <option value="local">local (unsafe host)</option>
          </select></div>
          <div><div className="section-title">container image</div><input value={image} disabled={sandbox !== "docker"} onChange={(e) => setImage(e.target.value)} /></div>
          <div><div className="section-title">environment input-token cap</div><input value={cap} placeholder="none" onChange={(e) => setCap(e.target.value.replace(/\D/g, ""))} /></div>
        </div>
        {sandbox === "macos" && <div className="dim mono-sm" style={{ marginTop: 8 }}>Trusted-code mode. Runs natively for Metal/MLX; agents may read host files and use the network, but Seatbelt confines writes and deletions to this environment.</div>}
        {sandbox === "local" && <div className="mono-sm" style={{ color: "var(--red)", marginTop: 8 }}>No isolation: shell commands can modify any host file your user account can access.</div>}
        <div className="section-title">environment .env</div>
        <div className="dim mono-sm" style={{ marginBottom: 6 }}>Optional variables available to every agent shell. Stored as plain text inside this trusted environment.</div>
        <textarea className="code-input" rows={6} value={envFile} onChange={(e) => setEnvFile(e.target.value)} placeholder={"API_BASE_URL=https://example.com\nPROJECT_MODE=research"} />
        {error && <div className="mono-sm" style={{ color: "var(--red)", marginTop: 8 }}>{error}</div>}
        <div className="row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
          <button onClick={onClose} disabled={busy}>cancel</button>
          <button onClick={create} disabled={!name || busy}>{busy ? "creating…" : "create environment"}</button>
        </div>
      </div>
    </div>
  );
}

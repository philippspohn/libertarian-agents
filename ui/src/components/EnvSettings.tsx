import { useEffect, useState } from "react";
import { api } from "../api";

export default function EnvSettings({ env, onChange, onDeleted }: { env: string; onChange: () => void; onDeleted: () => void }) {
  const [info, setInfo] = useState<any>(null);
  const [cap, setCap] = useState("");
  const [sandbox, setSandbox] = useState("docker");
  const [image, setImage] = useState("");
  const [secrets, setSecrets] = useState("");
  const [envFile, setEnvFile] = useState("");
  const [messageChars, setMessageChars] = useState("4000");
  const [notice, setNotice] = useState<string | null>(null);
  const [noticeError, setNoticeError] = useState(false);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  function load() {
    return api.env(env).then((d) => {
      setInfo(d);
      setCap(d.config.input_token_cap ? String(d.config.input_token_cap) : "");
      setSandbox(d.config.sandbox);
      setImage(d.config.image ?? "");
      setSecrets((d.config.secrets ?? []).join(", "));
      setEnvFile(d.env_file ?? "");
      setMessageChars(String(d.config.max_message_chars ?? 4000));
    });
  }

  useEffect(() => {
    load();
  }, [env]);

  if (!info) return <div className="content dim">loading…</div>;

  async function save() {
    setBusyAction("save");
    setNotice("saving…");
    setNoticeError(false);
    try {
      await api.patchEnv(env, {
        input_token_cap: cap ? Number(cap) : null,
        sandbox,
        image,
        secrets: secrets.split(",").map((x) => x.trim()).filter(Boolean),
        env_file: envFile,
        max_message_chars: Number(messageChars),
      });
      setNotice("saved");
      await load();
      onChange();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
      setNoticeError(true);
    } finally {
      setBusyAction(null);
    }
  }

  async function environmentAction(action: "start" | "stop") {
    setBusyAction(action);
    setNotice(`${action === "start" ? "starting" : "stopping"} environment…`);
    setNoticeError(false);
    try {
      const result = await api.envAction(env, action);
      const runners = result.started?.length
        ? ` Started runners: ${result.started.join(", ")}.`
        : action === "start" ? " No non-finished runners to start." : "";
      const blocked = result.blocked && Object.keys(result.blocked).length
        ? ` Blocked: ${Object.entries(result.blocked).map(([name, why]) => `${name}: ${why}`).join("; ")}.`
        : "";
      setNotice(`Environment ${action === "start" ? "started" : "stopped"}.${runners}${blocked}`);
      await load();
      onChange();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
      setNoticeError(true);
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <div className="content">
      <div className="card">
        <div className="spread"><b>{env}</b><div className="row">
          <button onClick={() => environmentAction("start")} disabled={busyAction !== null}>
            {busyAction === "start" ? "starting…" : "start environment"}
          </button>
          <button onClick={() => environmentAction("stop")} disabled={busyAction !== null}>
            {busyAction === "stop" ? "stopping…" : "stop environment"}
          </button>
        </div></div>
        <div className="dim mono-sm" style={{ marginTop: 6, lineHeight: 1.8 }}>
          <div>sandbox: {info.config.sandbox} ({info.sandbox_running ? "running" : "stopped"})</div>
          <div>image: {info.config.image}</div>
          <div>env root: {info.env_root}</div>
          <div>forwarded host variables: {info.config.secrets.join(", ") || "none"}</div>
          <div>agent goals and summary models: configured per runner</div>
          <div>web search: native hosted tool on each runner’s provider</div>
        </div>
      </div>
      <div className="card">
        <div className="spread"><b>environment budget</b><button onClick={save} disabled={busyAction !== null}>save</button></div>
        <div className="section-title">environment-wide input token cap</div>
        <input value={cap} placeholder="(none)" onChange={(e) => setCap(e.target.value.replace(/\D/g, ""))} />
        <div className="dim mono-sm" style={{ marginTop: 4 }}>Kill switch across all runners, on top of per-runner budgets.</div>
        {notice && <div className="mono-sm" style={{ color: noticeError ? "var(--red)" : "var(--green)", marginTop: 8 }}>{notice}</div>}
      </div>
      <div className="card">
        <div className="spread"><b>environment configuration</b><button onClick={save} disabled={busyAction !== null}>save</button></div>
        <div className="grid2">
          <div><div className="section-title">sandbox</div><select value={sandbox} onChange={(e) => setSandbox(e.target.value)}>
            <option value="docker">docker</option><option value="local">local (no isolation)</option>
          </select></div>
          <div><div className="section-title">container image</div><input value={image} onChange={(e) => setImage(e.target.value)} /></div>
          <div><div className="section-title">forward host variables</div><input value={secrets} placeholder="NAME, OTHER_NAME" onChange={(e) => setSecrets(e.target.value)} />
            <div className="dim mono-sm" style={{ marginTop: 4 }}>Names only. Their values come from the host process and override matching environment .env keys.</div>
          </div>
          <div><div className="section-title">max message characters</div><input value={messageChars} onChange={(e) => setMessageChars(e.target.value.replace(/\D/g, ""))} /></div>
        </div>
        <div className="dim mono-sm" style={{ marginTop: 8 }}>
          Web search is provider-native: OpenAI runners receive the Responses web_search tool; OpenRouter runners receive openrouter:web_search.
        </div>
        <div className="dim mono-sm" style={{ marginTop: 8 }}>Stop the environment before changing sandbox, image, or host-forwarded variables.</div>
      </div>
      <div className="card">
        <div className="spread"><b>environment .env</b><button onClick={save} disabled={busyAction !== null}>save</button></div>
        <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
          Plain-text environment-scoped variables available to every agent shell. All agents can read this file. Stop the environment before changing it; files and the existing container are preserved.
        </div>
        <textarea className="code-input" rows={10} value={envFile} onChange={(e) => setEnvFile(e.target.value)} placeholder={"API_BASE_URL=https://example.com\nPROJECT_MODE=research"} />
      </div>
      <div className="card">
        <b style={{ color: "var(--red)" }}>danger</b>
        <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>Deletes the environment directory, its board, and all profiles.</div>
        <button onClick={async () => {
          if (!confirm(`Delete environment "${env}" and all its files?`)) return;
          await api.deleteEnv(env);
          onDeleted();
        }}>delete environment</button>
      </div>
    </div>
  );
}

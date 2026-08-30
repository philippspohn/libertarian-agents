import { useEffect, useState } from "react";
import { api } from "../api";

export default function EnvSettings({ env, onChange, onDeleted }: { env: string; onChange: () => void; onDeleted: () => void }) {
  const [info, setInfo] = useState<any>(null);
  const [goal, setGoal] = useState("");
  const [cap, setCap] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    api.env(env).then((d) => {
      setInfo(d);
      setGoal(d.config.goal ?? "");
      setCap(d.config.input_token_cap ? String(d.config.input_token_cap) : "");
    });
  }, [env]);

  if (!info) return <div className="content dim">loading…</div>;

  async function save() {
    await api.patchEnv(env, { goal, input_token_cap: cap ? Number(cap) : null });
    setNotice("saved — GOAL.md updated in the shared folder");
    onChange();
  }

  return (
    <div className="content">
      <div className="card">
        <b>{env}</b>
        <div className="dim mono-sm" style={{ marginTop: 6, lineHeight: 1.8 }}>
          <div>sandbox: {info.config.sandbox} ({info.sandbox_running ? "running" : "stopped"})</div>
          <div>image: {info.config.image}</div>
          <div>env root: {info.env_root}</div>
          <div>forwarded secrets: {info.config.secrets.join(", ") || "none"}</div>
          <div>summary model: {info.config.summary_model}</div>
        </div>
      </div>
      <div className="card">
        <div className="spread"><b>project goal</b><button onClick={save}>save</button></div>
        <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
          Injected as the user turn for every agent and mirrored to <code>shared/GOAL.md</code>.
        </div>
        <textarea rows={6} value={goal} onChange={(e) => setGoal(e.target.value)} />
        <div className="section-title">environment-wide input token cap</div>
        <input value={cap} placeholder="(none)" onChange={(e) => setCap(e.target.value.replace(/\D/g, ""))} />
        <div className="dim mono-sm" style={{ marginTop: 4 }}>Kill switch across all runners, on top of per-runner budgets.</div>
        {notice && <div className="mono-sm" style={{ color: "var(--green)", marginTop: 8 }}>{notice}</div>}
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

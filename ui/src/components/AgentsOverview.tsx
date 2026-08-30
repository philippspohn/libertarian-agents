import { useEffect, useState } from "react";
import { Agent, api, cost, num } from "../api";

export default function AgentsOverview({ env, agents, onSelect, onCreate, onChange }: {
  env: string;
  agents: Agent[];
  onSelect: (profile: string) => void;
  onCreate: () => void;
  onChange: () => void;
}) {
  const [now, setNow] = useState(Date.now());
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  async function act(profile: string, action: string) {
    try {
      await api.action(env, profile, action);
      setError(null);
      onChange();
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  }

  return (
    <div className="content">
      <div className="spread" style={{ marginBottom: 12 }}>
        <div><b>agents</b><div className="dim mono-sm">All profiles, runner state, usage, and current status.</div></div>
        <button onClick={onCreate}>+ new agent</button>
      </div>
      {error && <div className="card mono-sm" style={{ color: "var(--red)" }}>{error}</div>}
      <div className="card table-wrap">
        <table className="agents-table">
          <thead><tr><th>agent</th><th>state</th><th>status</th><th>model</th><th>input</th><th>output</th><th>cost</th><th>inbox</th><th>controls</th></tr></thead>
          <tbody>
            {agents.map((a) => {
              const waiting = a.state === "waiting" ? Math.max(0, Math.floor((now - Date.parse(a.updated_at)) / 1000)) : null;
              return (
                <tr key={a.profile} className="clickable-row" onClick={() => onSelect(a.profile)}>
                  <td><span className={`dot s-${a.state}`} style={{ display: "inline-block", marginRight: 7 }} />{a.profile}</td>
                  <td>{a.state}{waiting !== null ? ` ${waiting}s` : ""}</td>
                  <td className="dim">{a.status || a.stop_reason || "—"}</td>
                  <td>{a.config.provider}/{a.config.model}</td>
                  <td>{num(a.usage.input_tokens)} / {num(a.config.budgets.input_tokens)}</td>
                  <td>{num(a.usage.output_tokens)} / {num(a.config.budgets.output_tokens)}</td>
                  <td>{cost(a.usage)}</td>
                  <td>{a.unread || "—"}</td>
                  <td><div className="row" onClick={(e) => e.stopPropagation()}>
                    <button onClick={() => act(a.profile, "start")} disabled={a.running}>start</button>
                    <button onClick={() => act(a.profile, "stop")} disabled={!a.running}>stop</button>
                    <button onClick={() => act(a.profile, "wake")}>wake</button>
                  </div></td>
                </tr>
              );
            })}
            {agents.length === 0 && <tr><td colSpan={9} className="dim">No agent profiles yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { Agent, api, cost, num } from "../api";

const BUDGET_INCREMENTS = [
  { tokens: 500_000, label: "+500k" },
  { tokens: 1_000_000, label: "+1M" },
  { tokens: 5_000_000, label: "+5M" },
];

export default function AgentsOverview({ env, agents, onSelect, onCreate, onChange }: {
  env: string;
  agents: Agent[];
  onSelect: (profile: string) => void;
  onCreate: () => void;
  onChange: () => void | Promise<void>;
}) {
  const [now, setNow] = useState(Date.now());
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [budgeting, setBudgeting] = useState<string | null>(null);
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

  async function addBudget(profile: string, amount: number) {
    const key = `${profile}:${amount}`;
    setBudgeting(key);
    try {
      const result = await api.addInputBudget(env, profile, amount);
      setError(null);
      setNotice(
        `${profile}: +${num(amount)} input tokens; new limit ${num(result.input_budget)}`
        + (result.resume_error ? ` (saved, but could not resume: ${result.resume_error})` : "")
      );
      await onChange();
    } catch (e: any) {
      setError(String(e.message ?? e));
    } finally {
      setBudgeting(null);
    }
  }

  const totalUsage = {
    input_tokens: agents.reduce((sum, a) => sum + a.usage.input_tokens, 0),
    cached_input_tokens: agents.reduce((sum, a) => sum + a.usage.cached_input_tokens, 0),
    cache_write_tokens: agents.reduce((sum, a) => sum + a.usage.cache_write_tokens, 0),
    output_tokens: agents.reduce((sum, a) => sum + a.usage.output_tokens, 0),
    reasoning_tokens: agents.reduce((sum, a) => sum + a.usage.reasoning_tokens, 0),
    cost_usd: agents.reduce((sum, a) => sum + a.usage.cost_usd, 0),
    cost_known: agents.every((a) => a.usage.cost_known),
  };
  const totalInputBudget = agents.reduce((sum, a) => sum + a.config.budgets.input_tokens, 0);
  const totalOutputBudget = agents.reduce((sum, a) => sum + a.config.budgets.output_tokens, 0);

  return (
    <div className="content">
      <div className="spread" style={{ marginBottom: 12 }}>
        <div><b>agents</b><div className="dim mono-sm">All profiles, runner state, usage, and current status.</div></div>
        <button onClick={onCreate}>+ new agent</button>
      </div>
      {error && <div className="card mono-sm" style={{ color: "var(--red)" }}>{error}</div>}
      {notice && <div className="card mono-sm" style={{ color: "var(--green)" }}>{notice}</div>}
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
                  <td>
                    <div>{num(a.usage.input_tokens)} / {num(a.config.budgets.input_tokens)}</div>
                    <div className="quick-budget" onClick={(e) => e.stopPropagation()}>
                      {BUDGET_INCREMENTS.map(({ tokens, label }) => (
                        <button key={tokens} disabled={budgeting !== null}
                          title={`Add ${num(tokens)} input tokens and continue ${a.profile}`}
                          onClick={() => addBudget(a.profile, tokens)}>{label}</button>
                      ))}
                    </div>
                  </td>
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
          {agents.length > 0 && (
            <tfoot><tr className="totals-row">
              <td><b>total</b></td><td>{agents.length} agents</td><td>—</td><td>—</td>
              <td>{num(totalUsage.input_tokens)} / {num(totalInputBudget)}</td>
              <td>{num(totalUsage.output_tokens)} / {num(totalOutputBudget)}</td>
              <td>{cost(totalUsage)}</td>
              <td>{agents.reduce((sum, a) => sum + a.unread, 0) || "—"}</td><td>—</td>
            </tr></tfoot>
          )}
        </table>
      </div>
    </div>
  );
}

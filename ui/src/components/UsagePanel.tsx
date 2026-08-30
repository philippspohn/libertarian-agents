import { useEffect, useState } from "react";
import { Agent, api, cost, num } from "../api";

export default function UsagePanel({ env, agents }: { env: string; agents: Agent[] }) {
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    const load = () => api.usage(env).then(setData).catch(() => {});
    load();
    const id = setInterval(load, 4000);
    return () => clearInterval(id);
  }, [env, agents.length]);

  if (!data) return <div className="content dim">loading…</div>;
  const t = data.total;
  return (
    <div className="content">
      <div className="card grid2">
        {[
          ["input tokens", num(t.input_tokens)],
          ["cached input", num(t.cached_input_tokens)],
          ["output tokens", num(t.output_tokens)],
          ["reasoning tokens", num(t.reasoning_tokens)],
          ["cost", cost(t)],
        ].map(([k, v]) => (
          <div key={k}>
            <div className="dim mono-sm">{k}</div>
            <div style={{ fontSize: 20 }}>{v}</div>
          </div>
        ))}
      </div>
      <div className="card">
        <b>per profile</b>
        <table style={{ marginTop: 8 }}>
          <thead><tr><th>profile</th><th>model</th><th>calls</th><th>input</th><th>cached</th><th>output</th><th>reasoning</th><th>cost</th></tr></thead>
          <tbody>
            {data.breakdown.map((r: any, i: number) => (
              <tr key={i}>
                <td>{r.profile}</td><td>{r.model}</td><td>{r.calls}</td>
                <td>{num(r.i)}</td><td>{num(r.ci)}</td><td>{num(r.o)}</td><td>{num(r.r)}</td>
                <td>{r.k ? `$${r.c.toFixed(4)}` : "pricing unavailable"}</td>
              </tr>
            ))}
            {data.breakdown.length === 0 && <tr><td colSpan={8} className="dim">no calls yet</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="dim mono-sm">
        OpenRouter-reported cost is authoritative. Other calls use <code>pricing.json</code> estimates
        when available; unknown pricing is shown explicitly. Budgets are enforced on tokens.
      </div>
    </div>
  );
}

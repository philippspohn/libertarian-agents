import { useCallback, useEffect, useRef, useState } from "react";
import { Agent, api, num } from "../api";

function Bar({ used, total }: { used: number; total: number }) {
  const pct = Math.min(100, (used / Math.max(1, total)) * 100);
  const cls = pct > 95 ? "over" : pct > 75 ? "warn" : "";
  return (
    <div className="bar" title={`${used} / ${total}`}>
      <div className={cls} style={{ width: `${pct}%` }} />
    </div>
  );
}

function renderEvent(ev: any) {
  switch (ev.kind) {
    case "tool_call":
      return `→ ${ev.tool} ${JSON.stringify(ev.arguments ?? {}).slice(0, 400)}`;
    case "tool_result":
      return `   ${String(ev.summary ?? "").slice(0, 1200)}`;
    case "usage":
      return `   +${ev.input_tokens} in / ${ev.output_tokens} out`;
    case "reasoning":
      return ev.text;
    case "message":
      return ev.text;
    case "compaction":
      return `── compacted (${ev.reason}) ──\n${ev.summary ?? ""}`;
    case "run_start":
      return `── run start: ${ev.reason} ──`;
    case "run_end":
      return `── run end: ${ev.state} (${ev.reason}) ──`;
    case "sleep":
      return `── sleeping${ev.seconds ? ` ${ev.seconds}s` : ""}: ${ev.status ?? ""} ──`;
    case "model_swap":
      return `── model swap ${ev.old} → ${ev.new}; context compacted ──`;
    default:
      return `${ev.kind}: ${JSON.stringify({ ...ev, ts: undefined, kind: undefined, seq: undefined })}`;
  }
}

export default function AgentPanel({ env, agent, onChange }: { env: string; agent: Agent; onChange: () => void }) {
  const [events, setEvents] = useState<any[]>([]);
  const [detail, setDetail] = useState<any>(null);
  const [view, setView] = useState<"history" | "config">("history");
  const [configText, setConfigText] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const seen = useRef(-1);
  const scroller = useRef<HTMLDivElement>(null);

  const profile = agent.profile;

  const loadDetail = useCallback(async () => {
    const d = await api.agent(env, profile);
    setDetail(d);
    setConfigText(JSON.stringify(d.config, null, 2));
  }, [env, profile]);

  useEffect(() => {
    seen.current = -1;
    setEvents([]);
    loadDetail();
  }, [loadDetail]);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      if (!alive) return;
      try {
        const next = await api.events(env, profile, seen.current);
        if (next.length) {
          seen.current = next[next.length - 1].seq;
          setEvents((e) => [...e, ...next].slice(-1200));
        }
      } catch { /* transient */ }
    };
    tick();
    const id = setInterval(tick, 1200);
    return () => { alive = false; clearInterval(id); };
  }, [env, profile]);

  useEffect(() => {
    const el = scroller.current;
    if (follow && el) el.scrollTop = el.scrollHeight;
  }, [events.length, follow]);

  async function act(a: string) {
    try {
      await api.action(env, profile, a);
      setNotice(null);
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
    onChange();
    loadDetail();
  }

  async function saveConfig() {
    try {
      const res = await api.patchAgent(env, profile, { config: JSON.parse(configText) });
      setNotice(res.warnings?.length ? res.warnings.join(" ") : "saved");
      onChange();
      loadDetail();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
  }

  const b = agent.config?.budgets ?? { input_tokens: 1, output_tokens: 1 };

  return (
    <div className="content" ref={scroller}>
      <div className="card">
        <div className="spread">
          <div className="row">
            <span className={`dot s-${agent.state}`} />
            <b>{profile}</b>
            <span className="dim mono-sm">{agent.config.provider}/{agent.config.model} · {agent.state}
              {agent.wake_at ? ` · wakes in ${Math.max(0, Math.round(agent.wake_at - Date.now() / 1000))}s` : ""}</span>
          </div>
          <div className="row">
            <button onClick={() => act("start")} disabled={agent.running}>start</button>
            <button onClick={() => act("stop")} disabled={!agent.running}>stop</button>
            <button onClick={() => act("wake")}>wake</button>
            <button onClick={() => act("reset")} disabled={agent.running} title="clear conversation, keep memory.md">reset ctx</button>
          </div>
        </div>
        <div className="dim mono-sm" style={{ margin: "8px 0" }}>{agent.status || agent.stop_reason || "—"}</div>
        <div className="grid2">
          <div>
            <div className="spread mono-sm dim"><span>input tokens</span><span>{num(agent.usage.input_tokens)} / {num(b.input_tokens)}</span></div>
            <Bar used={agent.usage.input_tokens} total={b.input_tokens} />
          </div>
          <div>
            <div className="spread mono-sm dim"><span>output tokens</span><span>{num(agent.usage.output_tokens)} / {num(b.output_tokens)}</span></div>
            <Bar used={agent.usage.output_tokens} total={b.output_tokens} />
          </div>
        </div>
        {detail && (
          <div className="dim mono-sm" style={{ marginTop: 8 }}>
            context: {detail.context.items} items · {detail.context.compactions} compactions · last input {num(detail.context.last_input_tokens)} tok
            {agent.config.memoryless ? " · memoryless" : ""} · cost ${agent.usage.cost_usd.toFixed(4)}
          </div>
        )}
        {notice && <div className="mono-sm" style={{ color: "var(--amber)", marginTop: 6 }}>{notice}</div>}
      </div>

      <div className="row" style={{ marginBottom: 8 }}>
        <button onClick={() => setView("history")} style={view === "history" ? { borderColor: "var(--accent)" } : {}}>history</button>
        <button onClick={() => setView("config")} style={view === "config" ? { borderColor: "var(--accent)" } : {}}>config &amp; files</button>
        <div style={{ flex: 1 }} />
        <label className="dim mono-sm row"><input type="checkbox" style={{ width: "auto" }} checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow</label>
      </div>

      {view === "history" && (
        <div className="card" style={{ fontSize: 12 }}>
          {events.length === 0 && <div className="dim">No history yet. Start the runner.</div>}
          {events.map((ev, i) => (
            <div key={i} className={`ev ev-${ev.kind} ${ev.error ? "err" : ""}`}>{renderEvent(ev)}</div>
          ))}
        </div>
      )}

      {view === "config" && detail && (
        <>
          <div className="card">
            <div className="spread"><b>runner config</b><button onClick={saveConfig} disabled={agent.running}>save</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
              Host-side. Agents cannot read or write this. Stop the runner to edit.
            </div>
            <textarea rows={16} value={configText} onChange={(e) => setConfigText(e.target.value)} disabled={agent.running} />
          </div>
          <div className="card">
            <div className="spread"><b>AGENT.md</b>
              <button onClick={() => api.patchAgent(env, profile, { agent_md: detail.agent_md }).then(() => setNotice("saved"))}>save</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>Agent-writable. Loaded into its instructions each run.</div>
            <textarea rows={10} value={detail.agent_md} onChange={(e) => setDetail({ ...detail, agent_md: e.target.value })} />
          </div>
          <div className="card">
            <div className="spread"><b>memory.md</b>
              <span className="dim mono-sm">{detail.memory.length} / {agent.config.memory_char_limit} chars</span></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>The state snapshot re-injected after every compaction.</div>
            <pre className="out">{detail.memory}</pre>
          </div>
          {detail.context.summary && (
            <div className="card">
              <b>last compaction summary</b>
              <pre className="out">{detail.context.summary}</pre>
            </div>
          )}
        </>
      )}
    </div>
  );
}

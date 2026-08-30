import { useCallback, useEffect, useRef, useState } from "react";
import { Agent, api, cost, num } from "../api";

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
      return `── compacted ${ev.native ? "server-side" : "by summary"} (${ev.reason})`
        + `${ev.dropped_items ? `, dropped ${ev.dropped_items} items` : ""} ──\n${ev.summary ?? ""}`;
    case "run_start":
      return `── run start: ${ev.reason} ──`;
    case "run_end":
      return `── run end: ${ev.state} (${ev.reason}) ──`;
    case "sleep":
      return `── sleeping${ev.seconds ? ` ${ev.seconds}s` : ""}: ${ev.status ?? ""} ──`;
    case "model_swap":
      return `── model swap ${ev.old} → ${ev.new}; context compacted ──`;
    case "hosted_tool":
      return `⇢ ${ev.provider} hosted ${ev.tool}`;
    default:
      return `${ev.kind}: ${JSON.stringify({ ...ev, ts: undefined, kind: undefined, seq: undefined })}`;
  }
}

export default function AgentPanel({ env, agent, onChange, onDeleted, onBack }: {
  env: string; agent: Agent; onChange: () => void; onDeleted: () => void; onBack: () => void;
}) {
  const [events, setEvents] = useState<any[]>([]);
  const [detail, setDetail] = useState<any>(null);
  const [view, setView] = useState<"history" | "config">("history");
  const [configText, setConfigText] = useState("");
  const [promptOverride, setPromptOverride] = useState("");
  const [goal, setGoal] = useState("");
  const [memory, setMemory] = useState("");
  const [toolOverrides, setToolOverrides] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<string | null>(null);
  const [follow, setFollow] = useState(true);
  const [now, setNow] = useState(Date.now());
  const seen = useRef(-1);
  const scroller = useRef<HTMLDivElement>(null);

  const profile = agent.profile;

  const loadDetail = useCallback(async () => {
    const d = await api.agent(env, profile);
    setDetail(d);
    setConfigText(JSON.stringify(d.config, null, 2));
    setPromptOverride(d.config.base_prompt_override ?? "");
    setGoal(d.config.goal ?? "");
    setMemory(d.memory ?? "");
    setToolOverrides(d.config.tool_description_overrides ?? {});
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

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

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

  async function savePromptOverride() {
    if (!detail) return;
    try {
      const config = { ...detail.config, base_prompt_override: promptOverride.trim() || null };
      await api.patchAgent(env, profile, { config });
      setNotice(promptOverride.trim() ? "base system prompt override saved" : "using built-in base prompt");
      onChange();
      loadDetail();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
  }

  async function saveGoal() {
    if (!detail) return;
    try {
      const config = {
        ...detail.config,
        goal,
      };
      await api.patchAgent(env, profile, { config });
      setNotice("agent goal saved");
      onChange();
      loadDetail();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
  }

  async function saveMemory() {
    try {
      await api.patchAgent(env, profile, { memory });
      setNotice("memory.md saved; the current context snapshot updates at the next compaction/reset");
      loadDetail();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
  }

  async function saveToolDescriptions() {
    if (!detail) return;
    try {
      const cleaned = Object.fromEntries(
        Object.entries(toolOverrides).filter(([, value]) => value.trim())
      );
      await api.patchAgent(env, profile, {
        config: { ...detail.config, tool_description_overrides: cleaned },
      });
      setNotice("tool descriptions saved");
      onChange();
      loadDetail();
    } catch (e: any) {
      setNotice(String(e.message ?? e));
    }
  }

  const b = agent.config?.budgets ?? { input_tokens: 1, output_tokens: 1 };
  const waitingFor = agent.state === "waiting"
    ? Math.max(0, Math.floor((now - Date.parse(agent.updated_at)) / 1000))
    : null;

  return (
    <div className="content" ref={scroller}>
      <div className="card">
        <div className="spread">
          <div className="row">
            <button onClick={onBack}>← all agents</button>
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
            <button onClick={async () => {
              if (!confirm(`Delete agent profile "${profile}" and its files?`)) return;
              try {
                await api.deleteAgent(env, profile);
                onDeleted();
              } catch (e: any) {
                setNotice(String(e.message ?? e));
              }
            }} style={{ color: "var(--red)" }}>delete</button>
          </div>
        </div>
        <div className="dim mono-sm" style={{ margin: "8px 0" }}>
          {agent.status || agent.stop_reason || "—"}
          {waitingFor !== null ? ` · waiting ${waitingFor}s` : ""}
        </div>
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
            {agent.config.memoryless ? " · memoryless" : ""} · cost {cost(agent.usage)}
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
            <div className="spread"><b>agent goal</b><button onClick={saveGoal} disabled={agent.running}>save</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
              Host-controlled and configured independently for this runner. The goal is supplied separately from the system prompt as application input.
            </div>
            <div className="section-title">goal</div>
            <textarea rows={6} value={goal} disabled={agent.running} onChange={(e) => setGoal(e.target.value)} />
          </div>
          <div className="card">
            <div className="spread"><b>runner config</b><button onClick={saveConfig} disabled={agent.running}>save</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
              Host-side. Agents cannot read or write this. Stop the runner to edit. This includes enabled tools, per-tool description overrides, summary model, budgets, and context settings.
            </div>
            <textarea rows={16} value={configText} onChange={(e) => setConfigText(e.target.value)} disabled={agent.running} />
          </div>
          <div className="card">
            <div className="spread"><b>tool descriptions</b><button onClick={saveToolDescriptions} disabled={agent.running}>save</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
              Optional per-runner overrides. Empty fields use the built-in description. The exact resulting provider payload is shown below.
            </div>
            {detail.prompt.tools.map((tool: any) => (
              <details key={tool.name} className="prompt-detail">
                <summary><code>{tool.name}</code></summary>
                <div className="dim mono-sm" style={{ margin: "7px 0" }}>Built in: {tool.default_description}</div>
                <textarea rows={4} disabled={agent.running} value={toolOverrides[tool.name] ?? ""}
                  placeholder="Use built-in description"
                  onChange={(e) => setToolOverrides({ ...toolOverrides, [tool.name]: e.target.value })} />
              </details>
            ))}
          </div>
          <div className="card">
            <div className="spread"><b>system-prompt override</b><button onClick={savePromptOverride} disabled={agent.running}>save override</button></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
              Optional. Empty uses the built-in prompt shown below. A non-empty value replaces it completely. Changes take effect on the next wake.
            </div>
            <textarea className="code-input" rows={8} value={promptOverride} disabled={agent.running} onChange={(e) => setPromptOverride(e.target.value)} placeholder="Leave empty to use the built-in base instructions." />
          </div>
          {detail.prompt && (
            <div className="card">
              <b>assembled system prompt</b>
              <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>
                Exact developer/system instructions for the next wake. Tool schemas and user-role runtime injections are shown separately because providers receive them outside this string.
              </div>
              {detail.prompt.using_base_override && (
                <div className="prompt-warning">
                  This agent is using a system-prompt override. Changes to the built-in prompt—including new collaboration and token-efficiency guidance—do not apply until you clear the override above.
                </div>
              )}
              <pre className="out prompt-preview">{detail.prompt.system_prompt}</pre>
              <details className="prompt-detail">
                <summary>current provider conversation items <span className="dim">· exact stored input before the next wake message</span></summary>
                <pre className="out prompt-preview">{JSON.stringify(detail.prompt.conversation_items, null, 2)}</pre>
              </details>
              <div className="section-title">tool definitions sent to {agent.config.provider}</div>
              {detail.prompt.tools.map((t: any) => (
                <details key={t.name} className="prompt-detail">
                  <summary><code>{t.name}</code> <span className="dim">· {t.transport}</span></summary>
                  <div className="mono-sm" style={{ margin: "7px 0" }}>{t.logical_description}</div>
                  <pre className="out">{JSON.stringify(t.provider_payload, null, 2)}</pre>
                </details>
              ))}
              <div className="section-title">other injected prompts</div>
              {detail.prompt.injected.map((p: any) => (
                <details key={p.name} className="prompt-detail">
                  <summary>{p.name} <span className="dim">· {p.timing}</span></summary>
                  <pre className="out">{p.content}</pre>
                </details>
              ))}
            </div>
          )}
          <div className="card">
            <div className="spread"><b>memory.md</b>
              <div className="row"><span className="dim mono-sm">{memory.length} / {agent.config.memory_char_limit} chars</span><button onClick={saveMemory}>save</button></div></div>
            <div className="dim mono-sm" style={{ margin: "4px 0 8px" }}>Agent-writable state re-injected after every compaction. Operator edits are persisted immediately.</div>
            <textarea className="code-input" rows={12} value={memory} onChange={(e) => setMemory(e.target.value)} />
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

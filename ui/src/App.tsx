import { useCallback, useEffect, useRef, useState } from "react";
import { Agent, EnvSummary, Message, api, num } from "./api";
import Board from "./components/Board";
import AgentPanel from "./components/AgentPanel";
import FilesPanel from "./components/FilesPanel";
import UsagePanel from "./components/UsagePanel";
import ConsolePanel from "./components/ConsolePanel";
import EnvSettings from "./components/EnvSettings";
import NewAgent from "./components/NewAgent";
import NewEnv from "./components/NewEnv";
import AgentsOverview from "./components/AgentsOverview";

type Tab = "board" | "agent" | "files" | "usage" | "console" | "settings";

export default function App() {
  const [envs, setEnvs] = useState<EnvSummary[]>([]);
  const [env, setEnv] = useState<string | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [tab, setTab] = useState<Tab>("board");
  const [selected, setSelected] = useState<string | null>(null);
  const [creatingAgent, setCreatingAgent] = useState(false);
  const [creatingEnv, setCreatingEnv] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const loadEnvs = useCallback(async () => {
    try {
      const list = await api.envs();
      setEnvs(list);
      setEnv((cur) => cur ?? list[0]?.name ?? null);
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  }, []);

  useEffect(() => {
    loadEnvs();
  }, [loadEnvs]);

  // One SSE stream per environment carries both board messages and agent state.
  useEffect(() => {
    esRef.current?.close();
    setMessages([]);
    setAgents([]);
    if (!env) return;
    let cancelled = false;
    (async () => {
      const [msgs, ags] = await Promise.all([api.board(env), api.agents(env)]);
      if (cancelled) return;
      setMessages(msgs);
      setAgents(ags);
      const after = msgs.length ? msgs[msgs.length - 1].id : 0;
      const es = new EventSource(`/api/envs/${env}/stream?after=${after}`);
      es.onmessage = (ev) => {
        const data = JSON.parse(ev.data);
        if (data.type === "messages") setMessages((m) => [...m, ...data.messages]);
        else if (data.type === "agents") setAgents(data.agents);
      };
      esRef.current = es;
    })().catch((e) => setError(String(e.message ?? e)));
    return () => {
      cancelled = true;
      esRef.current?.close();
    };
  }, [env]);

  const refreshAgents = useCallback(async () => {
    if (env) setAgents(await api.agents(env));
  }, [env]);

  const current = agents.find((a) => a.profile === selected) ?? null;
  const envInfo = envs.find((e) => e.name === env);

  return (
    <div className="app">
      <div className="sidebar">
        <div className="spread brand">
          <div className="row">
            <img src="/autonomous-agents-rising.png" alt="" className="brand-logo" />
            <b className="brand-name">libertarian<br />agents</b>
          </div>
          <button onClick={() => setCreatingEnv(true)} title="new environment">+</button>
        </div>

        <div className="section-title">Environments</div>
        {envs.map((e) => (
          <div key={e.name} className={`item ${e.name === env ? "on" : ""}`} onClick={() => { setEnv(e.name); setSelected(null); setTab("board"); }}>
            <span className={`dot ${e.status.running ? "s-active" : "s-inactive"}`} />
            <span>{e.name}</span>
          </div>
        ))}

        {env && (
          <>
            <div className="section-title">Environment</div>
            <div className="mono-sm dim" style={{ padding: "0 8px", lineHeight: 1.7 }}>
              <div>sandbox: {envInfo?.config?.sandbox}</div>
              <div>{envInfo?.status.running ? "running" : "quiescent"}</div>
              <div>tokens in: {num(agents.reduce((s, a) => s + a.usage.input_tokens, 0))}</div>
            </div>
          </>
        )}
      </div>

      <div className="main">
        <div className="tabs">
          {(["board", "agent", "files", "usage", "console", "settings"] as Tab[]).map((t) => (
            <div key={t} className={`tab ${tab === t ? "on" : ""}`} onClick={() => {
              setTab(t);
              if (t === "agent") setSelected(null);
            }}>
              {t === "agent" ? "agents" : t}
            </div>
          ))}
          <div style={{ flex: 1 }} />
          {error && <span className="dim mono-sm" style={{ color: "var(--red)" }} onClick={() => setError(null)}>{error}</span>}
        </div>

        {!env && <div className="content dim">Create an environment to begin.</div>}
        {env && tab === "board" && <Board env={env} messages={messages} agents={agents} />}
        {env && tab === "agent" && (current ? (
          <AgentPanel
            env={env}
            agent={current}
            onChange={refreshAgents}
            onBack={() => setSelected(null)}
            onDeleted={() => { setSelected(null); setTab("agent"); refreshAgents(); }}
          />
        ) : (
          <AgentsOverview
            env={env}
            agents={agents}
            onSelect={(profile) => setSelected(profile)}
            onCreate={() => setCreatingAgent(true)}
            onChange={refreshAgents}
          />
        ))}
        {env && tab === "files" && <FilesPanel env={env} />}
        {env && tab === "usage" && <UsagePanel env={env} agents={agents} />}
        {env && tab === "console" && <ConsolePanel env={env} />}
        {env && tab === "settings" && (
          <EnvSettings env={env} onChange={loadEnvs} onDeleted={() => { setEnv(null); loadEnvs(); }} />
        )}
      </div>

      {creatingEnv && (
        <NewEnv
          onClose={() => setCreatingEnv(false)}
          onCreated={async (name) => {
            setCreatingEnv(false);
            await loadEnvs();
            setEnv(name);
            setSelected(null);
            setTab("settings");
          }}
        />
      )}
      {creatingAgent && env && (
        <NewAgent env={env} onClose={() => setCreatingAgent(false)} onCreated={() => { setCreatingAgent(false); refreshAgents(); }} />
      )}
    </div>
  );
}

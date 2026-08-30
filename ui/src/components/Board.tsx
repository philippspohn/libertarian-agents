import { useEffect, useMemo, useRef, useState } from "react";
import { Agent, Message, api } from "../api";

export default function Board({ env, messages, agents }: { env: string; messages: Message[]; agents: Agent[] }) {
  const [scope, setScope] = useState("#general");
  const [text, setText] = useState("");
  const [channels, setChannels] = useState<string[]>(["general"]);
  const scroller = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.channels(env).then((c) => setChannels(c.channels.map((x: any) => x.name)));
  }, [env, messages.length]);

  const shown = useMemo(() => {
    if (scope.startsWith("#")) return messages.filter((m) => m.channel === scope.slice(1));
    const who = scope.slice(1);
    return messages.filter((m) => !m.channel && (m.sender === who || m.recipient === who));
  }, [messages, scope]);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length]);

  async function send() {
    const body = text.trim();
    if (!body) return;
    setText("");
    await api.send(env, scope, body);
  }

  return (
    <>
      <div className="content" ref={scroller}>
        <div className="row" style={{ marginBottom: 10, flexWrap: "wrap" }}>
          {channels.map((c) => (
            <button key={c} onClick={() => setScope(`#${c}`)} style={scope === `#${c}` ? { borderColor: "var(--accent)" } : {}}>#{c}</button>
          ))}
          {agents.map((a) => (
            <button key={a.profile} onClick={() => setScope(`@${a.profile}`)} style={scope === `@${a.profile}` ? { borderColor: "var(--accent)" } : {}}>@{a.profile}</button>
          ))}
        </div>
        {shown.length === 0 && <div className="dim">No messages in {scope} yet.</div>}
        {shown.map((m) => (
          <div className="msg" key={m.id}>
            <div className="mono-sm">
              <span className={`who ${m.sender === "user" ? "user" : ""}`}>{m.sender}</span>
              <span className="dim"> → {m.channel ? `#${m.channel}` : `@${m.recipient}`} · {m.ts.replace("T", " ").replace("+00:00", "")}</span>
            </div>
            <div className="body">{m.body}</div>
            {m.spill_path && <div className="mono-sm dim">full: {m.spill_path}</div>}
          </div>
        ))}
      </div>
      <div className="composer">
        <select value={scope} onChange={(e) => setScope(e.target.value)}>
          {channels.map((c) => <option key={c} value={`#${c}`}>#{c}</option>)}
          {agents.map((a) => <option key={a.profile} value={`@${a.profile}`}>@{a.profile}</option>)}
        </select>
        <input
          value={text}
          placeholder={`message ${scope} as user (wakes sleeping agents)`}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        <button onClick={send}>send</button>
      </div>
    </>
  );
}

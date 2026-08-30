import { useEffect, useState } from "react";
import { api } from "../api";

export default function NewAgent({ env, onClose, onCreated }: { env: string; onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [model, setModel] = useState("gpt-5.6-luna");
  const [provider, setProvider] = useState("openai");
  const [effort, setEffort] = useState("low");
  const [inBudget, setInBudget] = useState("1000000");
  const [outBudget, setOutBudget] = useState("100000");
  const [memoryless, setMemoryless] = useState(false);
  const [tools, setTools] = useState<string[]>([]);
  const [all, setAll] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tools().then((t) => { setAll(t); setTools(t.map((x) => x.name)); });
  }, []);

  async function create() {
    try {
      await api.createAgent(env, name, {
        provider, model, reasoning_effort: effort || null, memoryless, tools,
        budgets: { input_tokens: Number(inBudget), output_tokens: Number(outBudget) },
      });
      onCreated();
    } catch (e: any) {
      setError(String(e.message ?? e));
    }
  }

  return (
    <div style={{ position: "fixed", inset: 0, background: "#000a", display: "grid", placeItems: "center" }} onClick={onClose}>
      <div className="card" style={{ width: 520, maxHeight: "85vh", overflowY: "auto" }} onClick={(e) => e.stopPropagation()}>
        <b>new agent profile</b>
        <div className="section-title">name</div>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="alice" />
        <div className="grid2" style={{ marginTop: 10 }}>
          <div>
            <div className="section-title">provider</div>
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              <option value="openai">openai</option>
              <option value="openrouter">openrouter</option>
            </select>
          </div>
          <div>
            <div className="section-title">model</div>
            <input value={model} onChange={(e) => setModel(e.target.value)} />
          </div>
          <div>
            <div className="section-title">reasoning effort</div>
            <input value={effort} onChange={(e) => setEffort(e.target.value)} placeholder="low / medium / high" />
          </div>
          <div>
            <div className="section-title">input token budget</div>
            <input value={inBudget} onChange={(e) => setInBudget(e.target.value.replace(/\D/g, ""))} />
          </div>
          <div>
            <div className="section-title">output token budget</div>
            <input value={outBudget} onChange={(e) => setOutBudget(e.target.value.replace(/\D/g, ""))} />
          </div>
          <div>
            <div className="section-title">memoryless</div>
            <label className="row"><input type="checkbox" style={{ width: "auto" }} checked={memoryless} onChange={(e) => setMemoryless(e.target.checked)} />
              <span className="dim mono-sm">reset context on every sleep</span></label>
          </div>
        </div>
        <div className="section-title">tools</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          {all.map((t) => (
            <label key={t.name} className="row" title={t.description}>
              <input type="checkbox" style={{ width: "auto" }} checked={tools.includes(t.name)}
                onChange={(e) => setTools((c) => e.target.checked ? [...c, t.name] : c.filter((x) => x !== t.name))} />
              <span className="mono-sm">{t.name}</span>
            </label>
          ))}
        </div>
        {error && <div style={{ color: "var(--red)", marginTop: 8 }}>{error}</div>}
        <div className="row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
          <button onClick={onClose}>cancel</button>
          <button onClick={create} disabled={!name}>create</button>
        </div>
      </div>
    </div>
  );
}

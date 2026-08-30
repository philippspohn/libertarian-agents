import { useState } from "react";
import { api } from "../api";

export default function ConsolePanel({ env }: { env: string }) {
  const [cmd, setCmd] = useState("ls -la");
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    try {
      const r = await api.exec(env, cmd);
      setOut(`$ ${cmd}\n${r.output}\n[exit ${r.exit_code}${r.timed_out ? ", timed out" : ""}]\n\n${out}`);
    } catch (e: any) {
      setOut(`error: ${e.message ?? e}\n\n${out}`);
    }
    setBusy(false);
  }

  return (
    <div className="content">
      <div className="card">
        <div className="dim mono-sm" style={{ marginBottom: 8 }}>
          Operator shell into the sandbox, rooted at the environment. Output is not compressed here.
        </div>
        <div className="row">
          <input value={cmd} onChange={(e) => setCmd(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} />
          <button onClick={run} disabled={busy}>run</button>
        </div>
      </div>
      <pre className="out" style={{ maxHeight: "70vh" }}>{out || "(no output yet)"}</pre>
    </div>
  );
}

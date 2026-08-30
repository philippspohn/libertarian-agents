export interface Usage {
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cost_usd: number;
}

export interface Agent {
  profile: string;
  state: "inactive" | "active" | "waiting" | "finished";
  running: boolean;
  wake_at: number | null;
  stop_reason: string | null;
  status: string;
  config: any;
  usage: Usage;
  unread: number;
}

export interface Message {
  id: number;
  ts: string;
  sender: string;
  channel: string | null;
  recipient: string | null;
  body: string;
  spill_path: string | null;
}

export interface EnvSummary {
  name: string;
  created_at: string;
  config: any;
  status: { running: boolean; quiescent: boolean; active: string[]; waiting: string[] };
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error((await res.text()) || res.statusText);
  return res.json() as Promise<T>;
}

export const api = {
  envs: () => req<EnvSummary[]>("/envs"),
  createEnv: (name: string, config: any) =>
    req("/envs", { method: "POST", body: JSON.stringify({ name, config }) }),
  env: (e: string) => req<any>(`/envs/${e}`),
  patchEnv: (e: string, config: any) =>
    req(`/envs/${e}`, { method: "PATCH", body: JSON.stringify(config) }),
  deleteEnv: (e: string) => req(`/envs/${e}`, { method: "DELETE" }),

  agents: (e: string) => req<Agent[]>(`/envs/${e}/agents`),
  agent: (e: string, p: string) => req<any>(`/envs/${e}/agents/${p}`),
  createAgent: (e: string, name: string, config: any) =>
    req(`/envs/${e}/agents`, { method: "POST", body: JSON.stringify({ name, config }) }),
  patchAgent: (e: string, p: string, body: any) =>
    req<any>(`/envs/${e}/agents/${p}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteAgent: (e: string, p: string) => req(`/envs/${e}/agents/${p}`, { method: "DELETE" }),
  action: (e: string, p: string, a: string) =>
    req(`/envs/${e}/agents/${p}/${a}`, { method: "POST" }),
  events: (e: string, p: string, after: number) =>
    req<any[]>(`/envs/${e}/agents/${p}/events?after=${after}`),

  board: (e: string, after = 0) => req<Message[]>(`/envs/${e}/board?after=${after}`),
  channels: (e: string) => req<any>(`/envs/${e}/channels`),
  send: (e: string, to: string, body: string) =>
    req<any>(`/envs/${e}/board/send`, { method: "POST", body: JSON.stringify({ to, body }) }),

  files: (e: string, path: string) =>
    req<any>(`/envs/${e}/files?path=${encodeURIComponent(path)}`),
  exec: (e: string, command: string) =>
    req<any>(`/envs/${e}/exec`, { method: "POST", body: JSON.stringify({ command }) }),
  usage: (e: string) => req<any>(`/envs/${e}/usage`),
  tools: () => req<any[]>("/tools"),
};

export function num(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

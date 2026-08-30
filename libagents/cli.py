"""Command line interface. Everything the UI can do, the CLI can do."""

from __future__ import annotations

import json
import time
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import control, environment, paths
from .board import USER, Board
from .models import Budgets, EnvConfig, RunnerConfig
from .supervisor import SUPERVISOR, env_status

load_dotenv()
console = Console()

app = typer.Typer(help="Libertarian Agents: a shared sandbox for self-determined agents.", no_args_is_help=True)
env_app = typer.Typer(help="Manage environments.", no_args_is_help=True)
agent_app = typer.Typer(help="Manage agent profiles and runners.", no_args_is_help=True)
app.add_typer(env_app, name="env")
app.add_typer(agent_app, name="agent")


def _board(env: str) -> Board:
    return Board(paths.board_db(env))


DEFAULT_SERVER = "http://127.0.0.1:8848"


def _server() -> Optional[str]:
    """If a `libagents serve` daemon is up, drive it instead of spawning a
    runner inside this short-lived CLI process -- otherwise the agent would
    die the moment the command returns."""
    import os

    import httpx

    url = os.environ.get("LIBAGENTS_SERVER", DEFAULT_SERVER).rstrip("/")
    try:
        httpx.get(f"{url}/api/envs", timeout=0.75).raise_for_status()
        return url
    except Exception:
        return None


# --------------------------------------------------------------------- env


@env_app.command("create")
def env_create(
    name: str,
    sandbox: str = typer.Option("docker", "--sandbox", help="docker | local"),
    image: str = typer.Option("python:3.12-slim", "--image"),
    secret: list[str] = typer.Option([], "--secret", help="Host env var to forward into the sandbox."),
):
    cfg = EnvConfig(
        sandbox=sandbox,
        image=image,
        secrets=list(secret),
    )
    environment.create_environment(name, cfg)
    console.print(f"[green]created[/] environment {name} at {paths.env_dir(name)}")
    if sandbox == "local":
        console.print("[yellow]note:[/] the local sandbox runs commands on this host without isolation")


@env_app.command("ls")
def env_ls():
    table = Table("name", "sandbox", "agents", "state")
    for e in control.list_envs():
        st = env_status(e["name"])
        table.add_row(
            e["name"],
            e["config"].get("sandbox", "?"),
            str(len(environment.list_profiles(e["name"]))),
            "running" if st["running"] else "quiescent",
        )
    console.print(table)


@env_app.command("set")
def env_set(
    name: str,
    sandbox: Optional[str] = typer.Option(None, "--sandbox"),
    cap: Optional[int] = typer.Option(None, "--input-token-cap"),
):
    cfg = control.get_env(name)
    if sandbox is not None:
        cfg.sandbox = sandbox  # type: ignore[assignment]
    if cap is not None:
        cfg.input_token_cap = cap
    environment.update_environment(name, cfg)
    console.print(cfg.model_dump())


@env_app.command("rm")
def env_rm(name: str, keep_files: bool = typer.Option(False, "--keep-files")):
    remaining = SUPERVISOR.stop_env(name, join=15)
    if remaining:
        raise typer.BadParameter(f"still stopping: {', '.join(remaining)}; retry shortly")
    environment.delete_environment(name, remove_files=not keep_files)
    console.print(f"[red]deleted[/] environment {name}")


# ------------------------------------------------------------------- agent


@agent_app.command("create")
def agent_create(
    env: str,
    name: str,
    model: str = typer.Option("gpt-5.6-luna", "--model", "-m"),
    provider: str = typer.Option("openai", "--provider"),
    effort: str = typer.Option("low", "--effort"),
    input_budget: int = typer.Option(1_000_000, "--input-budget"),
    output_budget: int = typer.Option(100_000, "--output-budget"),
    tools: Optional[str] = typer.Option(None, "--tools", help="Comma-separated tool names."),
    memoryless: bool = typer.Option(False, "--memoryless"),
    goal: str = typer.Option("", "--goal", "-g"),
    summary_provider: str = typer.Option("openrouter", "--summary-provider"),
    summary_model: str = typer.Option("deepseek/deepseek-v4-flash-0731", "--summary-model"),
):
    cfg = RunnerConfig(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        reasoning_effort=effort or None,
        goal=goal,
        summary_provider=summary_provider,  # type: ignore[arg-type]
        summary_model=summary_model,
        budgets=Budgets(input_tokens=input_budget, output_tokens=output_budget),
        memoryless=memoryless,
    )
    if tools:
        cfg.tools = [t.strip() for t in tools.split(",") if t.strip()]
    environment.create_profile(env, name, cfg)
    console.print(f"[green]created[/] profile {env}/{name}")


@agent_app.command("ls")
def agent_ls(env: str):
    board = _board(env)
    statuses = {a["name"]: a for a in board.list_agents()}
    table = Table("profile", "state", "model", "in/budget", "out/budget", "status")
    for r in control.list_runners(env):
        used = control.usage_for(env, r.profile)
        table.add_row(
            r.profile,
            r.state,
            r.config.model,
            f"{used.input_tokens}/{r.config.budgets.input_tokens}",
            f"{used.output_tokens}/{r.config.budgets.output_tokens}",
            (statuses.get(r.profile, {}).get("status") or r.stop_reason or "")[:44],
        )
    console.print(table)


@agent_app.command("config")
def agent_config(env: str, name: str, set_json: Optional[str] = typer.Option(None, "--set", help="JSON patch.")):
    runner = control.get_runner(env, name)
    if set_json:
        if runner.state in ("active",):
            raise typer.BadParameter("stop the runner before changing its config")
        patch = json.loads(set_json)
        merged = {**runner.config.model_dump(), **patch}
        cfg = RunnerConfig.model_validate(merged)
        if cfg.model != runner.config.model:
            console.print("[yellow]warning:[/] changing the model forces a compaction; reasoning context is lost")
        control.upsert_runner(env, name, cfg)
        runner = control.get_runner(env, name)
    console.print_json(runner.config.model_dump_json())


@agent_app.command("rm")
def agent_rm(env: str, name: str, keep_files: bool = typer.Option(False, "--keep-files")):
    if not SUPERVISOR.stop(env, name, join=15):
        raise typer.BadParameter("runner is still stopping; retry shortly")
    environment.delete_profile(env, name, remove_files=not keep_files)
    console.print(f"[red]deleted[/] {env}/{name}")


# ------------------------------------------------------------------ running


@app.command("start")
def start(env: str, name: str, reason: str = typer.Option("operator", "--reason")):
    server = _server()
    if server:
        import httpx

        httpx.post(f"{server}/api/envs/{env}/agents/{name}/start", params={"reason": reason}, timeout=10)
        console.print(f"[green]started[/] {env}/{name} on {server}")
        return
    console.print("[yellow]no server running[/] -- use `libagents serve`, or `libagents run` to stay attached")
    raise typer.Exit(1)


@app.command("stop")
def stop(env: str, name: str):
    server = _server()
    if server:
        import httpx

        httpx.post(f"{server}/api/envs/{env}/agents/{name}/stop", timeout=10)
    else:
        SUPERVISOR.stop(env, name, join=10)
    console.print(f"[yellow]stopping[/] {env}/{name}")


@app.command("run")
def run(
    env: str,
    name: str,
    reason: str = typer.Option("started by operator", "--reason"),
    exit_on_sleep: bool = typer.Option(False, "--exit-on-sleep"),
):
    """Start a runner in the foreground and stream its events.

    Stays attached while the agent sleeps, so it can still be woken by
    messages. Ctrl-C stops the runner. For long-lived multi-agent work run
    `libagents serve` instead.
    """
    from .events import EventLog

    log = paths.history_dir(env, name) / "events.jsonl"
    seen = (sum(1 for _ in log.open()) - 1) if log.exists() else -1
    events = EventLog(log)
    SUPERVISOR.start(env, name, reason)
    try:
        while True:
            for ev in events.tail(limit=500, after=seen):
                seen = ev["seq"]
                _print_event(ev)
                if ev["kind"] == "sleep" and exit_on_sleep:
                    return
            if not SUPERVISOR.is_running(env, name):
                time.sleep(0.4)
                for ev in events.tail(limit=500, after=seen):
                    seen = ev["seq"]
                    _print_event(ev)
                break
            time.sleep(0.4)
    except KeyboardInterrupt:
        console.print("[dim]-- stopping runner --[/]")
        SUPERVISOR.stop(env, name, join=15)


def _print_event(ev: dict) -> None:
    kind = ev["kind"]
    if kind == "reasoning":
        console.print(f"[dim italic]{ev['text'][:800]}[/]")
    elif kind == "message":
        console.print(f"[cyan]{ev['text']}[/]")
    elif kind == "tool_call":
        console.print(f"[magenta]-> {ev['tool']}[/] {json.dumps(ev.get('arguments', {}))[:300]}")
    elif kind == "tool_result":
        style = "red" if ev.get("error") else "green"
        console.print(f"[{style}]   {str(ev.get('summary'))[:600]}[/]")
    elif kind == "usage":
        console.print(f"[dim]   +{ev['input_tokens']} in / {ev['output_tokens']} out[/]")
    elif kind == "compaction":
        console.print(f"[yellow]-- compacted ({ev.get('reason')}) --[/]")
    else:
        console.print(f"[dim]{kind}: {json.dumps({k: v for k, v in ev.items() if k not in ('ts', 'kind', 'seq')})[:300]}[/]")


# -------------------------------------------------------------------- board


@app.command("say")
def say(env: str, target: str, message: str):
    """Send a message to an agent ('@alice') or channel ('#general') as the operator."""
    server = _server()
    if server:
        import httpx

        r = httpx.post(f"{server}/api/envs/{env}/board/send",
                       json={"to": target, "body": message}, timeout=15).json()
        console.print(f"[green]sent[/] #{r['id']}" + (f", woke {', '.join(r['woken'])}" if r.get("woken") else ""))
        return
    mid, _ = _board(env).send(USER, target, message,
                              max_chars=control.get_env(env).max_message_chars,
                              spill_dir=paths.shared_dir(env) / "messages")
    console.print(f"[green]sent[/] #{mid} [dim](no server running -- nobody was woken)[/]")


@app.command("board")
def board_cmd(env: str, limit: int = typer.Option(30, "--limit", "-n")):
    for m in _board(env).recent(limit=limit):
        console.print(f"[dim]{m.ts}[/] {m.render()}")


@app.command("usage")
def usage_cmd(env: str):
    table = Table(
        "profile", "model", "calls", "input", "cached", "cache writes",
        "output", "reasoning", "cost $",
    )
    for row in control.usage_breakdown(env):
        shown_cost = f"{row['c']:.4f}" if row["k"] else "unavailable"
        table.add_row(row["profile"], row["model"], str(row["calls"]), str(row["i"]),
                      str(row["ci"]), str(row["cw"]), str(row["o"]), str(row["r"]), shown_cost)
    console.print(table)
    total = control.usage_for(env)
    shown_total = f"${total.cost_usd:.4f}" if total.cost_known else "pricing unavailable"
    console.print(f"[bold]total[/] {total.input_tokens} in / {total.output_tokens} out / {shown_total}")


@app.command("status")
def status_cmd(env: str):
    console.print_json(json.dumps(env_status(env)))


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8848, reload: bool = False):
    """Run the web UI and API."""
    import uvicorn

    uvicorn.run("libagents.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()

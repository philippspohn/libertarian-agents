"""The message board.

A plain SQLite file at `<root>/shared/board.db`, inside the environment. The
messaging tools are thin wrappers over it, and agents with a shell can query
it directly -- that is deliberate. The UI reads the same file, so there is no
second copy of the conversation to keep in sync.

Read model: one cursor per agent covering the whole inbox. `check_inbox`
advances it; `read_history` never does.
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
  name       TEXT PRIMARY KEY,
  status     TEXT NOT NULL DEFAULT '',
  state      TEXT NOT NULL DEFAULT 'inactive',
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS channels (
  name       TEXT PRIMARY KEY,
  topic      TEXT NOT NULL DEFAULT '',
  created_by TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subscriptions (
  agent   TEXT NOT NULL,
  channel TEXT NOT NULL,
  PRIMARY KEY (agent, channel)
);
CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  sender     TEXT NOT NULL,
  channel    TEXT,
  recipient  TEXT,
  body       TEXT NOT NULL,
  spill_path TEXT
);
CREATE INDEX IF NOT EXISTS messages_channel ON messages (channel, id);
CREATE INDEX IF NOT EXISTS messages_recipient ON messages (recipient, id);
CREATE TABLE IF NOT EXISTS cursors (
  agent   TEXT PRIMARY KEY,
  last_id INTEGER NOT NULL DEFAULT 0
);
"""

USER = "user"
"""The human's identity on the board, so they can DM agents from the UI."""


@dataclass
class Message:
    id: int
    ts: str
    sender: str
    channel: Optional[str]
    recipient: Optional[str]
    body: str
    spill_path: Optional[str]

    @property
    def scope(self) -> str:
        return f"#{self.channel}" if self.channel else f"@{self.sender}"

    def render(self) -> str:
        where = f"#{self.channel}" if self.channel else "DM"
        line = f"[{self.id}] {self.sender} -> {where}: {self.body}"
        if self.spill_path:
            line += f"\n      (full message: {self.spill_path})"
        return line


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_target(target: str) -> tuple[Optional[str], Optional[str]]:
    """'#general' -> (channel, None); '@alice'/'alice' -> (None, agent)."""
    t = (target or "").strip()
    if not t:
        raise ValueError("empty message target")
    if t.startswith("#"):
        return t[1:], None
    return None, t.lstrip("@")


class Board:
    def __init__(self, path: Path):
        self.path = Path(path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO channels (name, topic, created_by, created_at) VALUES ('general','Everyone','system',?)",
                (_now(),),
            )
            self._register(c, USER, "human operator", "active")
            c.execute("INSERT OR IGNORE INTO subscriptions VALUES (?, 'general')", (USER,))

    # ---------------------------------------------------------------- agents

    @staticmethod
    def _register(c: sqlite3.Connection, name: str, status: str, state: str) -> None:
        c.execute(
            """INSERT INTO agents (name, status, state, updated_at) VALUES (?,?,?,?)
               ON CONFLICT (name) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at""",
            (name, status, state, _now()),
        )

    def register(self, name: str, state: str = "inactive") -> None:
        with self._conn() as c:
            self._register(c, name, "", state)
            c.execute("INSERT OR IGNORE INTO subscriptions VALUES (?, 'general')", (name,))

    def unregister(self, name: str) -> None:
        """Remove live board identity while preserving historical messages."""
        with self._conn() as c:
            c.execute("DELETE FROM subscriptions WHERE agent=?", (name,))
            c.execute("DELETE FROM cursors WHERE agent=?", (name,))
            c.execute("DELETE FROM agents WHERE name=?", (name,))

    def purge_identity(self, name: str) -> None:
        """Remove board history involving one identity. Channel messages from
        other senders remain because channels are shared state."""
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE sender=? OR recipient=?", (name, name))
            c.execute("DELETE FROM subscriptions WHERE agent=?", (name,))
            c.execute("DELETE FROM cursors WHERE agent=?", (name,))
            c.execute("DELETE FROM agents WHERE name=?", (name,))

    def set_status(self, name: str, status: Optional[str] = None, state: Optional[str] = None) -> None:
        with self._conn() as c:
            row = c.execute("SELECT status, state FROM agents WHERE name=?", (name,)).fetchone()
            cur_status, cur_state = (row["status"], row["state"]) if row else ("", "inactive")
            c.execute(
                """INSERT INTO agents (name, status, state, updated_at) VALUES (?,?,?,?)
                   ON CONFLICT (name) DO UPDATE SET status=excluded.status,
                                                    state=excluded.state,
                                                    updated_at=excluded.updated_at""",
                (name, status if status is not None else cur_status,
                 state if state is not None else cur_state, _now()),
            )

    def list_agents(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM agents ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- channels

    def ensure_channel(self, name: str, topic: str = "", by: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO channels (name, topic, created_by, created_at) VALUES (?,?,?,?)",
                (name.lstrip("#"), topic, by, _now()),
            )

    def subscribe(self, agent: str, channel: str) -> None:
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO subscriptions VALUES (?,?)", (agent, channel.lstrip("#")))

    def unsubscribe(self, agent: str, channel: str) -> bool:
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM subscriptions WHERE agent=? AND channel=?",
                (agent, channel.lstrip("#")),
            )
            return cur.rowcount > 0

    def list_channels(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                """SELECT ch.name, ch.topic,
                          (SELECT COUNT(*) FROM subscriptions s WHERE s.channel=ch.name) members,
                          (SELECT COUNT(*) FROM messages m WHERE m.channel=ch.name) messages
                   FROM channels ch ORDER BY ch.name"""
            ).fetchall()
        return [dict(r) for r in rows]

    def subscribers(self, channel: str) -> set[str]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT agent FROM subscriptions WHERE channel=?", (channel.lstrip("#"),)
            ).fetchall()
        return {r["agent"] for r in rows}

    # -------------------------------------------------------------- messages

    def send(
        self,
        sender: str,
        target: str,
        body: str,
        *,
        max_chars: int = 4000,
        spill_dir: Optional[Path] = None,
    ) -> tuple[int, bool]:
        """Returns (message_id, truncated). Long bodies spill to a file that
        the recipient can read with `read_file`."""
        channel, recipient = parse_target(target)
        spill_path = None
        truncated = len(body) > max_chars
        stored = body
        if truncated and spill_dir is not None:
            spill_dir.mkdir(parents=True, exist_ok=True)
            fp = spill_dir / f"message-{uuid.uuid4().hex[:12]}.txt"
            fp.write_text(body, encoding="utf-8")
            try:
                spill_path = str(fp.resolve().relative_to(self.path.parent.parent.resolve()))
            except ValueError as exc:
                raise ValueError("spill directory must be inside the environment") from exc
            stored = body[:max_chars] + f"\n... [truncated, {len(body)} chars total]"
        elif truncated:
            stored = body[:max_chars] + "\n... [truncated]"

        with self._conn() as c:
            if channel:
                c.execute(
                    "INSERT OR IGNORE INTO channels (name, topic, created_by, created_at) VALUES (?,'',?,?)",
                    (channel, sender, _now()),
                )
                c.execute("INSERT OR IGNORE INTO subscriptions VALUES (?,?)", (sender, channel))
            cur = c.execute(
                "INSERT INTO messages (ts, sender, channel, recipient, body, spill_path) VALUES (?,?,?,?,?,?)",
                (_now(), sender, channel, recipient, stored, spill_path),
            )
            return int(cur.lastrowid), truncated

    def _unread_query(self) -> str:
        return """
          SELECT * FROM messages
          WHERE id > COALESCE((SELECT last_id FROM cursors WHERE agent=:a), 0)
            AND id > :after
            AND sender != :a
            AND (recipient = :a
                 OR (channel IS NOT NULL
                     AND channel IN (SELECT channel FROM subscriptions WHERE agent=:a)))
          ORDER BY id
        """

    def unread_count(self, agent: str, after: int = 0) -> int:
        """`after` lets the supervisor wake an agent only on messages that
        arrived since it went to sleep, so an agent that sleeps without
        clearing its inbox does not spin."""
        with self._conn() as c:
            return len(c.execute(self._unread_query(), {"a": agent, "after": after}).fetchall())

    def fetch_unread(self, agent: str, limit: int = 30, mark: bool = True) -> list[Message]:
        with self._conn() as c:
            rows = c.execute(self._unread_query(), {"a": agent, "after": 0}).fetchall()[:limit]
            msgs = [Message(**dict(r)) for r in rows]
            if mark and msgs:
                c.execute(
                    "INSERT INTO cursors (agent, last_id) VALUES (?,?) "
                    "ON CONFLICT (agent) DO UPDATE SET last_id=excluded.last_id",
                    (agent, msgs[-1].id),
                )
        return msgs

    def history(self, agent: str, target: Optional[str] = None, limit: int = 20) -> list[Message]:
        with self._conn() as c:
            if not target:
                rows = c.execute("SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            else:
                channel, other = parse_target(target)
                if channel:
                    rows = c.execute(
                        "SELECT * FROM messages WHERE channel=? ORDER BY id DESC LIMIT ?",
                        (channel, limit),
                    ).fetchall()
                else:
                    rows = c.execute(
                        """SELECT * FROM messages
                           WHERE (sender=? AND recipient=?) OR (sender=? AND recipient=?)
                           ORDER BY id DESC LIMIT ?""",
                        (agent, other, other, agent, limit),
                    ).fetchall()
        return [Message(**dict(r)) for r in reversed(rows)]

    def recent(self, limit: int = 200, after: int = 0) -> list[Message]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM messages WHERE id > ? ORDER BY id DESC LIMIT ?", (after, limit)
            ).fetchall()
        return [Message(**dict(r)) for r in reversed(rows)]

    def max_id(self) -> int:
        with self._conn() as c:
            return int(c.execute("SELECT COALESCE(MAX(id),0) m FROM messages").fetchone()["m"])

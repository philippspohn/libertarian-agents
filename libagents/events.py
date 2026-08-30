"""Append-only per-agent event log.

Human-readable JSONL under `<profile>/history/events.jsonl`. This is what the
UI streams; it is not the model's context (that lives in
`history/conversation.json`).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class EventLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, kind: str, **fields: Any) -> dict:
        event = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            **fields,
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        return event

    def tail(self, limit: int = 200, after: Optional[int] = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                if after is not None and idx <= after:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event["seq"] = idx
                out.append(event)
        return out[-limit:]

"""Run trace: one JSON line per solved problem (for debugging and rule-compliance review)."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

log = logging.getLogger("agent")


class TraceWriter:
    def __init__(self, log_dir: Optional[str]):
        self.path: Optional[str] = None
        if not log_dir:
            return
        try:
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.path = os.path.join(log_dir, f"run-{stamp}-{os.getpid()}.jsonl")
        except OSError as exc:
            log.warning("trace disabled (%s)", exc)

    def write(self, record: dict) -> None:
        if not self.path:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as exc:
            log.warning("trace write failed (%s); disabling trace", exc)
            self.path = None

"""Minimal Ollama REST client (standard library only).

Only `POST /api/chat`, `/api/generate` (warm-up) and `/api/show` are used.  Every
request goes to the configured local host; system proxy settings are ignored so a
proxy can never route inference elsewhere.  Errors never raise to callers — they
come back as `LLMResult(error=...)`.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .config import Settings, Think

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass
class LLMResult:
    role: str
    content: str = ""
    thinking_chars: int = 0
    prompt_tokens: int = 0
    gen_tokens: int = 0
    seconds: float = 0.0
    done_reason: Optional[str] = None
    error: Optional[str] = None
    think: Any = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.content.strip()) and self.done_reason != "length"


class OllamaLLM:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.calls = 0  # inference calls (warm-up and availability checks excluded)

    def _post(self, path: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        req = urllib.request.Request(self.settings.host + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with _OPENER.open(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, role: str, prompt: str, *, think: Think, num_predict: int, timeout: float,
             seed: Optional[int] = None, system: Optional[str] = None) -> LLMResult:
        self.calls += 1
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        options: Dict[str, Any] = {"num_ctx": self.settings.num_ctx, "num_predict": num_predict}
        if seed is not None:
            options["seed"] = seed
        payload = {"model": self.settings.model, "messages": messages, "stream": False,
                   "think": think, "options": options}
        started = time.monotonic()
        result = LLMResult(role=role, think=think)
        try:
            data = self._post("/api/chat", payload, timeout=max(1.0, timeout))
            message = data.get("message") or {}
            result.content = message.get("content") or ""
            result.thinking_chars = len(message.get("thinking") or "")
            result.prompt_tokens = int(data.get("prompt_eval_count") or 0)
            result.gen_tokens = int(data.get("eval_count") or 0)
            result.done_reason = data.get("done_reason")
            if data.get("error"):
                result.error = str(data["error"])
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:300]
            result.error = f"HTTP {exc.code}: {body}"
        except Exception as exc:  # URLError, timeout, bad JSON, connection reset ...
            result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = round(time.monotonic() - started, 3)
        return result

    def warmup(self, timeout: float = 600.0) -> LLMResult:
        """Load the model with the same num_ctx the solvers use (avoids a reload on the first call)."""
        started = time.monotonic()
        result = LLMResult(role="warmup")
        try:
            self._post("/api/generate", {"model": self.settings.model, "prompt": "", "keep_alive": -1,
                                         "options": {"num_ctx": self.settings.num_ctx}}, timeout=timeout)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.seconds = round(time.monotonic() - started, 3)
        return result

    def model_available(self, timeout: float = 15.0) -> bool:
        try:
            self._post("/api/show", {"model": self.settings.model}, timeout=timeout)
            return True
        except Exception:
            return False

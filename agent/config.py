"""Runtime settings: defaults < environment variables < command-line overrides."""
from __future__ import annotations

import logging
import os
import urllib.parse
from dataclasses import dataclass, fields, replace
from typing import Mapping, Optional, Union

Think = Union[bool, str]

log = logging.getLogger("agent.config")

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Settings:
    model: str = "qwen3.8:27b"
    host: str = "http://127.0.0.1:11434"
    num_ctx: int = 16384
    # Thinking effort per role (Ollama `think`: False/True or "low"/"medium"/"high"/"xhigh").
    think_reasoner: Think = "medium"
    think_programmer: Think = "low"
    think_verifier: Think = "medium"
    think_tiebreak: Think = "medium"
    think_formatter: Think = False
    num_predict_reasoner: int = 12000
    num_predict_programmer: int = 8000
    num_predict_verifier: int = 12000
    num_predict_tiebreak: int = 16000
    num_predict_formatter: int = 256
    time_budget: float = 300.0          # seconds per problem
    call_timeout_cap: float = 240.0     # seconds per LLM call (absolute cap)
    max_call_share: float = 0.4         # one call may use at most this share of the problem budget
    max_calls: int = 6                  # LLM calls per problem
    code_timeout: float = 20.0          # seconds per sandboxed program
    max_program_answers: int = 10
    seed: int = 42
    reserve_seconds: float = 5.0
    min_stage_seconds: float = 20.0
    log_dir: Optional[str] = os.path.join(_PKG_ROOT, "logs")
    warmup: bool = True


_ENV_MAP = {
    "AGENT_MODEL": "model",
    "OLLAMA_HOST": "host",
    "AGENT_NUM_CTX": "num_ctx",
    "AGENT_THINK_REASONER": "think_reasoner",
    "AGENT_THINK_PROGRAMMER": "think_programmer",
    "AGENT_THINK_VERIFIER": "think_verifier",
    "AGENT_THINK_TIEBREAK": "think_tiebreak",
    "AGENT_TIME_BUDGET": "time_budget",
    "AGENT_CALL_TIMEOUT": "call_timeout_cap",
    "AGENT_MAX_CALL_SHARE": "max_call_share",
    "AGENT_MAX_CALLS": "max_calls",
    "AGENT_CODE_TIMEOUT": "code_timeout",
    "AGENT_SEED": "seed",
    "AGENT_LOG_DIR": "log_dir",
    "AGENT_WARMUP": "warmup",
}

_TYPES = {f.name: f.type for f in fields(Settings)}


def normalize_host(value: str) -> str:
    """Interpret OLLAMA_HOST like Ollama does: no scheme -> http and port 11434; 0.0.0.0/:: -> loopback."""
    value = value.strip().rstrip("/")
    has_scheme = "://" in value
    parts = urllib.parse.urlsplit(value if has_scheme else "http://" + value)
    host = parts.hostname or ""
    try:
        port = parts.port
    except ValueError:
        port = None
    if host in ("", "0.0.0.0", "::"):
        host = "127.0.0.1"
    if port is None and not has_scheme:
        port = 11434
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return f"{parts.scheme}://{netloc}{parts.path.rstrip('/')}"


def parse_think(value: str) -> Think:
    low = value.strip().lower()
    if low in ("false", "0", "no", "off", "none"):
        return False
    if low in ("true", "1", "yes", "on"):
        return True
    return low


def _coerce(name: str, raw: str):
    kind = _TYPES[name]
    if name == "host":
        return normalize_host(raw)
    if name.startswith("think_"):
        return parse_think(raw)
    if name == "warmup":
        return raw.strip().lower() not in ("0", "false", "no", "off")
    if name == "log_dir":
        return raw if raw.strip() and raw.strip().lower() != "none" else None
    if kind in ("int", int):
        return int(raw)
    if kind in ("float", float):
        return float(raw)
    return raw


def load_settings(env: Optional[Mapping[str, str]] = None, **overrides) -> Settings:
    """Build Settings from defaults, then `env` (default: os.environ), then non-None overrides."""
    env = os.environ if env is None else env
    values = {}
    for key, name in _ENV_MAP.items():
        raw = env.get(key)
        if raw is None or raw == "":
            continue
        try:
            values[name] = _coerce(name, raw)
        except ValueError:  # a typo in the environment must not stop the run before output exists
            log.warning("ignoring %s=%r (not a valid value); using the default %r",
                        key, raw, getattr(Settings, name))
    for name, val in overrides.items():
        if val is None:
            continue
        values[name] = normalize_host(val) if name == "host" else val
    return replace(Settings(), **values)

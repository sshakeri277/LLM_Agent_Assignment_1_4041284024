"""Reading problems and writing answers (shape-preserving, atomic)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, List, Union


class InputError(Exception):
    """The input file is missing, unreadable, or not a supported JSON shape."""


@dataclass
class Task:
    id: Union[str, int]
    problem: str


@dataclass
class InputDoc:
    tasks: List[Task]
    shape: str  # "object" (single problem) or "list"


def _task(item: Any, index: int) -> Task:
    if not isinstance(item, dict):
        return Task(id=index, problem=str(item) if item is not None else "")
    problem = item.get("problem")
    return Task(id=item.get("id", index), problem=problem if isinstance(problem, str) else "")


def load_input(path: str) -> InputDoc:
    try:
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read input {path!r}: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("problems"), list):
        return InputDoc([_task(x, i) for i, x in enumerate(data["problems"])], "list")
    if isinstance(data, dict):
        return InputDoc([_task(data, 0)], "object")
    if isinstance(data, list):
        return InputDoc([_task(x, i) for i, x in enumerate(data)], "list")
    raise InputError(f"unsupported input JSON type: {type(data).__name__}")


def render_output(doc: InputDoc, answers: List[str]) -> Any:
    rows = [{"id": t.id, "answer": a} for t, a in zip(doc.tasks, answers)]
    if doc.shape == "object":
        return rows[0] if rows else {"id": None, "answer": ""}
    return rows


def write_json_atomic(path: str, obj: Any) -> None:
    target_dir = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".out-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

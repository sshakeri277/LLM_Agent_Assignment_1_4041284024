"""Run model-written Python programs in a restricted child process.

Defence in depth (this is a homework sandbox, not a security boundary):
1. AST gate: only allow-listed stdlib imports; no file/exec/eval/input builtins; no getattr-style
   reflection; no underscore attributes (except a few harmless dunders) and no module-valued attributes
   such as ``.os`` that lead out of the allow-list.
2. Child process `python -I -X utf8` in a fresh temporary directory with a minimal env.
3. Wall-clock timeout; on POSIX also CPU, memory and output-file-size rlimits.
Output protocol: each line ``ANSWER: <x>`` is one valid answer; otherwise the last
non-empty line is taken as the single answer.
"""
from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional

ALLOWED_MODULES = frozenset({
    "math", "itertools", "functools", "collections", "fractions", "decimal", "heapq", "bisect",
    "re", "string", "statistics", "datetime", "calendar", "operator", "copy", "typing",
    "dataclasses", "enum", "json", "sys", "cmath", "numbers", "random",
})
FORBIDDEN_NAMES = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
                             "globals", "locals", "vars", "memoryview", "getattr", "setattr", "delattr",
                             "attrgetter"})
# Public attributes that reach the OS through an allowed module (e.g. dataclasses.inspect.os).
FORBIDDEN_ATTRS = frozenset({"modules", "attrgetter", "os", "posix", "nt", "subprocess", "inspect", "builtins",
                             "importlib", "shutil", "socket", "ctypes", "pathlib", "io", "sysconfig"})
# Attributes starting with "_" are rejected (random._os, sys._getframe, x.__dict__, ...) except these.
ALLOWED_PRIVATE_ATTRS = frozenset({"__init__", "__name__", "__qualname__", "__doc__", "__class__",
                                   "__post_init__", "_replace", "_asdict", "_fields", "_make"})
_MAX_STDOUT_READ = 1_000_000
_STDOUT_KEEP = 20_000
_ANSWER_LINE = re.compile(r"^\s*ANSWER\s*:\s*(.*?)\s*$")
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
_PY_TAGS = ("python", "python3", "py")


@dataclass
class ProgramRun:
    ok: bool
    answers: List[str] = field(default_factory=list)
    error: Optional[str] = None
    stdout: str = ""
    code: str = ""
    seconds: float = 0.0


def extract_code(text: str) -> Optional[str]:
    """Last fenced code block; else the whole text if it is a Python program that prints."""
    if not text:
        return None
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(text) if body.strip()]
    tagged = [body for lang, body in blocks if lang in _PY_TAGS]
    if tagged:
        return tagged[-1]
    for _lang, body in reversed(blocks):  # untagged: the last block that is a printing program
        if "print(" in body:
            try:
                ast.parse(body)
                return body
            except SyntaxError:
                continue
    if blocks:
        return blocks[-1][1]
    if "print(" in text:
        try:
            ast.parse(text)
            return text
        except SyntaxError:
            return None
    return None


def check_code(code: str) -> Optional[str]:
    """Return a human-readable violation, or None if the program passes the AST gate."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"syntax error: {exc.msg} (line {exc.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                return f"forbidden import: {node.module or '.'}"
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"forbidden name: {node.id}"
        elif isinstance(node, ast.Attribute) and (
                node.attr in FORBIDDEN_ATTRS
                or (node.attr.startswith("_") and node.attr not in ALLOWED_PRIVATE_ATTRS)):
            return f"forbidden attribute: {node.attr}"
    return None


def parse_answers(stdout: str) -> List[str]:
    lines = stdout.splitlines()
    tagged = []
    for line in lines:
        m = _ANSWER_LINE.match(line)
        if m and m.group(1):
            tagged.append(m.group(1))
    if tagged:
        return list(dict.fromkeys(tagged))
    for line in reversed(lines):
        if line.strip():
            return [line.strip()]
    return []


def _limit_resources(timeout: float):
    def apply():  # runs in the child (POSIX only)
        import resource
        limits = [
            (resource.RLIMIT_CPU, int(timeout) + 2),
            (resource.RLIMIT_AS, 2 * 1024 ** 3),
            (resource.RLIMIT_FSIZE, 10 * 1024 ** 2),
        ]
        for res, value in limits:
            try:
                resource.setrlimit(res, (value, value))
            except (ValueError, OSError):
                pass
    return apply


def run_program(code: str, timeout: float, max_answers: int) -> ProgramRun:
    """Run `code` in a child process. Never raises: every failure becomes ProgramRun(ok=False)."""
    started = time.monotonic()

    def done(ok: bool, error: Optional[str] = None, stdout: str = "", answers=None) -> ProgramRun:
        return ProgramRun(ok=ok, answers=answers or [], error=error, stdout=stdout[:_STDOUT_KEEP],
                          code=code, seconds=round(time.monotonic() - started, 3))

    try:
        violation = check_code(code)
        if violation:
            return done(False, f"rejected: {violation}")
        return _execute(code, timeout, max_answers, done)
    except Exception as exc:  # MemoryError, PermissionError on Windows temp files, ...
        return done(False, f"sandbox error: {type(exc).__name__}: {exc}")


def _execute(code: str, timeout: float, max_answers: int, done) -> ProgramRun:
    env = {"PATH": os.environ.get("PATH", "")}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    tmp = tempfile.mkdtemp(prefix="agent-prog-")
    try:
        prog = os.path.join(tmp, "prog.py")
        out_path, err_path = os.path.join(tmp, "stdout.txt"), os.path.join(tmp, "stderr.txt")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(code)
        kwargs = {}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources(timeout)
        try:
            # stdout/stderr go to files (bounded by RLIMIT_FSIZE on POSIX), never to memory.
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                proc = subprocess.run([sys.executable, "-I", "-X", "utf8", prog], cwd=tmp, env=env,
                                      stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, **kwargs)
        except subprocess.TimeoutExpired:
            return done(False, f"timeout after {timeout:.0f}s")
        except OSError as exc:
            return done(False, f"could not start python: {exc}")
        with open(out_path, "rb") as f:
            stdout = f.read(_MAX_STDOUT_READ).decode("utf-8", errors="replace")
        with open(err_path, "rb") as f:
            f.seek(max(0, os.path.getsize(err_path) - 4000))
            stderr = f.read().decode("utf-8", errors="replace")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if proc.returncode != 0:
        return done(False, f"exit code {proc.returncode}: {stderr[-600:].strip()}", stdout)
    answers = parse_answers(stdout)
    if not answers:
        return done(False, "no output (program printed nothing)", stdout)
    if len(answers) > max_answers:
        return done(False, f"too many answers ({len(answers)}): the constraints are not encoded tightly enough",
                    stdout)
    return done(True, None, stdout, answers)

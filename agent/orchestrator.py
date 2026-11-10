"""The agent policy: two independent solvers, consensus, verification, tie-break vote.

    Reasoner (free reasoning) ─┐
                               ├─ agree? ─ yes ─► answer           (typical: 2 LLM calls)
    Programmer (code+sandbox) ─┘     │ no
                                     ▼
                          Verifier (judge with program shown) ─ matches a candidate? ─► answer
                                     │ no
                                     ▼
                          Tie-break reasoner (more thinking) ─► majority vote

    Final check: an answer that does not fit the required format goes to the formatter; if it still
    does not fit and some other candidate does, that candidate is written instead.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from . import prompts
from .answer_format import (FormatSpec, canonical, clean, extract_final_answer, extract_spec, normalize,
                            to_ascii_digits)
from .config import Settings
from .llm import LLMResult
from .sandbox import ProgramRun, extract_code, run_program

# Priority when votes tie and for the fallback (lower = preferred).
_PRIORITY = {"verifier": 0, "program": 1, "tiebreak": 2, "reasoner": 3}


@dataclass
class Candidate:
    source: str
    raw: str
    value: str
    valid: bool

    @property
    def key(self) -> str:
        return canonical(self.value)


@dataclass
class Decision:
    answer: str
    path: str
    calls: int
    seconds: float
    ambiguous: bool = False
    candidates: List[Candidate] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    spec_kind: str = ""


class Budget:
    def __init__(self, time_budget: float, max_calls: int, call_timeout_cap: float, reserve_seconds: float,
                 min_stage_seconds: float, clock: Callable[[], float], max_call_share: float = 1.0):
        self.clock = clock
        self.time_budget = time_budget
        self.max_call_share = max_call_share
        self.deadline = clock() + time_budget
        self.max_calls = max_calls
        self.call_timeout_cap = call_timeout_cap
        self.reserve = reserve_seconds
        self.min_stage = min_stage_seconds
        self.calls = 0

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def can_run(self, optional: bool = True) -> bool:
        """Mandatory stages need more than the reserve; optional stages need `min_stage` seconds."""
        if self.calls >= self.max_calls:
            return False
        need = max(self.min_stage, self.reserve) if optional else self.reserve
        return self.remaining() > need

    def call_timeout(self) -> float:
        share_cap = self.max_call_share * self.time_budget
        return max(1.0, min(self.call_timeout_cap, share_cap, self.remaining() - self.reserve))

    def note_call(self) -> None:
        self.calls += 1


class Orchestrator:
    def __init__(self, llm, settings: Settings, runner: Callable[..., ProgramRun] = run_program,
                 clock: Callable[[], float] = time.monotonic):
        self.llm = llm
        self.s = settings
        self.runner = runner
        self.clock = clock

    # ------------------------------------------------------------ helpers
    def _candidate(self, source: str, raw: Optional[str], spec: FormatSpec) -> Optional[Candidate]:
        if not raw:
            return None
        value = normalize(raw, spec)
        if value is not None:
            return Candidate(source, raw, value, True)
        cleaned = clean(raw)
        return Candidate(source, raw, cleaned, False) if cleaned else None

    def solve(self, problem: str) -> Decision:
        started = self.clock()
        s = self.s
        spec = extract_spec(problem)
        budget = Budget(s.time_budget, s.max_calls, s.call_timeout_cap, s.reserve_seconds, s.min_stage_seconds,
                        self.clock, s.max_call_share)
        events: List[Dict] = []
        state = {"any_reply": False}

        def call(role: str, prompt: str, think, num_predict: int, seed: int, optional: bool) -> Optional[LLMResult]:
            if not budget.can_run(optional):
                events.append({"role": role, "skipped": True, "remaining": round(budget.remaining(), 1)})
                return None
            result = self.llm.chat(role, prompt, think=think, num_predict=num_predict,
                                   timeout=budget.call_timeout(), seed=seed, system=prompts.SYSTEM)
            budget.note_call()
            if result.error is None:
                state["any_reply"] = True
            events.append({"role": role, "ok": result.ok, "seconds": result.seconds, "think": result.think,
                           "gen_tokens": result.gen_tokens, "done_reason": result.done_reason,
                           "error": result.error})
            return result

        def answer_of(result: Optional[LLMResult], source: str) -> Optional[Candidate]:
            if result is None or not result.ok:
                return None
            cand = self._candidate(source, extract_final_answer(result.content), spec)
            events[-1]["answer"] = cand.value if cand else None
            return cand

        def finish(winner: Optional[Candidate], path: str, ambiguous: bool = False) -> Decision:
            answer = ""
            if winner is not None:
                answer = winner.value
                if not winner.valid:
                    fr = call("formatter", prompts.formatter_prompt(problem, spec, winner.raw), s.think_formatter,
                              s.num_predict_formatter, s.seed, optional=False)
                    fixed = normalize(fr.content, spec) if fr is not None and fr.ok else None
                    if fixed is not None and _numbers(winner.raw) and not _numbers(fixed) <= _numbers(winner.raw):
                        events.append({"role": "formatter", "rejected": fixed})  # it changed the numbers
                        fixed = None
                    if fixed is not None:
                        answer = fixed
                    else:  # final check: never write an invalid answer while a valid candidate exists
                        backup = _best_valid(candidates)
                        if backup is not None:
                            events.append({"role": "final-check", "invalid": winner.value, "used": backup.value,
                                           "source": backup.source})
                            answer, path = backup.value, "format-fallback"
            elif path != "empty":
                path = "empty"
            return Decision(answer=answer, path=path, calls=budget.calls,
                            seconds=round(self.clock() - started, 3), ambiguous=ambiguous,
                            candidates=candidates, events=events, spec_kind=spec.kind)

        candidates: List[Candidate] = []

        # 1) Reasoner -------------------------------------------------------
        r1 = call("reasoner", prompts.reasoner_prompt(problem, spec), s.think_reasoner, s.num_predict_reasoner,
                  s.seed, optional=False)
        a = answer_of(r1, "reasoner")
        if a:
            candidates.append(a)

        # 2) Programmer (+ one repair) --------------------------------------
        run: Optional[ProgramRun] = None
        r2 = call("programmer", prompts.programmer_prompt(problem, spec), s.think_programmer,
                  s.num_predict_programmer, s.seed, optional=False)
        if r2 is not None and r2.error is None:
            run = self._run_reply(r2, budget)
            events[-1]["program"] = _program_event(run)
            if not run.ok:
                previous = run.code or r2.content[:3000]
                failure = run.error or "no output"
                r3 = call("repair", prompts.repair_prompt(problem, spec, previous, failure), s.think_programmer,
                          s.num_predict_programmer, s.seed, optional=True)
                if r3 is not None and r3.error is None:
                    run = self._run_reply(r3, budget)
                    events[-1]["program"] = _program_event(run)
        program: List[Candidate] = []
        if run is not None and run.ok:
            seen = set()
            for ans in run.answers:
                cand = self._candidate("program", ans, spec)
                if cand and cand.key not in seen:
                    seen.add(cand.key)
                    program.append(cand)
        candidates.extend(program)
        program_keys = {c.key for c in program}
        unique_program = program[0] if len(program) == 1 else None

        # 3) Consensus ------------------------------------------------------
        if a and a.key in program_keys:
            winner = a
            if not a.valid:
                winner = next((c for c in program if c.key == a.key and c.valid), a)
            ambiguous = len(program) > 1
            return finish(winner, "consensus-ambiguous" if ambiguous else "consensus", ambiguous)

        # 4) Verifier -------------------------------------------------------
        v: Optional[Candidate] = None
        if (a or program) and state["any_reply"]:
            listing = []
            if a:
                listing.append(("step-by-step reasoning", a.value))
            label = ("computer program" if len(program) <= 1
                     else "computer program — several answers satisfied its checks")
            listing.extend((label, c.value) for c in program)
            rv = call("verifier", prompts.verifier_prompt(problem, spec, listing, run.code if program else None),
                      s.think_verifier, s.num_predict_verifier, s.seed, optional=True)
            v = answer_of(rv, "verifier")
            if v:
                candidates.append(v)
                if v.key in program_keys or (a is not None and v.key == a.key):
                    return finish(v, "verified")

        # 5) Tie-break vote -------------------------------------------------
        t: Optional[Candidate] = None
        tiebreak_ran = False
        if state["any_reply"]:
            rt = call("tiebreak", prompts.reasoner_prompt(problem, spec), s.think_tiebreak, s.num_predict_tiebreak,
                      s.seed + 1, optional=True)
            tiebreak_ran = rt is not None
            t = answer_of(rt, "tiebreak")
            if t:
                candidates.append(t)

        voters = [c for c in (a, v, t, unique_program) if c is not None]
        if not voters:
            return finish(program[0] if program else None, "fallback")
        counts: Dict[str, float] = {}
        for c in voters:
            counts[c.key] = counts.get(c.key, 0) + 1
        if len(program) > 1:  # a program with several valid answers backs each of them, by less than a vote
            for key in counts:
                if key in program_keys:
                    counts[key] += 0.5
        winner = min(voters, key=lambda c: (-counts[c.key], _PRIORITY[c.source]))
        return finish(winner, "vote" if tiebreak_ran else "fallback")

    def _run_reply(self, reply: LLMResult, budget: Budget) -> ProgramRun:
        code = extract_code(reply.content)
        if not code:
            return ProgramRun(ok=False, error="the reply contained no python code block", code="")
        timeout = max(1.0, min(self.s.code_timeout, budget.remaining() - budget.reserve))
        try:
            return self.runner(code, timeout, self.s.max_program_answers)
        except Exception as exc:  # the sandbox must never take the reasoner's answer down with it
            return ProgramRun(ok=False, error=f"sandbox crashed: {type(exc).__name__}: {exc}", code=code)


def _best_valid(candidates: List[Candidate]) -> Optional[Candidate]:
    """The valid candidate with the most support (ties: the usual source priority), or None."""
    valid = [c for c in candidates if c.valid]
    if not valid:
        return None
    support: Dict[str, int] = {}
    for c in candidates:
        support[c.key] = support.get(c.key, 0) + 1
    return min(valid, key=lambda c: (-support[c.key], _PRIORITY[c.source]))


def _program_event(run: ProgramRun) -> Dict:
    """What the trace keeps about a program run (the code is kept so a reviewer can audit it)."""
    return {"ok": run.ok, "answers": run.answers, "error": run.error, "code": (run.code or "")[:_MAX_CODE_IN_TRACE]}


_MAX_CODE_IN_TRACE = 6000


def _numbers(text: str) -> set:
    return set(re.findall(r"-?\d+(?:\.\d+)?", to_ascii_digits(text or "")))

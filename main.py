#!/usr/bin/env python3
"""Entry point:  python main.py --input input.json --output output.json

Solves every problem in the input with a local Ollama model (default qwen3.8:27b) and
writes {"id", "answer"} records.  The output file is valid JSON at every moment of the
run: it is pre-filled with empty answers and rewritten atomically after each problem.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import asdict

from agent.config import load_settings
from agent.io_utils import InputError, load_input, render_output, write_json_atomic
from agent.llm import OllamaLLM
from agent.orchestrator import Decision, Orchestrator
from agent.trace import TraceWriter

log = logging.getLogger("agent")


def parse_args(argv):
    p = argparse.ArgumentParser(description="Local reasoning agent (Ollama).")
    p.add_argument("--input", required=True, help="input JSON: one {id, problem} object or a list of them")
    p.add_argument("--output", required=True, help="output JSON path")
    p.add_argument("--model", help="Ollama model name (default qwen3.8:27b, env AGENT_MODEL)")
    p.add_argument("--host", help="Ollama host (default env OLLAMA_HOST or http://127.0.0.1:11434)")
    p.add_argument("--time-budget", type=float, help="seconds per problem (default 300, env AGENT_TIME_BUDGET)")
    p.add_argument("--max-calls", type=int, help="max LLM calls per problem (default 6, env AGENT_MAX_CALLS)")
    p.add_argument("--log-dir", help="directory for run traces (default ./logs next to main.py; 'none' disables)")
    p.add_argument("--no-warmup", action="store_true", help="skip loading the model before the first problem")
    p.add_argument("--verbose", action="store_true", help="debug logging on stderr")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s", force=True)
    log_dir = "" if (args.log_dir or "").lower() == "none" else args.log_dir  # "" disables the trace
    settings = load_settings(None, model=args.model, host=args.host, time_budget=args.time_budget,
                             max_calls=args.max_calls, log_dir=log_dir,
                             warmup=False if args.no_warmup else None)
    try:
        doc = load_input(args.input)
    except InputError as exc:
        log.error("%s", exc)
        write_json_atomic(args.output, [])
        return 1

    answers = [""] * len(doc.tasks)
    write_json_atomic(args.output, render_output(doc, answers))
    log.info("model=%s host=%s problems=%d budget=%.0fs/problem max_calls=%d", settings.model, settings.host,
             len(doc.tasks), settings.time_budget, settings.max_calls)

    llm = OllamaLLM(settings)
    trace = TraceWriter(settings.log_dir)
    process_started = time.monotonic()
    warmup_seconds = None  # model-load time; spent before (outside) the per-problem budgets
    if settings.warmup and any(t.problem.strip() for t in doc.tasks):
        if llm.model_available():
            w = llm.warmup()
            warmup_seconds = float(w.seconds)
            log.info("warm-up %s in %.1fs", "ok" if w.error is None else f"failed ({w.error})", w.seconds)
        else:
            log.warning("model %s is not available at %s (is Ollama running and the model pulled?)",
                        settings.model, settings.host)

    orchestrator = Orchestrator(llm, settings)
    run_started = time.monotonic()
    total_calls = 0
    for i, task in enumerate(doc.tasks):
        try:
            if task.problem.strip():
                decision = orchestrator.solve(task.problem)
            else:
                decision = Decision(answer="", path="empty", calls=0, seconds=0.0)
        except Exception as exc:  # a bug must never cost the other problems
            log.exception("problem %s crashed", task.id)
            decision = Decision(answer="", path="error", calls=0, seconds=0.0, events=[{"error": repr(exc)}])
        answers[i] = decision.answer
        total_calls += decision.calls
        write_json_atomic(args.output, render_output(doc, answers))
        trace.write({"id": task.id, "spec": decision.spec_kind, "path": decision.path, "answer": decision.answer,
                     "calls": decision.calls, "seconds": decision.seconds, "ambiguous": decision.ambiguous,
                     "candidates": [asdict(c) for c in decision.candidates], "events": decision.events})
        log.info("[%d/%d] id=%s answer=%r path=%s calls=%d %.1fs", i + 1, len(doc.tasks), task.id,
                 decision.answer, decision.path, decision.calls, decision.seconds)

    elapsed = time.monotonic() - run_started
    total = time.monotonic() - process_started
    trace.write({"summary": True, "problems": len(doc.tasks), "llm_calls": total_calls,
                 "seconds": round(elapsed, 1), "warmup_seconds": warmup_seconds, "total_seconds": round(total, 1),
                 "model": settings.model})
    log.info("done: %d problems, %d LLM calls, %.1fs solving + %.1fs warm-up; output=%s trace=%s", len(doc.tasks),
             total_calls, elapsed, warmup_seconds or 0.0, args.output, trace.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

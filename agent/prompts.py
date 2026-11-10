"""Prompt templates. Instructions are in English; the problem text is inserted verbatim."""
from __future__ import annotations

from typing import List, Optional, Tuple

from .answer_format import FormatSpec

SYSTEM = ("You are a meticulous solver of logic, arithmetic and constraint puzzles. Problems may be written "
          "in Persian (Farsi). Read every condition literally and precisely, use every given fact, and check "
          "your result against all conditions before answering.")

_MAX_CODE_IN_PROMPT = 6000

_PROGRAM_RULES = """\
- Model every entity, variable and constraint explicitly. Translate each condition of the problem into its own
  check, with a short comment saying which condition it encodes.
- Prefer exhaustive search over all possibilities (itertools.permutations / itertools.product, loops over
  candidate values, BFS for paths) instead of clever shortcuts. Use fractions.Fraction for exact arithmetic
  with percentages or ratios.
- Collect ALL solutions that satisfy ALL constraints. Print each DISTINCT final answer on its own line as:
  ANSWER: <answer in the required format>
- Print nothing else. Use only the Python standard library. No input(), no files, no network.
- Do not access attributes whose name starts with an underscore (except __init__ / __name__) and do not use
  getattr, setattr or delattr: programs that do are rejected."""


def _problem_block(problem: str) -> str:
    return f"PROBLEM:\n<<<\n{problem}\n>>>"


def _final_line_rule(spec: FormatSpec) -> str:
    return (f"Required answer format: {spec.description}\n"
            "End your reply with exactly one final line:\n"
            "FINAL ANSWER: <answer in the required format>")


def reasoner_prompt(problem: str, spec: FormatSpec) -> str:
    return f"""Solve the following problem.

{_problem_block(problem)}

Identify every quantity and every condition, solve step by step, then verify your result against each
condition of the problem. If the problem seems to allow several answers, re-read the wording and choose the
one the author most likely intended. Keep the visible reply short.

{_final_line_rule(spec)}"""


def programmer_prompt(problem: str, spec: FormatSpec) -> str:
    return f"""Solve the following problem by writing a Python 3 program.

{_problem_block(problem)}

Write a complete, self-contained Python 3 program that computes the answer by exhaustive search or exact
computation:
{_PROGRAM_RULES}

Required answer format: {spec.description}

Reply with only one ```python code block."""


def repair_prompt(problem: str, spec: FormatSpec, code: str, failure: str) -> str:
    return f"""Your previous program for the problem below failed.

{_problem_block(problem)}

PREVIOUS PROGRAM:
```python
{code[:_MAX_CODE_IN_PROMPT]}
```

FAILURE:
{failure}

Write a corrected, complete program. If it found zero or too many solutions, some condition is probably
mis-encoded: re-read every condition of the problem. Rules:
{_PROGRAM_RULES}

Required answer format: {spec.description}

Reply with only one ```python code block."""


def verifier_prompt(problem: str, spec: FormatSpec, candidates: List[Tuple[str, str]],
                    program_code: Optional[str]) -> str:
    listing = "\n".join(f"{i}. ({label}) {answer}" for i, (label, answer) in enumerate(candidates, 1))
    program = ""
    if program_code:
        program = (f"\nPROGRAM USED BY THE COMPUTER-PROGRAM ATTEMPT (programs often mis-encode a condition, so treat "
                   f"its output as a claim to check, not as proof):\n```python\n{program_code[:_MAX_CODE_IN_PROMPT]}\n```\n")
    return f"""Independent attempts to solve the problem below did not agree. Act as a careful, skeptical judge:
neither attempt is trusted by default.

{_problem_block(problem)}

CANDIDATE ANSWERS:
{listing}
{program}
Instructions:
1. List every condition of the problem, numbered, in your own words, exactly as stated.
2. For EACH candidate, build the complete solution it implies (the full arrangement, all values, or the path)
   and test that solution against every numbered condition, writing PASS or FAIL for each one. A candidate with
   any FAIL is wrong. For "how many" or "minimum/maximum" questions, also check that nothing was missed and
   that no better value exists.
3. If no candidate passes, solve the problem yourself and check your answer the same way.
4. If several answers pass every condition, choose the one the problem's author most likely intended.

{_final_line_rule(spec)}"""


def formatter_prompt(problem: str, spec: FormatSpec, answer: str) -> str:
    return f"""Rewrite a proposed answer so that it follows the required answer format exactly.

{_problem_block(problem)}

PROPOSED ANSWER: {answer}

Required answer format: {spec.description}
Reply with only the reformatted answer and nothing else. Do not change its meaning."""

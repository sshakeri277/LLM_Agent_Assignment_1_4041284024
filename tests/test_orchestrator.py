import unittest
from collections import defaultdict, deque

from agent.config import load_settings
from agent.llm import LLMResult
from agent.orchestrator import Orchestrator
from agent.sandbox import ProgramRun

INT_PROBLEM = "یک انبار ۳ قفسه دارد و هر قفسه ۴ جعبه. چند جعبه وجود دارد؟\nفقط عدد صحیح را برگردانید."
OPT_PROBLEM = "رنگ جعبه چیست؟\nفقط یکی از این سه عبارت را برگردانید:\nRed\nGreen\nBlue"
CODE_REPLY = "```python\nprint('ANSWER: x')\n```"


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeLLM:
    """Pops scripted replies per role; each call advances the fake clock."""

    def __init__(self, clock, script, seconds_per_call=10.0, jump_after_first=None):
        self.clock = clock
        self.script = {k: deque(v) for k, v in script.items()}
        self.roles = []
        self.calls = 0
        self.seconds_per_call = seconds_per_call
        self.jump_after_first = jump_after_first
        self.timeouts = []
        self.prompts = []

    def chat(self, role, prompt, *, think, num_predict, timeout, seed=None, system=None):
        self.calls += 1
        self.roles.append(role)
        self.prompts.append(prompt)
        self.timeouts.append(timeout)
        self.clock.t += self.seconds_per_call
        if self.jump_after_first and self.calls == 1:
            self.clock.t += self.jump_after_first
        queue = self.script.get(role)
        item = queue.popleft() if queue else LLMResult(role=role, error="unscripted")
        if isinstance(item, str):
            return LLMResult(role=role, content=item, done_reason="stop")
        return item


class FakeRunner:
    def __init__(self, runs):
        self.runs = deque(runs)
        self.codes = []
        self.timeouts = []
        self.raise_with = None

    def __call__(self, code, timeout, max_answers):
        self.codes.append(code)
        self.timeouts.append(timeout)
        if self.raise_with:
            raise self.raise_with
        return self.runs.popleft() if self.runs else ProgramRun(ok=False, error="unscripted", code=code)


def ok_run(*answers):
    return ProgramRun(ok=True, answers=list(answers), code="print('ANSWER: x')")


def bad_run(error="exit code 1: NameError"):
    return ProgramRun(ok=False, error=error, code="print(x)")


def make(script, runs, **overrides):
    clock = FakeClock()
    settings = load_settings({}, log_dir=None, **overrides)
    llm = FakeLLM(clock, script)
    runner = FakeRunner(runs)
    return Orchestrator(llm, settings, runner=runner, clock=clock), llm, runner, clock


class OrchestratorTests(unittest.TestCase):
    def test_consensus_two_calls(self):
        orch, llm, runner, _ = make({"reasoner": ["Done.\nFINAL ANSWER: 8"], "programmer": [CODE_REPLY]},
                                    [ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "consensus", 2))
        self.assertEqual(llm.roles, ["reasoner", "programmer"])
        self.assertFalse(d.ambiguous)
        self.assertEqual(d.spec_kind, "integer")

    def test_ambiguous_prefers_reasoner_in_set(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 5"], "programmer": [CODE_REPLY]}, [ok_run("4", "5")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls, d.ambiguous), ("5", "consensus-ambiguous", 2, True))

    def test_disagreement_verifier_resolves(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY],
                                "verifier": ["checked\nFINAL ANSWER: 8"]}, [ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "verified", 3))
        self.assertEqual(llm.roles, ["reasoner", "programmer", "verifier"])

    def test_verifier_is_not_told_the_program_is_authoritative(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY],
                                "verifier": ["FINAL ANSWER: 8"]}, [ok_run("8")])
        orch.solve(INT_PROBLEM)
        verifier_prompt = llm.prompts[llm.roles.index("verifier")]
        self.assertNotIn("enumerated all possibilities", verifier_prompt)

    def test_trace_keeps_the_program_code(self):
        orch, _, _, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY]}, [ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        program_event = next(e for e in d.events if e.get("role") == "programmer")
        self.assertEqual(program_event["program"]["code"], "print('ANSWER: x')")

    def test_program_crash_then_repair(self):
        orch, llm, runner, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY],
                                     "repair": [CODE_REPLY]}, [bad_run(), ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "consensus", 3))
        self.assertIn("repair", llm.roles)
        self.assertEqual(len(runner.codes), 2)

    def test_no_code_block_triggers_repair(self):
        orch, llm, runner, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": ["I would loop over values."],
                                     "repair": [CODE_REPLY]}, [ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "consensus", 3))
        self.assertEqual(len(runner.codes), 1)

    def test_vote_after_verifier_new_answer(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY],
                                "verifier": ["FINAL ANSWER: 9"], "tiebreak": ["FINAL ANSWER: 8"]}, [ok_run("8")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "vote", 4))
        self.assertEqual(llm.roles, ["reasoner", "programmer", "verifier", "tiebreak"])

    def test_vote_counts_multi_answer_program_support(self):
        # M4: reasoner 7, program {4, 5}, verifier 9, tie-break 5 -> the program backs 5.
        orch, _, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY],
                              "verifier": ["FINAL ANSWER: 9"], "tiebreak": ["FINAL ANSWER: 5"]}, [ok_run("4", "5")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("5", "vote", 4))

    def test_vote_tie_prefers_verifier(self):
        orch, _, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY],
                              "verifier": ["FINAL ANSWER: 9"], "tiebreak": ["FINAL ANSWER: 10"]}, [ok_run("8")])
        self.assertEqual(orch.solve(INT_PROBLEM).answer, "9")

    def test_budget_exhausted_fallback(self):
        clock = FakeClock()
        settings = load_settings({}, log_dir=None)
        llm = FakeLLM(clock, {"reasoner": ["FINAL ANSWER: 7"]}, jump_after_first=10_000)
        orch = Orchestrator(llm, settings, runner=FakeRunner([]), clock=clock)
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("7", "fallback", 1))

    def test_call_timeout_respects_remaining_budget(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY]}, [ok_run("8")],
                               time_budget=100.0, max_call_share=1.0)
        orch.solve(INT_PROBLEM)
        self.assertEqual(llm.timeouts[0], 95.0)   # min(cap 240, 100 - reserve 5)
        self.assertEqual(llm.timeouts[1], 85.0)   # 10 s spent by the first call

    def test_call_timeout_capped_at_share_of_budget(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY]}, [ok_run("8")])
        orch.solve(INT_PROBLEM)
        self.assertEqual(llm.timeouts[0], 120.0)  # 0.4 * 300 s: one runaway call cannot eat the budget

    def test_code_timeout_charged_to_budget(self):
        orch, llm, runner, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY]}, [ok_run("8")],
                                    time_budget=30.0, max_call_share=1.0)
        orch.solve(INT_PROBLEM)
        self.assertEqual(runner.timeouts, [5.0])  # 30 - 2*10 s spent - 5 s reserve

    def test_runner_exception_does_not_lose_reasoner(self):
        orch, llm, runner, _ = make({"reasoner": ["FINAL ANSWER: 8"], "programmer": [CODE_REPLY],
                                     "repair": [CODE_REPLY], "verifier": ["FINAL ANSWER: 8"]}, [])
        runner.raise_with = MemoryError()
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path), ("8", "verified"))

    def test_formatter_rewrite_with_different_numbers_rejected(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 5 apples, maybe 6"], "programmer": [CODE_REPLY],
                                "repair": [CODE_REPLY], "verifier": ["FINAL ANSWER: 5 apples, maybe 6"],
                                "formatter": ["9"]}, [bad_run(), bad_run()])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual(d.answer, "5 apples, maybe 6")

    def test_max_calls_cap(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: 7"], "programmer": [CODE_REPLY]}, [ok_run("8")],
                               max_calls=2)
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path, d.calls), ("8", "fallback", 2))

    def test_all_fail_empty(self):
        down = [LLMResult(role="x", error="URLError: refused")] * 3
        orch, llm, runner, _ = make({"reasoner": list(down), "programmer": list(down), "verifier": list(down),
                                     "tiebreak": list(down)}, [])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path), ("", "empty"))
        self.assertEqual(runner.codes, [])
        self.assertEqual(d.calls, 2)  # no repair/verifier/tie-break when the server is unreachable

    def test_formatter_used_for_invalid_winner(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: the second one"], "programmer": [CODE_REPLY],
                                "repair": [CODE_REPLY], "verifier": ["FINAL ANSWER: the second one"],
                                "formatter": ["Green"]}, [bad_run(), bad_run()])
        d = orch.solve(OPT_PROBLEM)
        self.assertEqual(d.answer, "Green")
        self.assertEqual(d.path, "verified")
        self.assertEqual(llm.roles[-1], "formatter")

    def test_truncated_reasoner_is_not_a_candidate(self):
        truncated = LLMResult(role="reasoner", content="FINAL ANSWER: 3", done_reason="length")
        orch, _, _, _ = make({"reasoner": [truncated], "programmer": [CODE_REPLY],
                              "verifier": ["FINAL ANSWER: 12"]}, [ok_run("12")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path), ("12", "verified"))


    def test_unfixable_invalid_winner_falls_back_to_a_valid_candidate(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: about twelve"], "programmer": [CODE_REPLY],
                                "verifier": ["FINAL ANSWER: about twelve"], "formatter": ["twelve"]},
                               [ok_run("12")])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual(d.answer, "12")
        self.assertEqual(d.path, "format-fallback")
        self.assertEqual(llm.roles[-1], "formatter")  # the winner's own answer was tried first

    def test_invalid_winner_without_valid_alternative_is_kept(self):
        orch, llm, _, _ = make({"reasoner": ["FINAL ANSWER: about twelve"], "programmer": [CODE_REPLY],
                                "repair": [CODE_REPLY], "verifier": ["FINAL ANSWER: about twelve"],
                                "formatter": ["twelve"]}, [bad_run(), bad_run()])
        d = orch.solve(INT_PROBLEM)
        self.assertEqual((d.answer, d.path), ("about twelve", "verified"))


if __name__ == "__main__":
    unittest.main()

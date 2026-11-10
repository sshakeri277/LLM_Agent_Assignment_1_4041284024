"""End-to-end: the real main() against a fake Ollama HTTP server (no model needed)."""
import glob
import json
import os
import tempfile
import unittest

import main as agent_main
from tests.fake_ollama import FakeOllama

P1 = "یک باغ ۳ ردیف درخت دارد و هر ردیف ۴ درخت. چند درخت دارد؟\nفقط عدد صحیح را برگردانید."
P2 = "یک کتابخانه ۶ قفسه دارد و هر قفسه ۲ کتاب. چند کتاب دارد؟\nفقط عدد صحیح را برگردانید."


def solver_handler(path, body):
    if path != "/api/chat":
        return 200, {"done": True}
    prompt = body["messages"][-1]["content"]
    if "Python 3 program" in prompt:
        content = "```python\nprint('ANSWER: 12')\n```"
    else:
        content = "Checked all conditions.\nFINAL ANSWER: 12"
    return 200, {"message": {"content": content}, "done_reason": "stop", "eval_count": 5}


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.inp = os.path.join(self.d, "input.json")
        self.out = os.path.join(self.d, "output.json")
        self.logs = os.path.join(self.d, "logs")

    def tearDown(self):
        self.tmp.cleanup()

    def _write_input(self, obj):
        with open(self.inp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    def _read_output(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)

    def test_end_to_end_list(self):
        self._write_input([{"id": "a", "problem": P1}, {"id": "b", "problem": P2}])
        with FakeOllama(solver_handler) as fake:
            code = agent_main.main(["--input", self.inp, "--output", self.out, "--host", fake.url,
                                    "--log-dir", self.logs])
            warm = [p for p, _ in fake.requests if p == "/api/generate"]
        self.assertEqual(code, 0)
        self.assertEqual(self._read_output(), [{"id": "a", "answer": "12"}, {"id": "b", "answer": "12"}])
        self.assertEqual(len(warm), 1)
        traces = glob.glob(os.path.join(self.logs, "run-*.jsonl"))
        self.assertEqual(len(traces), 1)
        with open(traces[0], encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        per_task = [r for r in records if "id" in r]
        self.assertEqual([r["path"] for r in per_task], ["consensus", "consensus"])
        self.assertEqual([r["calls"] for r in per_task], [2, 2])
        summary = [r for r in records if r.get("summary")][0]
        self.assertIsInstance(summary["warmup_seconds"], float)  # model-load time, outside the problem budgets
        self.assertGreaterEqual(summary["total_seconds"], summary["seconds"])

    def test_unreachable_ollama_still_valid_json(self):
        self._write_input([{"id": "a", "problem": P1}, {"id": "b", "problem": P2}])
        code = agent_main.main(["--input", self.inp, "--output", self.out, "--host", "http://127.0.0.1:9",
                                "--time-budget", "8", "--log-dir", self.logs])
        self.assertEqual(code, 0)
        self.assertEqual(self._read_output(), [{"id": "a", "answer": ""}, {"id": "b", "answer": ""}])

    def test_bad_input_writes_empty_list(self):
        with open(self.inp, "w") as f:
            f.write("{oops")
        code = agent_main.main(["--input", self.inp, "--output", self.out, "--log-dir", self.logs])
        self.assertEqual(code, 1)
        self.assertEqual(self._read_output(), [])

    def test_prefill_shape_object(self):
        self._write_input({"id": "solo", "problem": P1})
        with FakeOllama(solver_handler) as fake:
            code = agent_main.main(["--input", self.inp, "--output", self.out, "--host", fake.url,
                                    "--no-warmup", "--log-dir", self.logs])
        self.assertEqual(code, 0)
        self.assertEqual(self._read_output(), {"id": "solo", "answer": "12"})

    def test_empty_problem_makes_no_calls(self):
        self._write_input([{"id": "e", "problem": "  "}])
        with FakeOllama(solver_handler) as fake:
            code = agent_main.main(["--input", self.inp, "--output", self.out, "--host", fake.url,
                                    "--no-warmup", "--log-dir", self.logs])
            chats = fake.chat_bodies()
        self.assertEqual(code, 0)
        self.assertEqual(self._read_output(), [{"id": "e", "answer": ""}])
        self.assertEqual(chats, [])


if __name__ == "__main__":
    unittest.main()

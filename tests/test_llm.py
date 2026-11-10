import unittest

from agent import prompts
from agent.answer_format import extract_spec
from agent.config import load_settings
from agent.llm import OllamaLLM
from tests.fake_ollama import FakeOllama

PROBLEM = "یک مغازه ۳ جعبه دارد و در هر جعبه ۴ سیب است. چند سیب دارد؟\nفقط عدد صحیح را برگردانید."


def settings_for(url):
    return load_settings({}, host=url, log_dir=None)


class LLMTests(unittest.TestCase):
    def test_chat_payload_and_parse(self):
        def handler(path, body):
            return 200, {"message": {"content": "FINAL ANSWER: 5", "thinking": "abc"}, "done_reason": "stop",
                         "prompt_eval_count": 10, "eval_count": 20}
        with FakeOllama(handler) as fake:
            llm = OllamaLLM(settings_for(fake.url))
            r = llm.chat("reasoner", "hello", think="medium", num_predict=100, timeout=10, seed=42, system="sys")
            body = fake.chat_bodies()[0]
        self.assertTrue(r.ok)
        self.assertEqual(r.content, "FINAL ANSWER: 5")
        self.assertEqual((r.thinking_chars, r.prompt_tokens, r.gen_tokens), (3, 10, 20))
        self.assertEqual(body["model"], "qwen3.8:27b")
        self.assertIs(body["stream"], False)
        self.assertEqual(body["think"], "medium")
        self.assertEqual(body["options"], {"num_ctx": 16384, "num_predict": 100, "seed": 42})
        self.assertEqual(body["messages"], [{"role": "system", "content": "sys"},
                                            {"role": "user", "content": "hello"}])

    def test_length_is_not_ok(self):
        def handler(path, body):
            return 200, {"message": {"content": "partial"}, "done_reason": "length", "eval_count": 100}
        with FakeOllama(handler) as fake:
            r = OllamaLLM(settings_for(fake.url)).chat("reasoner", "x", think=False, num_predict=100, timeout=10)
        self.assertFalse(r.ok)

    def test_http_error_and_unreachable(self):
        with FakeOllama(lambda p, b: (404, {"error": "model not found"})) as fake:
            r = OllamaLLM(settings_for(fake.url)).chat("reasoner", "x", think=False, num_predict=10, timeout=10)
        self.assertFalse(r.ok)
        self.assertIn("404", r.error)
        r2 = OllamaLLM(settings_for("http://127.0.0.1:9")).chat("reasoner", "x", think=False,
                                                               num_predict=10, timeout=3)
        self.assertFalse(r2.ok)
        self.assertTrue(r2.error)

    def test_warmup_and_model_available(self):
        with FakeOllama() as fake:
            llm = OllamaLLM(settings_for(fake.url))
            self.assertTrue(llm.model_available())
            self.assertTrue(llm.warmup().error is None)
            paths = [p for p, _ in fake.requests]
            gen_body = [b for p, b in fake.requests if p == "/api/generate"][0]
        self.assertEqual(paths, ["/api/show", "/api/generate"])
        self.assertEqual(gen_body["options"]["num_ctx"], 16384)
        self.assertEqual(gen_body["keep_alive"], -1)
        self.assertFalse(OllamaLLM(settings_for("http://127.0.0.1:9")).model_available())


class PromptTests(unittest.TestCase):
    def test_prompts_contain_spec_and_problem(self):
        spec = extract_spec(PROBLEM)
        cands = [("step-by-step reasoning", "12"), ("computer program", "11")]
        all_prompts = [prompts.reasoner_prompt(PROBLEM, spec), prompts.programmer_prompt(PROBLEM, spec),
                       prompts.repair_prompt(PROBLEM, spec, "print(1)", "exit code 1"),
                       prompts.verifier_prompt(PROBLEM, spec, cands, "print('ANSWER: 11')"),
                       prompts.formatter_prompt(PROBLEM, spec, "twelve")]
        for p in all_prompts:
            self.assertIn(PROBLEM, p)
            self.assertIn(spec.description, p)
        self.assertIn("ANSWER:", prompts.programmer_prompt(PROBLEM, spec))
        self.assertIn("exit code 1", all_prompts[2])
        self.assertIn("print(1)", all_prompts[2])
        self.assertIn("12", all_prompts[3])
        self.assertIn("11", all_prompts[3])
        self.assertIn("print('ANSWER: 11')", all_prompts[3])
        self.assertIn("FINAL ANSWER:", prompts.reasoner_prompt(PROBLEM, spec))
        self.assertIn("twelve", all_prompts[4])


    def test_verifier_prompt_demands_condition_by_condition_check(self):
        spec = extract_spec(PROBLEM)
        p = prompts.verifier_prompt(PROBLEM, spec, [("step-by-step reasoning", "12"), ("computer program", "11")],
                                    "print('ANSWER: 11')")
        self.assertIn("PASS", p)
        self.assertIn("FAIL", p)
        self.assertIn("mis-encode", p)  # the program is evidence to check, not an authority

if __name__ == "__main__":
    unittest.main()

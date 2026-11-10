import unittest

from agent.config import load_settings


class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        s = load_settings({})
        self.assertEqual(s.model, "qwen3.8:27b")
        self.assertEqual(s.host, "http://127.0.0.1:11434")
        self.assertEqual(s.num_ctx, 16384)
        self.assertEqual(s.max_calls, 6)
        self.assertIs(s.think_formatter, False)
        self.assertEqual(s.think_reasoner, "medium")
        self.assertEqual(s.think_programmer, "low")
        self.assertEqual(s.think_tiebreak, "medium")
        self.assertEqual(s.max_call_share, 0.4)

    def test_env_overrides(self):
        s = load_settings({"OLLAMA_HOST": "10.0.0.5:11434",
                           "AGENT_THINK_REASONER": "false",
                           "AGENT_TIME_BUDGET": "90"})
        self.assertEqual(s.host, "http://10.0.0.5:11434")
        self.assertIs(s.think_reasoner, False)
        self.assertEqual(s.time_budget, 90.0)

    def test_host_with_scheme_kept(self):
        self.assertEqual(load_settings({"OLLAMA_HOST": "https://h:1/"}).host, "https://h:1")

    def test_host_defaults_like_ollama(self):
        cases = {"0.0.0.0": "http://127.0.0.1:11434", "127.0.0.1": "http://127.0.0.1:11434",
                 "0.0.0.0:11434": "http://127.0.0.1:11434", "10.0.0.5:11434": "http://10.0.0.5:11434",
                 ":11434": "http://127.0.0.1:11434", "[::]:11434": "http://127.0.0.1:11434",
                 "localhost": "http://localhost:11434", "http://example.com": "http://example.com"}
        for raw, want in cases.items():
            self.assertEqual(load_settings({"OLLAMA_HOST": raw}).host, want, raw)

    def test_malformed_env_value_keeps_default(self):
        # M5: a bad value must not crash the run before output.json exists.
        with self.assertLogs("agent", level="WARNING") as logs:
            s = load_settings({"AGENT_MAX_CALLS": "six", "AGENT_TIME_BUDGET": "90"})
        self.assertEqual(s.max_calls, 6)
        self.assertEqual(s.time_budget, 90.0)
        self.assertIn("AGENT_MAX_CALLS", "\n".join(logs.output))

    def test_cli_override_wins(self):
        self.assertEqual(load_settings({"AGENT_MODEL": "a"}, model="b").model, "b")
        self.assertEqual(load_settings({}, model=None).model, "qwen3.8:27b")


if __name__ == "__main__":
    unittest.main()

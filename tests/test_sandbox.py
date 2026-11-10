import os
import subprocess
import unittest
from unittest import mock

from agent.sandbox import check_code, extract_code, parse_answers, run_program


class ExtractTests(unittest.TestCase):
    def test_extract_last_block(self):
        text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint('ANSWER: 2')\n```\n"
        self.assertEqual(extract_code(text).strip(), "print('ANSWER: 2')")
        self.assertEqual(extract_code("x = 3\nprint('ANSWER:', x)").strip(), "x = 3\nprint('ANSWER:', x)")
        self.assertIsNone(extract_code("I think the answer is clearly five because reasons."))
        self.assertEqual(extract_code("```\nprint(5)\n```").strip(), "print(5)")

    def test_extract_prefers_python_block_over_trailing_output(self):
        text = "```python\nprint('ANSWER: 8')\n```\nOutput:\n```\nANSWER: 8\n```"
        self.assertEqual(extract_code(text).strip(), "print('ANSWER: 8')")

    def test_check_code(self):
        self.assertIn("os", check_code("import os\nprint(1)"))
        self.assertIn("subprocess", check_code("from subprocess import run"))
        self.assertIn("open", check_code("open('x', 'w').write('y')"))
        self.assertIn("__import__", check_code("__import__('socket')"))
        self.assertIsNone(check_code("from itertools import permutations\nimport math\nprint(1)"))
        self.assertIn("syntax", check_code("def broken(:\n  pass").lower())

    def test_check_code_blocks_private_and_reflective_access(self):
        # M6: the reviewer's bypasses and their close relatives.
        self.assertIn("_os", check_code("import random\nrandom._os.system('true')"))
        self.assertIn("getattr", check_code("import random\ngetattr(random, '_os')"))
        self.assertIn("setattr", check_code("import math\nsetattr(math, 'pi', 3)"))
        self.assertIn("attrgetter", check_code("from operator import attrgetter\nattrgetter('_os')"))
        self.assertIsNotNone(check_code("import dataclasses\ndataclasses.inspect.os.system('true')"))
        self.assertIn("__dict__", check_code("import random\nrandom.__dict__['_os']"))

    def test_check_code_allows_common_dunders_and_namedtuple_methods(self):
        code = ("from collections import namedtuple\n"
                "P = namedtuple('P', 'x y')\n"
                "p = P(1, 2)._replace(x=3)\n"
                "class A:\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.total = 0\n"
                "print('ANSWER:', type(p).__name__, p._asdict())\n")
        self.assertIsNone(check_code(code))

    def test_program_rules_tell_the_model_about_the_gate(self):
        from agent.answer_format import FormatSpec
        from agent.prompts import programmer_prompt
        prompt = programmer_prompt("problem", FormatSpec("integer"))
        self.assertIn("underscore", prompt)
        self.assertIn("getattr", prompt)

    def test_parse_answers(self):
        self.assertEqual(parse_answers("ANSWER: 3\nANSWER: 1\nANSWER: 3\n"), ["3", "1"])
        self.assertEqual(parse_answers("debug\n42\n"), ["42"])
        self.assertEqual(parse_answers("\n \n"), [])


class RunTests(unittest.TestCase):
    def test_run_multi_answer(self):
        r = run_program("for a in (3, 1, 3):\n    print(f'ANSWER: {a}')", timeout=10, max_answers=10)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.answers, ["3", "1"])

    def test_fallback_last_line(self):
        r = run_program("print('debug')\nprint(42)", timeout=10, max_answers=10)
        self.assertTrue(r.ok)
        self.assertEqual(r.answers, ["42"])

    def test_timeout(self):
        r = run_program("while True:\n    pass", timeout=2, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("timeout", r.error)
        self.assertLess(r.seconds, 6)

    def test_exception_tail(self):
        r = run_program("print('ANSWER: 1')\n1/0", timeout=10, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("ZeroDivisionError", r.error)

    def test_too_many_answers(self):
        r = run_program("for i in range(1000):\n    print(f'ANSWER: {i}')", timeout=10, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("too many", r.error)

    def test_no_output(self):
        r = run_program("x = 1", timeout=10, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("no output", r.error)

    def test_forbidden_import_not_executed(self):
        r = run_program("import socket\nprint('ANSWER: 1')", timeout=10, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("socket", r.error)

    @unittest.skipUnless(os.name == "posix", "file-size rlimit is POSIX-only")
    def test_stderr_flood_is_bounded(self):
        code = "import sys\nfor _ in range(800):\n    sys.stderr.write('e' * 65536)\nprint('ANSWER: 1')"
        r = run_program(code, timeout=10, max_answers=10)
        self.assertFalse(r.ok)          # 50 MB of stderr exceeds the 10 MB file limit
        self.assertLess(len(r.error), 1000)

    def test_run_program_never_raises(self):
        with mock.patch.object(subprocess, "run", side_effect=RuntimeError("boom")):
            r = run_program("print('ANSWER: 1')", timeout=5, max_answers=10)
        self.assertFalse(r.ok)
        self.assertIn("boom", r.error)

    def test_persian_output_utf8(self):
        r = run_program("print('ANSWER: دوشنبه')", timeout=10, max_answers=10)
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.answers, ["دوشنبه"])


if __name__ == "__main__":
    unittest.main()

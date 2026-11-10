import json
import os
import tempfile
import unittest

from agent.io_utils import InputError, load_input, render_output, write_json_atomic


class IoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, data: bytes) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_single_object_with_bom(self):
        path = self._write("in.json", b"\xef\xbb\xbf" + '{"id":"t1","problem":"x"}'.encode())
        doc = load_input(path)
        self.assertEqual(doc.shape, "object")
        self.assertEqual(render_output(doc, ["7"]), {"id": "t1", "answer": "7"})

    def test_list_and_wrapped(self):
        items = [{"id": "a", "problem": "p1"}, {"id": "b", "problem": "p2"}]
        doc = load_input(self._write("l.json", json.dumps(items).encode()))
        self.assertEqual(doc.shape, "list")
        self.assertEqual(render_output(doc, ["1", "2"]),
                         [{"id": "a", "answer": "1"}, {"id": "b", "answer": "2"}])
        wrapped = load_input(self._write("w.json", json.dumps({"problems": items}).encode()))
        self.assertEqual(wrapped.shape, "list")
        self.assertEqual([t.problem for t in wrapped.tasks], ["p1", "p2"])

    def test_missing_fields(self):
        doc = load_input(self._write("m.json", json.dumps([{"problem": "p"}, {"id": "b"}]).encode()))
        self.assertEqual([t.id for t in doc.tasks], [0, "b"])
        self.assertEqual([t.problem for t in doc.tasks], ["p", ""])

    def test_invalid_json_raises(self):
        with self.assertRaises(InputError):
            load_input(self._write("bad.json", b"{not json"))
        with self.assertRaises(InputError):
            load_input(os.path.join(self.dir, "missing.json"))

    def test_atomic_write_unicode(self):
        out = os.path.join(self.dir, "out.json")
        write_json_atomic(out, {"answer": "دوشنبه"})
        with open(out, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("دوشنبه", text)
        self.assertEqual(json.loads(text), {"answer": "دوشنبه"})
        self.assertEqual([n for n in os.listdir(self.dir) if n.endswith(".tmp")], [])


if __name__ == "__main__":
    unittest.main()

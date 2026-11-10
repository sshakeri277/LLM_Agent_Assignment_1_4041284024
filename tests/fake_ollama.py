"""A tiny in-process HTTP server that mimics the parts of the Ollama REST API the agent uses."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Tuple

Handler = Callable[[str, dict], Tuple[int, dict]]


def default_handler(path: str, body: dict) -> Tuple[int, dict]:
    if path == "/api/show":
        return 200, {"details": {"family": "fake"}}
    if path == "/api/generate":
        return 200, {"response": "", "done": True}
    return 200, {"message": {"role": "assistant", "content": "FINAL ANSWER: 0"}, "done_reason": "stop",
                 "prompt_eval_count": 1, "eval_count": 1}


class FakeOllama:
    def __init__(self, handler: Handler = default_handler):
        self.handler = handler
        self.requests: List[Tuple[str, dict]] = []
        outer = self

        class _H(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, body))
                status, payload = outer.handler(self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def chat_bodies(self) -> List[dict]:
        return [b for p, b in self.requests if p == "/api/chat"]

    def __enter__(self) -> "FakeOllama":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()


"""Authenticated HTTP proxy for local Ollama."""
import hmac
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

OLLAMA_URL = "http://127.0.0.1:11434"
TOKEN = os.environ.get("CODESAGE_PROXY_TOKEN", "")

ALLOWED_PATHS = {
    "/api/tags",
    "/api/generate",
    "/api/chat",
    "/api/show",
}

class ProxyHandler(BaseHTTPRequestHandler):
    def handle_request(self):
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {TOKEN}"

        if not TOKEN or not hmac.compare_digest(supplied, expected):
            self.send_error(401, "Unauthorized")
            return

        path = urlsplit(self.path).path
        if path not in ALLOWED_PATHS:
            self.send_error(404, "Endpoint not allowed")
            return

        body = None
        if self.command == "POST":
            length = int(self.headers.get("Content-Length", 0))
            if length > 5 * 1024 * 1024:
                self.send_error(413, "Request too large")
                return
            body = self.rfile.read(length)

        request = Request(
            OLLAMA_URL + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method=self.command,
        )

        try:
            with urlopen(request, timeout=180) as response:
                result = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(result)))
                self.end_headers()
                self.wfile.write(result)

        except HTTPError as exc:
            result = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

        except (URLError, TimeoutError, OSError) as exc:
            self.send_error(502, f"Ollama unavailable: {exc}")

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("CODESAGE_PROXY_TOKEN is not configured")

    server = ThreadingHTTPServer(
        ("127.0.0.1", 8000), ProxyHandler
    )
    print("Authenticated Ollama proxy running on port 8000")
    server.serve_forever()

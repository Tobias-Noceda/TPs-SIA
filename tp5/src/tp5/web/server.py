import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tp5.web.explorer import LatentExplorer

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Largest batch of latent points decoded in one request (a 64 × 64 grid)
MAX_POINTS = 4096

def _handler(explorer: LatentExplorer) -> type[BaseHTTPRequestHandler]:
    # The model is shared by the request threads: one decode at a time
    lock = threading.Lock()
    with lock:
        dataset = json.dumps(explorer.dataset()).encode()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Silent: a slider drag makes dozens of requests per second
            pass

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/dataset":
                self._send(HTTPStatus.OK, dataset, "application/json")
                return

            name = "index.html" if path == "/" else path.removeprefix("/static/")
            file = (STATIC_DIR / name).resolve()
            # Only files inside static/ (no ../ escapes)
            if STATIC_DIR not in file.parents or not file.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
                return
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, file.read_bytes(), content_type)

        def do_POST(self) -> None:
            if self.path not in ("/api/decode", "/api/generate"):
                self._json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                points = body["z"]
                if self.path == "/api/generate":
                    # One latent point [z0, ..., zn-1] and the file name chosen by the user
                    if not isinstance(points, list) or not all(isinstance(v, (int, float)) for v in points):
                        raise ValueError("'z' must be a list of latent coordinates.")
                    name = body.get("name")
                    if not isinstance(name, str):
                        raise ValueError("A file name is required.")
                    with lock:
                        files = explorer.generate(points, name)
                    result = {"files": [str(path) for path in files]}
                else:
                    if not isinstance(points, list) or not 0 < len(points) <= MAX_POINTS:
                        raise ValueError(f"'z' must be a list of 1 to {MAX_POINTS} latent points.")
                    with lock:
                        result = explorer.decode(points)
            except (ValueError, KeyError, TypeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except FileExistsError as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            except OSError as error:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not save: {error}"})
                return
            self._json(HTTPStatus.OK, result)

    return Handler

def serve(explorer: LatentExplorer, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Serve the latent space explorer until Ctrl+C."""
    try:
        server = ThreadingHTTPServer((host, port), _handler(explorer))
    except OSError as error:
        raise SystemExit(f"Cannot listen on {host}:{port} ({error.strerror}). Try another --port.") from error

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Latent space explorer at {url} (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the explorer")
    finally:
        server.server_close()

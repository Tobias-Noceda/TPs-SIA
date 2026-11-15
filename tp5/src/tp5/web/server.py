import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict
from urllib.parse import unquote

from tp5.web.explorer import LatentExplorer

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Pages: route → file in static/
PAGES = {"/": "index.html", "/denoise": "denoise.html"}

# Largest batch of latent points / images processed in one request (a 64 x 64 grid)
MAX_POINTS = 4096

def _list(body: Dict[str, Any], key: str) -> list:
    value = body.get(key)
    if not isinstance(value, list) or not 0 < len(value) <= MAX_POINTS:
        raise ValueError(f"'{key}' must be a list of 1 to {MAX_POINTS} elements.")
    return value

def _string(body: Dict[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise ValueError(f"'{key}' is required.")
    return value

def _routes(explorer: LatentExplorer) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """POST endpoints: JSON body → JSON result."""

    def generate(body: Dict[str, Any]) -> Any:
        # One latent point [z0, ..., zn-1] and the file name chosen by the user
        points = body.get("z")
        if not isinstance(points, list) or not all(isinstance(v, (int, float)) for v in points):
            raise ValueError("'z' must be a list of latent coordinates.")
        return {"files": [str(path) for path in explorer.generate(points, _string(body, "name"))]}

    def noise(body: Dict[str, Any]) -> Any:
        level = body.get("level")
        if not isinstance(level, (int, float)):
            raise ValueError("'level' must be a number.")
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("'seed' must be a non-negative integer or null.")
        return explorer.noise(_list(body, "samples"), _string(body, "type"), float(level), seed)

    def save_noisy(body: Dict[str, Any]) -> Any:
        noise = body.get("noise")
        if not isinstance(noise, dict):
            raise ValueError("'noise' must be an object.")
        return {"file": str(explorer.save_noisy(body.get("sample"), noise, _list(body, "noisy")))}

    def save_denoising(body: Dict[str, Any]) -> Any:
        noise = body.get("noise")
        if not isinstance(noise, dict):
            raise ValueError("'noise' must be an object.")
        noisy = _list(body, "noisy")
        return {"files": [str(explorer.save_denoising(body.get("sample"), noisy, noise, _string(body, "name")))]}

    def sample(body: Dict[str, Any]) -> Any:
        seed = body.get("seed")
        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("'seed' must be a non-negative integer or null.")
        return explorer.sample(body.get("count", 1), seed)

    return {
        "/api/decode": lambda body: explorer.decode(_list(body, "z")),
        "/api/sample": sample,
        "/api/generate": generate,
        "/api/noise": noise,
        "/api/reconstruct": lambda body: explorer.reconstruct(_list(body, "images")),
        "/api/noisy": save_noisy,
        "/api/save-denoising": save_denoising,
    }

def _handler(explorer: LatentExplorer) -> type[BaseHTTPRequestHandler]:
    # The model is shared by the request threads: one request at a time touches it
    lock = threading.Lock()
    with lock:
        dataset = json.dumps(explorer.dataset()).encode()
    routes = _routes(explorer)

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
            if path == "/api/noisy":
                self._json(HTTPStatus.OK, {"files": explorer.list_noisy()})
                return
            if path.startswith("/api/noisy/"):
                try:
                    self._json(HTTPStatus.OK, explorer.load_noisy(unquote(path.removeprefix("/api/noisy/"))))
                except ValueError as error:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                except (OSError, json.JSONDecodeError) as error:
                    self._json(HTTPStatus.NOT_FOUND, {"error": f"Cannot read the noisy file: {error}"})
                return

            name = PAGES.get(path, path.removeprefix("/static/"))
            file = (STATIC_DIR / name).resolve()
            # Only files inside static/ (no ../ escapes)
            if STATIC_DIR not in file.parents or not file.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {path}"})
                return
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            self._send(HTTPStatus.OK, file.read_bytes(), content_type)

        def do_POST(self) -> None:
            route = routes.get(self.path)
            if route is None:
                self._json(HTTPStatus.NOT_FOUND, {"error": f"Not found: {self.path}"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("The body must be a JSON object.")
                with lock:
                    result = route(body)
            except FileExistsError as error:
                self._json(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            except (ValueError, KeyError, TypeError) as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except OSError as error:
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Could not save: {error}"})
                return
            self._json(HTTPStatus.OK, result)

    return Handler

def serve(explorer: LatentExplorer, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    """Serve the latent space explorer (/) and the denoising page (/denoise) until Ctrl+C."""
    try:
        server = ThreadingHTTPServer((host, port), _handler(explorer))
    except OSError as error:
        raise SystemExit(f"Cannot listen on {host}:{port} ({error.strerror}). Try another --port.") from error

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
    print(f"Latent space explorer at {url} · denoising page at {url}denoise (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the explorer")
    finally:
        server.server_close()

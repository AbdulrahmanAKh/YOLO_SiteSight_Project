"""Preview the Pages site locally with the right MIME types for ONNX Runtime."""

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "application/javascript"}


print("Open http://127.0.0.1:8000/site/")
ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()

"""Static files + POST /log/<name> (appends one line to logs/<name>.log). Phone pages report progress here."""
import http.server
import os
import sys

os.makedirs("logs", exist_ok=True)


class H(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n).decode()
        name = (self.path.split("/log/")[-1] or "page").replace("/", "_")
        with open(f"logs/{name}.log", "a") as f:
            f.write(body + "\n")
        self.send_response(204)
        self.end_headers()

    def end_headers(self):
        big = self.path.split("?")[0].endswith((".bin", "rows.json", "weights.json"))
        self.send_header("Cache-Control", "max-age=86400" if big else "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


port = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
http.server.ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()

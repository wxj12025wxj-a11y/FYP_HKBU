"""Local static server for the A/B listening page.

Why not `python -m http.server`: the stdlib handler ignores HTTP Range
requests.  A browser playing a 430 s / 76 MB WAV then cannot seek -- it can
only play the part it already buffered.  This handler implements single-range
GET (206 Partial Content) so seeking works on full-length lossless audio.

Usage:
    python tools/_listen_server.py --root E:\\FYP_HKBU --port 8123
"""
from __future__ import annotations

import argparse
import mimetypes
import os
import posixpath
import re
import socketserver
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")

EXTRA_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".opus": "audio/ogg",
    ".ogg": "audio/ogg",
}


class QuietHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FYPListen/1.0"

    # ---- helpers -------------------------------------------------------
    def guess_type(self, path):  # noqa: A003
        ext = posixpath.splitext(path)[1].lower()
        if ext in EXTRA_TYPES:
            return EXTRA_TYPES[ext]
        return mimetypes.guess_type(path)[0] or "application/octet-stream"

    def log_message(self, fmt, *args):  # keep the console readable
        code = args[1] if len(args) > 1 else ""
        if str(code).startswith(("2", "3")) and str(code) != "206":
            return
        sys.stderr.write("[srv] %s %s\n" % (self.address_string(), fmt % args))

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    # ---- core ----------------------------------------------------------
    def _resolve(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            index = os.path.join(path, "index.html")
            if os.path.isfile(index):
                return index
            return None
        return path if os.path.isfile(path) else None

    def send_head(self):  # noqa: C901
        path = self._resolve()
        if path is None:
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        ctype = self.guess_type(path)
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            return f

        m = RANGE_RE.match(rng.strip())
        if not m:
            f.close()
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return None
        start_s, end_s = m.group(1), m.group(2)
        if start_s == "":                      # suffix range: bytes=-N
            length = min(int(end_s or 0), size)
            start, end = size - length, size - 1
        else:
            start = int(start_s)
            end = min(int(end_s), size - 1) if end_s else size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", "bytes */%d" % size)
            self.end_headers()
            return None

        f.seek(start)
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        self._range_left = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        left = getattr(self, "_range_left", None)
        if left is None:
            return super().copyfile(source, outputfile)
        self._range_left = None
        while left > 0:
            chunk = source.read(min(262144, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd())
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        raise SystemExit("root not found: %s" % root)

    def handler(*args, **kw):
        return QuietHandler(*args, directory=root, **kw)

    page = "http://%s:%d/04_reports/separation/html/listen_compare.html" % (a.host, a.port)
    print("root :", root)
    print("page :", page)
    print("stop : Ctrl-C")
    with Server((a.host, a.port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()

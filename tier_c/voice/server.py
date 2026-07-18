# -*- coding: utf-8 -*-
"""Голосовий міст: маленький локальний веб-сервер.

Роздає index.html (голосову сторінку) і приймає команди POST /cmd, які пише у
файл out/voice_cmd.txt — його слухає Blender (_voice_poll у blender_manual.py).

Запуск:
    cd tier_c/voice
    python3 server.py            # → http://localhost:8000

Із телефону (той самий Wi-Fi): http://<IP-Mac>:8000 — але мікрофон у мобільному
браузері вимагає HTTPS, тож без тунелю (ngrok) з телефону голос не працюватиме;
на самому Mac (localhost) — працює.
"""

import http.server
import os
import pathlib
import socketserver
import urllib.parse

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
CMD_FILE = OUT / "voice_cmd.txt"
PORT = 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") == "/cmd":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            cmd = urllib.parse.parse_qs(body).get("cmd", [""])[0]
            CMD_FILE.write_text(cmd, encoding="utf-8")
            print("← команда:", cmd)
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(("OK: " + cmd).encode("utf-8"))
        else:
            self.send_error(404)

    def do_GET(self):
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def log_message(self, *args):
        pass                                     # тихо


if __name__ == "__main__":
    os.chdir(HERE)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
        print("Голосовий сервер: http://localhost:%d" % PORT)
        print("Команди пишуться у:", CMD_FILE)
        print("Ctrl+C щоб зупинити.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nЗупинено.")

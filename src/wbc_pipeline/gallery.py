# This project was developed with assistance from AI tools.
"""Video gallery for browsing training recordings stored in S3/MinIO."""

import json
import os
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote

import boto3

S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://minio.wbc-training.svc:9000")
S3_BUCKET = os.environ.get("S3_BUCKET", "wbc-training")
PORT = int(os.environ.get("GALLERY_PORT", "8080"))

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WBC Training Videos</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0d1117; color: #c9d1d9;
  }
  header {
    background: #161b22; border-bottom: 1px solid #30363d;
    padding: 1.2rem 2rem; display: flex; align-items: center; gap: 1rem;
  }
  header h1 { font-size: 1.3rem; font-weight: 600; color: #f0f6fc; }
  header .badge {
    background: #238636; color: #fff; padding: 0.2rem 0.6rem;
    border-radius: 12px; font-size: 0.75rem; font-weight: 500;
  }
  main { max-width: 1400px; margin: 0 auto; padding: 2rem; }
  .empty {
    text-align: center; padding: 4rem 2rem; color: #8b949e;
  }
  .empty code { background: #21262d; padding: 0.2rem 0.5rem; border-radius: 4px; }
  .run-group { margin-bottom: 2.5rem; }
  .run-header {
    font-size: 1.1rem; font-weight: 600; color: #f0f6fc;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
  }
  .run-header .count { color: #8b949e; font-weight: 400; font-size: 0.85rem; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 1rem;
  }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    overflow: hidden; cursor: pointer; transition: border-color 0.15s;
  }
  .card:hover { border-color: #58a6ff; }
  .card.active { border-color: #58a6ff; }
  .card-body {
    padding: 0.8rem 1rem; display: flex;
    justify-content: space-between; align-items: center;
  }
  .card-title { font-size: 0.9rem; font-weight: 500; color: #c9d1d9; }
  .card-meta { font-size: 0.75rem; color: #8b949e; }
  .card video {
    width: 100%; aspect-ratio: 16/9; background: #010409;
    display: none; object-fit: contain;
  }
  .card.active video { display: block; }
  .card .thumb {
    width: 100%; aspect-ratio: 16/9; background: #010409;
    display: flex; align-items: center; justify-content: center;
    color: #30363d; font-size: 2.5rem;
  }
  .card.active .thumb { display: none; }
  footer {
    text-align: center; padding: 2rem; color: #484f58; font-size: 0.8rem;
  }
</style>
</head>
<body>
<header>
  <h1>WBC Training Videos</h1>
  <span class="badge" id="count"></span>
</header>
<main id="app"></main>
<footer>Auto-refreshes every 30s</footer>
<script>
let activeCard = null;
function render(groups) {
  const app = document.getElementById('app');
  const total = Object.values(groups).reduce((s, v) => s + v.length, 0);
  document.getElementById('count').textContent = total + ' videos';
  if (total === 0) {
    app.innerHTML = '<div class="empty"><p>No videos yet.</p>'
      + '<p style="margin-top:1rem">Videos appear here automatically when training'
      + ' runs with <code>VIDEO_ENABLED=true</code>.</p></div>';
    return;
  }
  let html = '';
  for (const [run, videos] of Object.entries(groups)) {
    html += '<div class="run-group">';
    html += '<div class="run-header">' + esc(run)
      + ' <span class="count">(' + videos.length + ' videos)</span></div>';
    html += '<div class="grid">';
    for (const v of videos) {
      html += '<div class="card" data-src="/video/' + esc(v.key) + '" onclick="toggle(this)">';
      html += '<div class="thumb">&#9654;</div>';
      html += '<video loop muted></video>';
      html += '<div class="card-body">';
      html += '<span class="card-title">Iteration ' + v.iter + '</span>';
      html += '<span class="card-meta">' + v.size + '</span>';
      html += '</div></div>';
    }
    html += '</div></div>';
  }
  app.innerHTML = html;
}
function toggle(card) {
  const video = card.querySelector('video');
  if (activeCard && activeCard !== card) {
    activeCard.classList.remove('active');
    const av = activeCard.querySelector('video');
    av.pause(); av.removeAttribute('src'); av.load();
  }
  if (card.classList.toggle('active')) {
    video.src = card.dataset.src;
    video.play();
    activeCard = card;
  } else {
    video.pause(); video.removeAttribute('src'); video.load();
    activeCard = null;
  }
}
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
function refresh() {
  fetch('/api/videos').then(r => r.json()).then(render).catch(() => {});
}
refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


def _format_size(size_bytes: int) -> str:
    if size_bytes > 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    return f"{size_bytes / 1_000:.0f} KB"


def _scan_videos() -> dict:
    """List .mp4 files in the S3 bucket, grouped by training run prefix."""
    groups: dict[str, list] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".mp4"):
                continue
            parts = key.rsplit("/", 1)
            if len(parts) != 2:
                continue
            parent, filename = parts

            # Group by S3 prefix: {prefix}/videos/{name}.mp4 → prefix
            if parent.endswith("/videos"):
                run_name = parent[: -len("/videos")]
            else:
                run_name = parent
            run_name = run_name.replace("-", " ").replace("_", " ").title()

            match = re.match(r"^.+?_iter_(\d+)\.mp4$", filename)
            iteration = int(match.group(1)) if match else 0

            groups.setdefault(run_name, []).append(
                {
                    "key": key,
                    "iter": iteration,
                    "size": _format_size(obj.get("Size", 0)),
                }
            )

    for vids in groups.values():
        vids.sort(key=lambda v: v["iter"])

    return groups


class GalleryHandler(SimpleHTTPRequestHandler):
    """Serves the gallery HTML and proxies video files from S3."""

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_html()
        elif self.path == "/api/videos":
            self._serve_api()
        elif self.path.startswith("/video/"):
            self._serve_video()
        elif self.path == "/healthz":
            self._serve_health()
        else:
            self.send_error(404)

    def _serve_html(self):
        data = HTML_TEMPLATE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_api(self):
        groups = _scan_videos()
        data = json.dumps(groups).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    _ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv"}

    def _serve_video(self):
        s3_key = unquote(self.path[len("/video/") :])
        if ".." in s3_key or not any(s3_key.endswith(ext) for ext in self._ALLOWED_VIDEO_EXTENSIONS):
            self.send_error(403)
            return
        try:
            resp = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", resp["ContentLength"])
            self.end_headers()
            body = resp["Body"]
            while chunk := body.read(65536):
                self.wfile.write(chunk)
        except Exception:
            self.send_error(404)

    def _serve_health(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        msg = str(args)
        if "/api/videos" not in msg and "/healthz" not in msg:
            super().log_message(format, *args)


if __name__ == "__main__":
    print(f"Gallery serving s3://{S3_BUCKET} at http://0.0.0.0:{PORT}")
    HTTPServer(("0.0.0.0", PORT), GalleryHandler).serve_forever()

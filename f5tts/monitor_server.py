from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts_text import SENTENCE_GAP, SPEED, join_wavs, max_chunk_bytes, prepare_text, split_text

PORT = 8787
BASE_DIR = os.path.dirname(__file__)
HTML_PATH = os.path.join(BASE_DIR, "monitor.html")
REPO_DIR = os.path.join(BASE_DIR, "F5-TTS-Vietnamese")
VENV_PY = os.path.join(BASE_DIR, ".venv", "bin", "f5-tts_infer-cli")
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")
WORKER = os.path.join(BASE_DIR, "tts_worker.py")
CKPT_DIR = os.path.join(REPO_DIR, "ckpts", "your_training_dataset")
VOCAB_FILE = os.path.join(REPO_DIR, "data", "your_training_dataset", "vocab.txt")
REF_AUDIO = os.path.join(REPO_DIR, "data", "your_training_dataset", "wavs", "sample_00004.wav")
REF_TEXT = "tình cờ lạc bước vào một ngôi mộ hoang của quý phi đời trước."
OUTPUT_DIR = os.path.join(BASE_DIR, "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def job_update(job_id: str, **fields) -> None:
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(fields)


def job_read(job_id: str) -> dict:
    with JOBS_LOCK:
        return dict(JOBS.get(job_id) or {})


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def cpu_percent() -> float:
    out = sh(["top", "-l", "1", "-n", "0"])
    m = re.search(r"CPU usage:\s*([\d.]+)%\s*user,\s*([\d.]+)%\s*sys", out)
    if not m:
        return 0.0
    return round(float(m.group(1)) + float(m.group(2)), 1)


def mem_stats() -> dict:
    total_bytes = int(sh(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    out = sh(["vm_stat"])
    page_size_m = re.search(r"page size of (\d+) bytes", out)
    page_size = int(page_size_m.group(1)) if page_size_m else 4096

    def pages(key: str) -> int:
        m = re.search(rf"{key}:\s*(\d+)\.", out)
        return int(m.group(1)) if m else 0

    used_pages = pages("Pages active") + pages("Pages wired down") + pages("Pages occupied by compressor")
    used_bytes = used_pages * page_size
    total_gb = total_bytes / (1024 ** 3)
    used_gb = used_bytes / (1024 ** 3)
    percent = round(100.0 * used_bytes / total_bytes, 1) if total_bytes else 0.0
    return {"percent": percent, "used_gb": round(used_gb, 1), "total_gb": round(total_gb, 1)}


def training_status() -> dict:
    pids = sh(["pgrep", "-f", "finetune_cli.py"]).split()
    if not pids:
        return {"running": False, "cpu_percent": 0.0, "state": None}
    total_cpu = 0.0
    state = None
    for pid in pids:
        out = sh(["ps", "-o", "pcpu=,stat=", "-p", pid]).strip()
        if not out:
            continue
        parts = out.split()
        total_cpu += float(parts[0])
        state = parts[1] if len(parts) > 1 else state
    paused = bool(state and state.startswith("T"))
    return {"running": True, "cpu_percent": round(total_cpu, 1), "paused": paused}


def list_checkpoints() -> list[dict]:
    if not os.path.isdir(CKPT_DIR):
        return []
    numbered = []
    for f in os.listdir(CKPT_DIR):
        if f.startswith("model_") and f.endswith(".pt") and f != "model_last.pt":
            digits = "".join(ch for ch in f if ch.isdigit())
            if digits:
                numbered.append((int(digits), f))
    numbered.sort(key=lambda x: -x[0])

    items = []
    for update, f in numbered:
        items.append({"value": f, "label": f"update {update}"})
    if os.path.isfile(os.path.join(CKPT_DIR, "model_last.pt")):
        items.append({"value": "model_last.pt", "label": "moi nhat (model_last.pt)"})
    if os.path.isfile(os.path.join(CKPT_DIR, "pretrained_vn1000h.pt")):
        items.append({"value": "pretrained_vn1000h.pt", "label": "pretrained goc (chua finetune)"})
    return items


def resolve_checkpoint(name: str) -> str:
    safe = os.path.basename(name or "")
    path = os.path.join(CKPT_DIR, safe)
    if safe and os.path.isfile(path):
        return path
    options = list_checkpoints()
    if options:
        return os.path.join(CKPT_DIR, options[0]["value"])
    return os.path.join(CKPT_DIR, "pretrained_vn1000h.pt")


def synthesize(text: str, checkpoint_name: str = "", name: str = "", speed: float = SPEED) -> tuple[bool, str]:
    text = prepare_text(text)
    name = name or f"gen_{int(time.time() * 1000)}"
    wav_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    mp3_path = os.path.join(OUTPUT_DIR, f"{name}.mp3")
    checkpoint = resolve_checkpoint(checkpoint_name)
    parts = split_text(text, max_chunk_bytes(REF_AUDIO, REF_TEXT))
    part_paths: list[str] = []
    job_update(name, total=len(parts), done=0, state="running", started=time.time())

    worker = subprocess.Popen(
        ["arch", "-arm64", VENV_PYTHON, WORKER,
         "--ckpt_file", checkpoint,
         "--vocab_file", VOCAB_FILE,
         "--ref_audio", REF_AUDIO,
         "--ref_text", REF_TEXT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, cwd=REPO_DIR, bufsize=1,
    )

    def ask(payload: dict | None) -> dict:
        if payload is not None:
            worker.stdin.write(json.dumps(payload) + "\n")
            worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            return {"ok": False, "error": "worker thoat bat ngo"}
        return json.loads(line)

    try:
        job_update(name, state="loading")
        ready = ask(None)
        if not ready.get("ok"):
            return False, ready.get("error", "worker loi khi khoi dong")

        job_update(name, state="running", started=time.time())
        for position, part in enumerate(parts):
            part_path = os.path.join(OUTPUT_DIR, f"{name}_{position:03d}.wav")
            result = ask({"text": part, "out": part_path, "speed": speed})
            if not result.get("ok"):
                return False, result.get("error", "loi khong ro")
            part_paths.append(part_path)
            job_update(name, done=position + 1, done_at=time.time())

        job_update(name, state="joining")
        join_wavs(part_paths, wav_path, SENTENCE_GAP)
    finally:
        try:
            worker.stdin.write(json.dumps({"stop": True}) + "\n")
            worker.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        worker.terminate()
        worker.wait(timeout=10)
        for path in part_paths:
            if os.path.isfile(path):
                os.remove(path)

    convert = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path, mp3_path],
        capture_output=True, text=True,
    )
    os.remove(wav_path)
    if convert.returncode != 0:
        return False, convert.stderr[-4000:]
    return True, f"{name}.mp3"


def disk_percent() -> float:
    out = sh(["df", "-k", "/"])
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    parts = lines[1].split()
    return float(parts[4].rstrip("%")) if len(parts) > 4 else 0.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/stats":
            payload = {
                "cpu_percent": cpu_percent(),
                "mem": mem_stats(),
                "disk_percent": disk_percent(),
                "load_avg": [round(x, 2) for x in os.getloadavg()],
                "training": training_status(),
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/checkpoints":
            self._json(200, {"checkpoints": list_checkpoints()})
            return

        if self.path.startswith("/progress"):
            query = self.path.split("?", 1)[1] if "?" in self.path else ""
            job_id = ""
            for pair in query.split("&"):
                key, _, value = pair.partition("=")
                if key == "job":
                    job_id = value
            job = job_read(job_id)
            if not job:
                self._json(404, {"ok": False, "error": "khong tim thay job"})
                return
            now = time.time()
            started = job.get("started", now)
            done = job.get("done", 0)
            total = job.get("total", 0)
            job["elapsed"] = round(now - started)
            if done:
                done_at = job.get("done_at", now)
                rate = (done_at - started) / done
                job["eta"] = max(0, round(rate * (total - done) - (now - done_at)))
            else:
                job["eta"] = None
            self._json(200, {"ok": True, **job})
            return

        if self.path in ("/", "/monitor.html"):
            with open(HTML_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/generated/"):
            path_only = self.path.split("?", 1)[0]
            filename = os.path.basename(path_only[len("/generated/"):])
            filepath = os.path.join(OUTPUT_DIR, filename)
            if not os.path.isfile(filepath):
                self.send_response(404)
                self.end_headers()
                return
            with open(filepath, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/generate":
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text", "")).strip()
            checkpoint_name = str(payload.get("checkpoint", "")).strip()
            if not text:
                self._json(400, {"ok": False, "error": "thieu text"})
                return

            try:
                speed = float(payload.get("speed") or SPEED)
            except (TypeError, ValueError):
                speed = SPEED
            speed = min(1.5, max(0.5, speed))

            job_id = f"gen_{int(time.time() * 1000)}"
            parts = split_text(prepare_text(text), max_chunk_bytes(REF_AUDIO, REF_TEXT))
            job_update(job_id, total=len(parts), done=0, state="running", started=time.time())

            def worker() -> None:
                try:
                    ok, result = synthesize(text, checkpoint_name, job_id, speed)
                except subprocess.TimeoutExpired:
                    job_update(job_id, state="error", error="qua thoi gian cho (10 phut)")
                    return
                except Exception as exc:
                    job_update(job_id, state="error", error=str(exc))
                    return
                if ok:
                    job_update(job_id, state="done", file=result)
                else:
                    job_update(job_id, state="error", error=result)

            threading.Thread(target=worker, daemon=True).start()
            self._json(200, {"ok": True, "job": job_id, "total": len(parts)})
            return

        self.send_response(404)
        self.end_headers()

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Monitor dang chay tai http://127.0.0.1:{PORT}/")
    server.serve_forever()

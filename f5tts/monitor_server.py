from __future__ import annotations

import json
import os
import re
import subprocess
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8787
BASE_DIR = os.path.dirname(__file__)
HTML_PATH = os.path.join(BASE_DIR, "monitor.html")
REPO_DIR = os.path.join(BASE_DIR, "F5-TTS-Vietnamese")
VENV_PY = os.path.join(BASE_DIR, ".venv", "bin", "f5-tts_infer-cli")
CKPT_DIR = os.path.join(REPO_DIR, "ckpts", "your_training_dataset")
VOCAB_FILE = os.path.join(REPO_DIR, "data", "your_training_dataset", "vocab.txt")
REF_AUDIO = os.path.join(REPO_DIR, "data", "your_training_dataset", "wavs", "sample_00004.wav")
REF_TEXT = "tình cờ lạc bước vào một ngôi mộ hoang của quý phi đời trước."
OUTPUT_DIR = os.path.join(BASE_DIR, "generated")
SENTENCE_GAP = 0.35
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def max_chunk_bytes() -> int:
    with wave.open(REF_AUDIO) as ref:
        seconds = ref.getnframes() / ref.getframerate()
    if seconds >= 22:
        return len(REF_TEXT.encode("utf-8"))
    return max(1, int(len(REF_TEXT.encode("utf-8")) / seconds * (22 - seconds)))


def pack(pieces: list[str], limit: int) -> list[str]:
    packed: list[str] = []
    buffer = ""
    for piece in pieces:
        merged = f"{buffer} {piece}".strip()
        if buffer and len(merged.encode("utf-8")) > limit:
            packed.append(buffer)
            buffer = piece
        else:
            buffer = merged
    if buffer:
        packed.append(buffer)
    return packed


def split_text(text: str, limit: int) -> list[str]:
    units: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        if len(sentence.encode("utf-8")) <= limit:
            units.append(sentence)
        else:
            units.extend(pack(re.split(r"(?<=[,;:])\s+", sentence), limit))
    return pack(units, limit) or [text]


def join_wavs(sources: list[str], target: str, gap: float) -> None:
    with wave.open(sources[0]) as first:
        params = first.getparams()
    frame_size = params.sampwidth * params.nchannels
    silence = b"\x00" * (int(gap * params.framerate) * frame_size)
    with wave.open(target, "wb") as out:
        out.setparams(params)
        for position, source in enumerate(sources):
            if position:
                out.writeframes(silence)
            with wave.open(source) as part:
                out.writeframes(part.readframes(part.getnframes()))


def synthesize(text: str, checkpoint_name: str = "") -> tuple[bool, str]:
    text = text.lower()
    name = f"gen_{int(time.time() * 1000)}"
    wav_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
    mp3_path = os.path.join(OUTPUT_DIR, f"{name}.mp3")
    checkpoint = resolve_checkpoint(checkpoint_name)
    parts = split_text(text, max_chunk_bytes())
    part_paths: list[str] = []

    try:
        for position, part in enumerate(parts):
            part_name = f"{name}_{position:03d}.wav"
            cmd = [
                "arch", "-arm64", VENV_PY,
                "--model", "F5TTS_Base",
                "--ckpt_file", checkpoint,
                "--vocab_file", VOCAB_FILE,
                "--ref_audio", REF_AUDIO,
                "--ref_text", REF_TEXT,
                "--gen_text", part,
                "--vocoder_name", "vocos",
                "--output_dir", OUTPUT_DIR,
                "--output_file", part_name,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_DIR, timeout=600)
            if proc.returncode != 0:
                return False, proc.stderr[-4000:]
            part_paths.append(os.path.join(OUTPUT_DIR, part_name))

        join_wavs(part_paths, wav_path, SENTENCE_GAP)
    finally:
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
                ok, result = synthesize(text, checkpoint_name)
            except subprocess.TimeoutExpired:
                self._json(504, {"ok": False, "error": "qua thoi gian cho (10 phut)"})
                return
            except Exception as exc:
                self._json(500, {"ok": False, "error": str(exc)})
                return
            if ok:
                self._json(200, {"ok": True, "file": result})
            else:
                self._json(500, {"ok": False, "error": result})
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

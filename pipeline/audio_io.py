from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import soxr


class AudioError(RuntimeError):
    pass


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if not exe:
        raise AudioError(f"{name} khong co trong PATH (brew install ffmpeg)")
    return exe


def sh(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"lenh loi: {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc.stdout


def probe(path) -> dict:
    out = sh([_tool("ffprobe"), "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)])
    return json.loads(out)


def source_summary(path) -> dict:
    info = probe(path)
    stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = info.get("format", {})
    return {
        "path": str(path),
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream.get("sample_rate", 0) or 0),
        "channels": int(stream.get("channels", 0) or 0),
        "bit_rate": int(float(fmt.get("bit_rate", 0) or 0)),
        "duration": float(fmt.get("duration", 0) or 0),
    }


def decode(src, dst, sample_rate: int, highpass_hz: float = 0.0,
           preview_seconds: float = 0.0, extra_filters: tuple[str, ...] = ()) -> None:
    filters = []
    if highpass_hz and highpass_hz > 0:
        filters.append(f"highpass=f={int(highpass_hz)}")
    filters.extend(extra_filters)
    cmd = [_tool("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if preview_seconds and preview_seconds > 0:
        cmd += ["-t", f"{preview_seconds:.3f}"]
    cmd += ["-i", str(src), "-vn", "-map", "a:0", "-ac", "1", "-ar", str(int(sample_rate))]
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-c:a", "pcm_f32le", str(dst)]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sh(cmd)


def read_all(path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def read_slice(path, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as f:
        sr = int(f.samplerate)
        a = max(0, int(round(start_s * sr)))
        b = min(len(f), int(round(end_s * sr)))
        if b <= a:
            return np.zeros(0, dtype=np.float32), sr
        f.seek(a)
        x = f.read(b - a, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def duration_of(path) -> float:
    with sf.SoundFile(str(path)) as f:
        return len(f) / float(f.samplerate)


def write_wav(path, x: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(x, -1.0, 1.0), int(sr), subtype=subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if int(sr_in) == int(sr_out):
        return np.ascontiguousarray(x, dtype=np.float32)
    y = soxr.resample(np.asarray(x, dtype=np.float32), int(sr_in), int(sr_out), quality="VHQ")
    return np.ascontiguousarray(y, dtype=np.float32)


def db_to_amp(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def peak_limit(x: np.ndarray, ceiling_db: float) -> np.ndarray:
    if x.size == 0:
        return x
    ceiling = db_to_amp(ceiling_db)
    peak = float(np.max(np.abs(x)))
    if peak > ceiling > 0:
        x = x * (ceiling / peak)
    return np.asarray(x, dtype=np.float32)


def loudness_normalize(x: np.ndarray, sr: int, target_lufs: float, ceiling_db: float) -> np.ndarray:
    if x.size < int(sr * 0.5):
        return peak_limit(x, ceiling_db)
    meter = pyln.Meter(int(sr))
    loudness = float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    if not np.isfinite(loudness):
        return peak_limit(x, ceiling_db)
    gain = db_to_amp(float(target_lufs) - loudness)
    return peak_limit(np.asarray(x, dtype=np.float32) * gain, ceiling_db)


def frame_rms_db(x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000.0))
    h = max(1, int(sr * hop_ms / 1000.0))
    if x.size < n:
        return np.array([-120.0], dtype=np.float32)
    count = 1 + (x.size - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(count)[:, None]
    rms = np.sqrt(np.mean(np.asarray(x[idx], dtype=np.float64) ** 2, axis=1) + 1e-12)
    return np.asarray(20.0 * np.log10(rms), dtype=np.float32)


def snr_db(x: np.ndarray, sr: int) -> float:
    d = frame_rms_db(x, sr)
    if d.size < 8:
        return 0.0
    return float(np.percentile(d, 92) - np.percentile(d, 8))


def clip_ratio(x: np.ndarray, threshold: float = 0.995) -> float:
    if x.size == 0:
        return 1.0
    return float(np.mean(np.abs(x) >= threshold))


def dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2) + 1e-12))
    return float(20.0 * np.log10(rms))


def trim_silence(x: np.ndarray, sr: int, top_db: float = 40.0, pad_ms: float = 30.0) -> np.ndarray:
    d = frame_rms_db(x, sr)
    if d.size < 3:
        return x
    voiced = np.nonzero(d > (float(np.max(d)) - float(top_db)))[0]
    if voiced.size == 0:
        return x
    hop = max(1, int(sr * 0.01))
    pad = int(sr * pad_ms / 1000.0)
    a = max(0, int(voiced[0]) * hop - pad)
    b = min(x.size, int(voiced[-1]) * hop + int(sr * 0.02) + pad)
    return x[a:b]

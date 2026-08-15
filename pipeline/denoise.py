from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import audio_io

BACKENDS = ("none", "ffmpeg", "noisereduce", "demucs")


def needs_noise_profile(backend: str) -> bool:
    return str(backend).lower() == "noisereduce"


def apply(cfg: dict, src: Path, dst: Path, noise_regions: list[tuple[float, float]]) -> str:
    dcfg = cfg["denoise"]
    backend = str(dcfg["backend"]).lower()
    sample_rate = int(cfg["audio"]["sample_rate"])
    if backend not in BACKENDS:
        raise ValueError(f"denoise.backend khong hop le: {backend} (chon {BACKENDS})")
    if backend == "none":
        shutil.copyfile(src, dst)
    elif backend == "ffmpeg":
        _ffmpeg(dcfg, src, dst, sample_rate)
    elif backend == "noisereduce":
        _noisereduce(dcfg, src, dst, noise_regions, sample_rate)
    else:
        _demucs(dcfg, src, dst, sample_rate)
    return backend


def _ffmpeg(dcfg: dict, src: Path, dst: Path, sample_rate: int) -> None:
    nf = int(dcfg.get("ffmpeg_afftdn_nf", -25))
    audio_io.decode(src, dst, sample_rate, extra_filters=(f"afftdn=nf={nf}",))


def collect_noise(src: Path, regions: list[tuple[float, float]], max_seconds: float) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    total = 0.0
    for start, end in sorted(regions, key=lambda r: r[1] - r[0], reverse=True):
        if end - start < 0.30:
            continue
        seg, sr = audio_io.read_slice(src, start + 0.05, end - 0.05)
        if seg.size < sr // 10:
            continue
        parts.append(seg)
        total += seg.size / sr
        if total >= max_seconds:
            break
    if not parts:
        return None
    return np.concatenate(parts)


def _noisereduce(dcfg: dict, src: Path, dst: Path,
                 noise_regions: list[tuple[float, float]], sample_rate: int) -> None:
    import noisereduce as nr

    x, sr = audio_io.read_all(src)
    noise = collect_noise(src, noise_regions, float(dcfg.get("noise_profile_max_seconds", 30)))
    prop = float(dcfg.get("strength", 0.6))
    block = max(int(float(dcfg.get("block_seconds", 60)) * sr), sr)
    context = max(int(float(dcfg.get("overlap_seconds", 1.0)) * sr), 0)

    out = np.zeros_like(x)
    for start in range(0, x.size, block):
        lo = max(0, start - context)
        hi = min(x.size, start + block + context)
        chunk = np.asarray(x[lo:hi], dtype=np.float32)
        cleaned = nr.reduce_noise(
            y=chunk, sr=sr, y_noise=noise,
            stationary=noise is not None, prop_decrease=prop,
        )
        cleaned = np.asarray(cleaned, dtype=np.float32).reshape(-1)
        keep_lo, keep_hi = start, min(x.size, start + block)
        out[keep_lo:keep_hi] = cleaned[keep_lo - lo: keep_hi - lo]

    audio_io.write_wav(dst, audio_io.resample(out, sr, sample_rate), sample_rate, subtype="FLOAT")


def _demucs(dcfg: dict, src: Path, dst: Path, sample_rate: int) -> None:
    model = str(dcfg.get("demucs_model", "htdemucs"))
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, "-m", "demucs", "-n", model, "--two-stems=vocals",
               "--float32", "-o", td, str(src)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"demucs loi (pip install demucs):\n{proc.stderr[-2000:]}")
        found = sorted(Path(td).rglob("vocals.wav"))
        if not found:
            raise RuntimeError("demucs khong tao ra vocals.wav")
        x, sr = audio_io.read_all(found[0])
    audio_io.write_wav(dst, audio_io.resample(x, sr, sample_rate), sample_rate, subtype="FLOAT")

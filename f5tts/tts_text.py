from __future__ import annotations

import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
if PIPELINE_DIR.is_dir():
    sys.path.append(str(PIPELINE_DIR))

import vi_norm

SENTENCE_GAP = 0.35
SPEED = 0.85
WINDOW_SECONDS = 22
MAX_CHUNK_BYTES = 230
PAUSE_CHARS = ":;-–—()[]"
HARD_BREAK = "\x00"
DROP_CHARS = "\"'“”‘’«"
SHORT_BYTES = 25
LONG_BYTES = 115
SHORT_SPEED_RATIO = 0.76
TAIL_WINDOW = 0.02
TAIL_DROP_DB = 25.0
TAIL_FADE_IN = 0.003
SAMPLE_DTYPES = {1: np.int8, 2: np.int16, 4: np.int32}
LOUDNESS_LUFS = -14.0
TRUE_PEAK_DB = -1.5
LOUDNESS_RANGE = 11.0
MP3_BITRATE = "64k"
MEASURED_KEYS = {
    "input_i": "measured_I",
    "input_tp": "measured_TP",
    "input_lra": "measured_LRA",
    "input_thresh": "measured_thresh",
}
TEXT_CONFIG = {
    "le_word": "lẻ",
    "thousand_word": "nghìn",
    "four_after_ten": "tư",
    "decimal_by_digit": True,
    "spell_unknown_acronyms": True,
    "lowercase": False,
    "keep_punctuation": ",.!?…",
}
NORMALIZER = vi_norm.build(TEXT_CONFIG)


def prepare_text(text: str) -> str:
    text = re.sub(r"(\.{2,}|…)(?=\s*$)", ".", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]*([.!?])?[ \t]*»",
                  lambda m: f"{m.group(1) or '.'}{HARD_BREAK}", text)
    text = re.sub(r"(?m)[ \t]*:[ \t]*$", f".{HARD_BREAK}", text)
    text = re.sub(r"[ \t]*\n[\s\n]*", HARD_BREAK, text)
    text = text.lower()
    text = text.translate(str.maketrans(PAUSE_CHARS, "," * len(PAUSE_CHARS)))
    text = text.translate({ord(c): None for c in DROP_CHARS})
    text = re.sub(r"\.{2,}|…", ",", text)
    text = re.sub(r"\s*!+[\s.,!]*", ". ", text)
    text = re.sub(r"\s*,[\s,]*", ", ", text)
    text = re.sub(r",\s*(?=[.!?…])", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?m)^[\s,]+", "", text)
    return re.sub(r"[\s,]+$", "", text).strip()


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


def demote_inner_questions(chunk: str) -> str:
    if len(chunk) < 2:
        return chunk
    return chunk[:-1].replace("?", ".") + chunk[-1]


def pack_sentences(block: str, limit: int) -> list[str]:
    units: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", block.strip()):
        sentence = NORMALIZER(sentence)
        if not sentence:
            continue
        if len(sentence.encode("utf-8")) <= limit:
            units.append(sentence)
        else:
            units.extend(pack(re.split(r"(?<=[,;:])\s+", sentence), limit))
    return pack(units, limit)


def split_text(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    for block in text.split(HARD_BREAK):
        block = block.strip(" \t\n,")
        if re.search(r"[^\W_]", block):
            chunks.extend(pack_sentences(block, limit))
    return [demote_inner_questions(c) for c in chunks] or [text.replace(HARD_BREAK, " ").strip()]


def speed_for(chunk: str, base: float = SPEED) -> float:
    size = len(chunk.encode("utf-8"))
    if size >= LONG_BYTES:
        return base
    if size <= SHORT_BYTES:
        return base * SHORT_SPEED_RATIO
    ramp = (size - SHORT_BYTES) / (LONG_BYTES - SHORT_BYTES)
    return base * (SHORT_SPEED_RATIO + (1 - SHORT_SPEED_RATIO) * ramp)


def max_chunk_bytes(ref_audio: str, ref_text: str) -> int:
    with wave.open(ref_audio) as ref:
        seconds = ref.getnframes() / ref.getframerate()
    ref_bytes = len(ref_text.encode("utf-8"))
    if seconds >= WINDOW_SECONDS:
        return min(MAX_CHUNK_BYTES, ref_bytes)
    window = int(ref_bytes / seconds * (WINDOW_SECONDS - seconds))
    return max(1, min(MAX_CHUNK_BYTES, window))


def ramp_tail(samples: np.ndarray, framerate: int) -> np.ndarray:
    window = int(TAIL_WINDOW * framerate)
    count = len(samples) // window if window else 0
    if count < 2:
        return samples
    block = samples[:count * window].reshape(count, window).astype(np.float64)
    levels = np.sqrt(np.mean(block ** 2, axis=1))
    peak = levels.max()
    if peak <= 0:
        return samples
    voiced = np.flatnonzero(levels > peak * 10.0 ** (-TAIL_DROP_DB / 20.0))
    if not len(voiced):
        return samples
    shaped = samples.astype(np.float64)
    start = min((voiced[-1] + 1) * window, len(shaped))
    if len(shaped) - start > 1:
        shaped[start:] *= np.linspace(1.0, 0.0, len(shaped) - start) ** 2
    lead = int(TAIL_FADE_IN * framerate)
    if lead > 1:
        shaped[:lead] *= np.linspace(0.0, 1.0, lead)
    return np.rint(shaped).astype(samples.dtype)


def join_wavs(sources: list[str], target: str, gap: float = SENTENCE_GAP) -> None:
    with wave.open(sources[0]) as first:
        params = first.getparams()
    frame_size = params.sampwidth * params.nchannels
    silence = b"\x00" * (int(gap * params.framerate) * frame_size)
    dtype = SAMPLE_DTYPES.get(params.sampwidth) if params.nchannels == 1 else None
    with wave.open(target, "wb") as out:
        out.setparams(params)
        for position, source in enumerate(sources):
            if position:
                out.writeframes(silence)
            with wave.open(source) as part:
                raw = part.readframes(part.getnframes())
            if dtype is None:
                out.writeframes(raw)
                continue
            out.writeframes(ramp_tail(np.frombuffer(raw, dtype=dtype), params.framerate).tobytes())


def measure_loudness(wav_path: str) -> dict:
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", wav_path,
         "-af", f"loudnorm=I={LOUDNESS_LUFS}:TP={TRUE_PEAK_DB}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    blocks = re.findall(r"\{[^{}]*\}", probe.stderr)
    if not blocks:
        return {}
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return {}


def loudnorm_filter(wav_path: str) -> str:
    stats = measure_loudness(wav_path)
    measured = {name: stats.get(key) for key, name in MEASURED_KEYS.items()}
    if any(str(value) in ("None", "-inf", "inf", "nan") for value in measured.values()):
        return ""
    pairs = ":".join(f"{name}={value}" for name, value in measured.items())
    return (f"loudnorm=I={LOUDNESS_LUFS}:TP={TRUE_PEAK_DB}:LRA={LOUDNESS_RANGE}"
            f":{pairs}:linear=true")


def wav_to_mp3(wav_path: str, mp3_path: str) -> tuple[bool, str]:
    with wave.open(wav_path) as source:
        framerate = source.getframerate()
    audio_filter = loudnorm_filter(wav_path)
    convert = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path]
        + (["-af", audio_filter] if audio_filter else [])
        + ["-ar", str(framerate), "-b:a", MP3_BITRATE, mp3_path],
        capture_output=True, text=True,
    )
    return convert.returncode == 0, convert.stderr[-4000:]

from __future__ import annotations

import re
import wave

SENTENCE_GAP = 0.35
SPEED = 0.95
WINDOW_SECONDS = 22
MAX_CHUNK_BYTES = 230
PAUSE_CHARS = ":;-–—()[]"
HARD_BREAK = "\x00"
DROP_CHARS = "\"'“”‘’«"


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


def max_chunk_bytes(ref_audio: str, ref_text: str) -> int:
    with wave.open(ref_audio) as ref:
        seconds = ref.getnframes() / ref.getframerate()
    ref_bytes = len(ref_text.encode("utf-8"))
    if seconds >= WINDOW_SECONDS:
        return min(MAX_CHUNK_BYTES, ref_bytes)
    window = int(ref_bytes / seconds * (WINDOW_SECONDS - seconds))
    return max(1, min(MAX_CHUNK_BYTES, window))


def join_wavs(sources: list[str], target: str, gap: float = SENTENCE_GAP) -> None:
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

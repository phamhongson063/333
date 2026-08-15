from __future__ import annotations

import bisect

import numpy as np

SENTENCE_END = ".!?…"
TRAILING_WRAP = "\"')]}»"


def speech_regions(x16: np.ndarray, sr: int, vcfg: dict) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad(onnx=bool(vcfg.get("use_onnx", True)))
    try:
        model.reset_states()
    except AttributeError:
        pass
    stamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(x16, dtype=np.float32)),
        model,
        sampling_rate=int(sr),
        threshold=float(vcfg.get("threshold", 0.5)),
        min_speech_duration_ms=int(vcfg.get("min_speech_ms", 250)),
        min_silence_duration_ms=int(vcfg.get("min_silence_ms", 300)),
        speech_pad_ms=int(vcfg.get("speech_pad_ms", 60)),
        return_seconds=True,
    )
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def silence_regions(regions: list[tuple[float, float]], total: float,
                    min_length: float = 0.05) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    prev = 0.0
    for start, end in regions:
        if start - prev >= min_length:
            out.append((prev, start))
        prev = max(prev, end)
    if total - prev >= min_length:
        out.append((prev, total))
    return out


def frame_levels(x: np.ndarray, sr: int, hop_ms: float = 10.0) -> tuple[np.ndarray, int]:
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    count = int(x.size // hop)
    if count == 0:
        return np.zeros(0, dtype=np.float32), hop
    levels = np.empty(count, dtype=np.float32)
    block = max(1, 2_000_000 // hop)
    for begin in range(0, count, block):
        stop = min(count, begin + block)
        chunk = np.asarray(x[begin * hop: stop * hop], dtype=np.float32)
        frames = chunk.reshape(stop - begin, hop)
        power = np.mean(frames * frames, axis=1, dtype=np.float64)
        levels[begin:stop] = 10.0 * np.log10(power + 1e-12)
    return levels, hop


def silence_intervals(x: np.ndarray, sr: int, speech: list[tuple[float, float]],
                      scfg: dict) -> tuple[list[tuple[float, float]], float]:
    levels, hop = frame_levels(x, sr, float(scfg.get("silence_hop_ms", 10.0)))
    if levels.size == 0:
        return [], -120.0

    step = hop / float(sr)
    inside = np.zeros(levels.size, dtype=bool)
    for start, end in speech:
        lo = max(0, int(start / step))
        hi = min(levels.size, int(np.ceil(end / step)))
        if hi > lo:
            inside[lo:hi] = True
    reference = float(np.median(levels[inside])) if bool(inside.any()) else float(np.median(levels))
    threshold = reference - float(scfg.get("silence_offset_db", 25.0))

    quiet = levels < threshold
    edges = np.diff(quiet.astype(np.int8))
    begins = (np.nonzero(edges == 1)[0] + 1).tolist()
    ends = (np.nonzero(edges == -1)[0] + 1).tolist()
    if bool(quiet[0]):
        begins.insert(0, 0)
    if bool(quiet[-1]):
        ends.append(int(quiet.size))

    min_length = float(scfg.get("min_cut_silence", 0.10))
    intervals = []
    for lo, hi in zip(begins, ends):
        start, end = lo * step, hi * step
        if end - start >= min_length:
            intervals.append((round(start, 4), round(end, 4)))
    return intervals, round(threshold, 2)


def _flatten_words(asr_segments: list[dict]) -> list[dict]:
    words: list[dict] = []
    for index, seg in enumerate(asr_segments):
        listed = seg.get("words") or []
        if listed:
            for w in listed:
                text = str(w.get("word", "")).strip()
                if not text:
                    continue
                words.append({
                    "text": text,
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "prob": float(w.get("probability", 1.0)),
                    "ctc_score": float(w.get("ctc_score", 1.0)),
                    "seg": index,
                })
        else:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            words.append({
                "text": text,
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "prob": float(seg.get("prob", 1.0)),
                "seg": index,
            })
    words.sort(key=lambda w: (w["start"], w["end"]))
    for a, b in zip(words, words[1:]):
        if b["start"] < a["end"]:
            b["start"] = a["end"]
        if b["end"] < b["start"]:
            b["end"] = b["start"]
    return words


def flatten_words(asr_segments: list[dict]) -> list[dict]:
    return _flatten_words(asr_segments)


def word_rate(word: dict) -> float:
    length = len(word["text"].strip(".,!?…;:\"')]}»")) or 1
    return (word["end"] - word["start"]) / length


def rate_limit(words: list[dict], factor: float, floor: float = 0.12) -> float:
    if not words:
        return floor
    return max(floor, float(np.median([word_rate(w) for w in words])) * factor)


def cut_points(silences: list[tuple[float, float]], words: list[dict], pad: float,
               max_overlap: float = 0.8, max_rate: float = 0.16) -> list[dict]:
    if not words:
        return []
    starts = [w["start"] for w in words]
    points: list[dict] = []
    for start, end in silences:
        length = end - start
        if length <= 0:
            continue
        stretched = False
        i = bisect.bisect_right(starts, end) - 1
        while i >= 0 and words[i]["end"] > start:
            overlap = min(words[i]["end"], end) - max(words[i]["start"], start)
            if overlap / length >= max_overlap and word_rate(words[i]) >= max_rate:
                stretched = True
                break
            i -= 1
        if stretched:
            continue
        middle = (start + end) / 2.0
        points.append({
            "time": middle,
            "before": min(middle, start + pad),
            "after": max(middle, end - pad),
            "length": length,
        })
    return points


def _ends_sentence(words: list[dict]) -> bool:
    if not words:
        return False
    tail = words[-1]["text"].rstrip(TRAILING_WRAP)
    return tail[-1:] in tuple(SENTENCE_END)


def _blocks(words: list[dict], cuts: list[dict], total: float) -> list[dict]:
    bounds = [{"time": 0.0, "before": 0.0, "after": 0.0}]
    bounds += sorted(cuts, key=lambda c: c["time"])
    bounds.append({"time": total, "before": total, "after": total})

    middles = [(w["start"] + w["end"]) / 2.0 for w in words]
    blocks: list[dict] = []
    for left, right in zip(bounds, bounds[1:]):
        lo = bisect.bisect_left(middles, left["time"])
        hi = bisect.bisect_left(middles, right["time"])
        blocks.append({
            "span_start": left["after"],
            "span_end": right["before"],
            "words": words[lo:hi],
        })
    return blocks


def _groups(blocks: list[dict], scfg: dict) -> list[dict]:
    min_duration = float(scfg["min_duration"])
    target = float(scfg["target_duration"])
    max_duration = float(scfg["max_duration"])

    groups: list[dict] = []
    current: dict | None = None
    for block in blocks:
        if not block["words"]:
            current = None
            continue
        if current is None:
            current = {"start": block["span_start"], "end": block["span_end"],
                       "words": list(block["words"])}
            groups.append(current)
            continue

        span = current["end"] - current["start"]
        joined = block["span_end"] - current["start"]
        must_extend = span < min_duration and joined <= max_duration
        want_stop = joined > target or (_ends_sentence(current["words"]) and span >= min_duration)
        if must_extend or not want_stop:
            current["end"] = block["span_end"]
            current["words"] += block["words"]
            continue
        current = {"start": block["span_start"], "end": block["span_end"],
                   "words": list(block["words"])}
        groups.append(current)
    return groups


def build(asr_segments: list[dict], silences: list[tuple[float, float]],
          total: float, scfg: dict) -> tuple[list[dict], dict]:
    words = _flatten_words(asr_segments)
    stats = {"silences": len(silences), "cuts": 0, "blocks": 0, "groups": 0,
             "oversize": 0, "words": len(words)}
    if not words:
        return [], stats

    pad = float(scfg.get("pad_ms", 80)) / 1000.0
    max_rate = rate_limit(words, float(scfg.get("max_word_rate_factor", 2.0)))
    cuts = cut_points(silences, words, pad,
                      float(scfg.get("max_word_silence_overlap", 0.8)), max_rate)
    stats["word_rate_limit"] = round(max_rate, 4)
    blocks = _blocks(words, cuts, total)
    groups = _groups(blocks, scfg)
    stats.update(cuts=len(cuts), blocks=len(blocks), groups=len(groups))

    max_duration = float(scfg["max_duration"])
    segments: list[dict] = []
    for index, group in enumerate(groups):
        members = group["words"]
        duration = group["end"] - group["start"]
        if duration > max_duration:
            stats["oversize"] += 1
        probs = [w["prob"] for w in members]
        ctc = [float(w.get("ctc_score", 1.0)) for w in members]
        owners = {w["seg"] for w in members}
        no_speech = max((float(asr_segments[i].get("no_speech_prob", 0.0)) for i in owners),
                        default=0.0)
        raw = " ".join(w["text"] for w in members)
        segments.append({
            "index": index,
            "start": round(group["start"], 3),
            "end": round(group["end"], 3),
            "duration": round(duration, 3),
            "speech_start": round(members[0]["start"], 3),
            "speech_end": round(members[-1]["end"], 3),
            "raw_text": " ".join(raw.split()),
            "asr_prob": round(float(np.mean(probs)) if probs else 1.0, 4),
            "min_word_prob": round(float(np.min(probs)) if probs else 1.0, 4),
            "ctc_score": round(float(np.mean(ctc)) if ctc else 1.0, 4),
            "min_ctc_word_score": round(float(np.min(ctc)) if ctc else 1.0, 4),
            "no_speech_prob": round(no_speech, 4),
            "word_count": len(members),
        })
    return segments, stats


def boundary_report(x: np.ndarray, sr: int, segments: list[dict], threshold: float,
                    half_ms: float = 40.0) -> list[tuple[int, str, float]]:
    half = max(1, int(sr * half_ms / 1000.0))
    violations: list[tuple[int, str, float]] = []
    for item in segments:
        for label in ("start", "end"):
            centre = int(float(item[label]) * sr)
            lo, hi = max(0, centre - half), min(x.size, centre + half)
            if hi <= lo:
                continue
            window = np.asarray(x[lo:hi], dtype=np.float64)
            level = 10.0 * np.log10(float(np.mean(window * window)) + 1e-12)
            if level > threshold:
                violations.append((int(item["index"]), label, round(level, 1)))
    return violations

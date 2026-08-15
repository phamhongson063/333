from __future__ import annotations

import unicodedata

import numpy as np


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def silence_purity(words: list[dict], silences: list[tuple[float, float]],
                   min_length: float = 0.10, max_length: float = 1.0,
                   max_coverage: float = 0.2) -> float:
    import bisect

    targets = [(s, e) for s, e in silences if min_length <= e - s <= max_length]
    if not targets or not words:
        return 0.0
    starts = [w["start"] for w in words]
    clean = 0
    for start, end in targets:
        covered = 0.0
        pivot = bisect.bisect_left(starts, start)
        i = pivot - 1
        while i >= 0 and words[i]["end"] > start:
            covered += min(words[i]["end"], end) - max(words[i]["start"], start)
            i -= 1
        i = pivot
        while i < len(words) and words[i]["start"] < end:
            covered += min(words[i]["end"], end) - max(words[i]["start"], start)
            i += 1
        if covered / (end - start) < max_coverage:
            clean += 1
    return clean / len(targets)


def chunk_bounds(silences: list[tuple[float, float]], total: float,
                 target: float, min_silence: float) -> list[float]:
    centres = [(s + e) / 2.0 for s, e in silences if e - s >= min_silence]
    bounds = [0.0]
    for centre in centres:
        if centre - bounds[-1] >= target:
            bounds.append(centre)
    if total - bounds[-1] < target * 0.5 and len(bounds) > 1:
        bounds[-1] = total
    else:
        bounds.append(total)
    return bounds


class Aligner:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.torch = torch
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).eval()
        self.device = device
        try:
            self.model = self.model.to(device)
        except Exception:
            self.device = "cpu"
            self.model = self.model.to("cpu")
        self.vocab = dict(self.processor.tokenizer.get_vocab())
        self.inverse = {i: t for t, i in self.vocab.items()}
        self.delimiter = self.vocab.get("|")
        pad = getattr(self.processor.tokenizer, "pad_token", "<pad>")
        self.blank = int(self.vocab.get(pad, 0))

    def normalise(self, text: str) -> str:
        lowered = unicodedata.normalize("NFC", str(text)).lower()
        return "".join(ch for ch in lowered if ch != "|" and ch in self.vocab)

    def _emission(self, audio: np.ndarray):
        torch = self.torch
        values = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None]
        with torch.inference_mode():
            try:
                logits = self.model(values.to(self.device)).logits
            except Exception:
                self.device = "cpu"
                self.model = self.model.to("cpu")
                logits = self.model(values).logits
        return torch.log_softmax(logits.float().cpu(), dim=-1)

    def _tokenise(self, texts: list[str]):
        normalised = [self.normalise(t) for t in texts]
        usable = [i for i, text in enumerate(normalised) if text]
        if not usable:
            return None, None
        tokens: list[int] = []
        owner: list[int] = []
        for position, index in enumerate(usable):
            if position:
                tokens.append(self.delimiter)
                owner.append(-1)
            for char in normalised[index]:
                tokens.append(int(self.vocab[char]))
                owner.append(index)
        return tokens, owner

    def emission(self, audio: np.ndarray):
        return self._emission(audio)

    def decode_from(self, emission) -> str:
        ids = emission[0].argmax(-1).tolist()
        picked: list[int] = []
        previous = None
        for token in ids:
            if token != previous and token != self.blank:
                picked.append(token)
            previous = token
        text = "".join(self.inverse.get(t, "") for t in picked)
        return " ".join(text.replace("|", " ").split())

    def text_length(self, text: str) -> int:
        return len(self.normalise(str(text)).replace(" ", ""))

    def completeness(self, text: str, decoded: str) -> float:
        reference = self.text_length(decoded)
        if reference <= 0:
            return 1.0
        return self.text_length(text) / reference

    def score_from(self, emission, samples: int, sr: int, text: str) -> float | None:
        pieces = str(text).split()
        if not pieces or self.delimiter is None:
            return None
        tokens, owner = self._tokenise(pieces)
        if not tokens:
            return None
        spans, _ = self._spans_from(emission, samples, sr, tokens)
        if spans is None:
            return None
        values = [float(s.score) for s, o in zip(spans, owner) if o >= 0]
        return float(np.mean(values)) if values else None

    def _spans_from(self, emission, samples: int, sr: int, tokens: list[int]):
        import torchaudio

        if emission.shape[1] < len(tokens):
            return None, None
        targets = self.torch.tensor([tokens], dtype=self.torch.int32)
        try:
            labels, scores = torchaudio.functional.forced_align(
                emission, targets, blank=self.blank)
            spans = torchaudio.functional.merge_tokens(
                labels[0], scores[0].exp(), blank=self.blank)
        except Exception:
            return None, None
        if len(spans) != len(tokens):
            return None, None
        return spans, samples / float(emission.shape[1]) / float(sr)

    def _spans(self, audio: np.ndarray, sr: int, tokens: list[int]):
        return self._spans_from(self._emission(audio), audio.size, sr, tokens)

    def align(self, audio: np.ndarray, sr: int,
              words: list[dict]) -> dict[int, tuple[float, float, float]] | None:
        if audio.size < sr // 10 or not words or self.delimiter is None:
            return None
        tokens, owner = self._tokenise([w["text"] for w in words])
        if not tokens:
            return None
        spans, ratio = self._spans(audio, sr, tokens)
        if spans is None:
            return None

        gathered: dict[int, list] = {}
        for span, index in zip(spans, owner):
            if index < 0:
                continue
            gathered.setdefault(index, []).append(
                (span.start * ratio, span.end * ratio, float(span.score)))
        return {
            index: (min(p[0] for p in parts), max(p[1] for p in parts),
                    float(np.mean([p[2] for p in parts])))
            for index, parts in gathered.items()
        }

    def score_text(self, audio: np.ndarray, sr: int, text: str) -> float | None:
        if audio.size < sr // 10 or self.delimiter is None:
            return None
        pieces = str(text).split()
        if not pieces:
            return None
        tokens, owner = self._tokenise(pieces)
        if not tokens:
            return None
        spans, _ = self._spans(audio, sr, tokens)
        if spans is None:
            return None
        values = [float(s.score) for s, o in zip(spans, owner) if o >= 0]
        return float(np.mean(values)) if values else None


def run(cfg: dict, x: np.ndarray, sr: int, asr_segments: list[dict],
        words: list[dict], silences: list[tuple[float, float]], total: float,
        progress=None) -> tuple[list[dict], dict]:
    acfg = cfg["align"]
    device = resolve_device(str(acfg.get("device", "auto")))
    aligner = Aligner(str(acfg["model"]), device)

    target = float(acfg.get("chunk_seconds", 20.0))
    margin = float(acfg.get("chunk_margin", 2.5))
    bounds = chunk_bounds(silences, total, target,
                          float(acfg.get("min_boundary_silence", 0.4)))

    middles = [(w["start"] + w["end"]) / 2.0 for w in words]
    realigned = 0
    failed_chunks = 0
    for position, (left, right) in enumerate(zip(bounds, bounds[1:])):
        members = [i for i, m in enumerate(middles) if left <= m < right]
        if not members:
            continue
        lo = max(0, int((left - margin) * sr))
        hi = min(x.size, int((right + margin) * sr))
        offset = lo / float(sr)
        times = aligner.align(x[lo:hi], sr, [words[i] for i in members])
        if not times:
            failed_chunks += 1
            continue
        for local, index in enumerate(members):
            hit = times.get(local)
            if not hit:
                continue
            start, end = offset + hit[0], offset + hit[1]
            if end <= start:
                continue
            words[index]["start"] = round(start, 3)
            words[index]["end"] = round(end, 3)
            words[index]["ctc_score"] = round(float(hit[2]), 4)
            words[index]["realigned"] = True
            realigned += 1
        if progress is not None:
            progress(position + 1, len(bounds) - 1)

    updated = [dict(seg) for seg in asr_segments]
    grouped: dict[int, list[dict]] = {}
    for word in words:
        grouped.setdefault(int(word["seg"]), []).append(word)
    for index, seg in enumerate(updated):
        members = grouped.get(index)
        if not members:
            continue
        seg["words"] = [{
            "word": w["text"],
            "start": float(w["start"]),
            "end": float(w["end"]),
            "probability": float(w["prob"]),
            "ctc_score": float(w.get("ctc_score", 1.0)),
            "realigned": bool(w.get("realigned", False)),
        } for w in members]
        seg["start"] = float(min(w["start"] for w in members))
        seg["end"] = float(max(w["end"] for w in members))

    stats = {
        "chunks": len(bounds) - 1,
        "failed_chunks": failed_chunks,
        "words": len(words),
        "realigned": realigned,
        "device": aligner.device,
        "blank_id": aligner.blank,
    }
    return updated, stats

from __future__ import annotations

from pathlib import Path

import numpy as np

import audio_io

MLX_ALIASES = ("mlx", "mlx-whisper", "mlx_whisper")
FASTER_ALIASES = ("faster", "faster-whisper", "faster_whisper", "fw")


def transcribe(cfg: dict, audio_path: Path) -> list[dict]:
    backend = str(cfg["asr"]["backend"]).lower()
    if backend in MLX_ALIASES:
        return _mlx(cfg, audio_path)
    if backend in FASTER_ALIASES:
        return _faster_whisper(cfg, audio_path)
    raise ValueError(f"asr.backend khong hop le: {backend}")


def _normalise(start, end, text, words, no_speech, avg_logprob) -> dict:
    return {
        "start": float(start),
        "end": float(end),
        "text": str(text).strip(),
        "no_speech_prob": float(no_speech or 0.0),
        "avg_logprob": float(avg_logprob or 0.0),
        "words": words,
    }


def _mlx(cfg: dict, audio_path: Path) -> list[dict]:
    import mlx_whisper

    a = cfg["asr"]
    kwargs = {
        "path_or_hf_repo": str(a["mlx_model"]),
        "language": str(a["language"]),
        "task": "transcribe",
        "word_timestamps": True,
        "verbose": False,
        "condition_on_previous_text": bool(a.get("condition_on_previous_text", False)),
    }
    if a.get("initial_prompt"):
        kwargs["initial_prompt"] = str(a["initial_prompt"])
    result = mlx_whisper.transcribe(str(audio_path), **kwargs)

    segments = []
    for seg in result.get("segments", []):
        words = [{
            "word": str(w.get("word", "")),
            "start": float(w.get("start", seg["start"])),
            "end": float(w.get("end", seg["end"])),
            "probability": float(w.get("probability", 1.0)),
        } for w in (seg.get("words") or [])]
        segments.append(_normalise(seg["start"], seg["end"], seg.get("text", ""), words,
                                   seg.get("no_speech_prob"), seg.get("avg_logprob")))
    return segments


def _faster_whisper(cfg: dict, audio_path: Path) -> list[dict]:
    from faster_whisper import WhisperModel

    a = cfg["asr"]
    model = WhisperModel(str(a["faster_whisper_model"]), device="cpu",
                         compute_type=str(a.get("faster_whisper_compute", "int8")))
    iterator, _ = model.transcribe(
        str(audio_path),
        language=str(a["language"]),
        task="transcribe",
        beam_size=int(a.get("beam_size", 5)),
        word_timestamps=True,
        vad_filter=False,
        condition_on_previous_text=bool(a.get("condition_on_previous_text", False)),
        initial_prompt=str(a["initial_prompt"]) or None if a.get("initial_prompt") else None,
    )

    segments = []
    for seg in iterator:
        words = [{
            "word": str(w.word),
            "start": float(w.start),
            "end": float(w.end),
            "probability": float(w.probability),
        } for w in (seg.words or [])]
        segments.append(_normalise(seg.start, seg.end, seg.text, words,
                                   seg.no_speech_prob, seg.avg_logprob))
    return segments


_FASTER_CACHE: dict[str, object] = {}


def transcribe_array(cfg: dict, audio: np.ndarray, sr: int) -> str:
    a = cfg["asr"]
    backend = str(a["backend"]).lower()
    clip = np.ascontiguousarray(audio, dtype=np.float32)

    if backend in MLX_ALIASES:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            clip, path_or_hf_repo=str(a["mlx_model"]), language=str(a["language"]),
            task="transcribe", word_timestamps=False, verbose=None,
            condition_on_previous_text=False)
        return " ".join(str(result.get("text", "")).split())

    if backend in FASTER_ALIASES:
        from faster_whisper import WhisperModel
        key = f"{a['faster_whisper_model']}|{a.get('faster_whisper_compute', 'int8')}"
        model = _FASTER_CACHE.get(key)
        if model is None:
            model = WhisperModel(str(a["faster_whisper_model"]), device="cpu",
                                 compute_type=str(a.get("faster_whisper_compute", "int8")))
            _FASTER_CACHE[key] = model
        pieces, _ = model.transcribe(
            clip, language=str(a["language"]), task="transcribe",
            beam_size=int(a.get("beam_size", 5)), word_timestamps=False,
            vad_filter=False, condition_on_previous_text=False)
        return " ".join(" ".join(p.text.split()) for p in pieces).strip()

    raise ValueError(f"asr.backend khong hop le: {backend}")


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def refine_clips(cfg: dict, clips: list[dict], vad_sample_rate: int, progress=None) -> dict[int, str]:
    import torch
    from transformers import pipeline

    rcfg = cfg["asr"]["refine"]
    device = _resolve_device(str(rcfg.get("device", "auto")))
    recogniser = pipeline(
        "automatic-speech-recognition",
        model=str(rcfg["model"]),
        device=device,
        torch_dtype=torch.float32,
    )
    generate_kwargs = {"language": str(cfg["asr"]["language"]), "task": "transcribe"}
    batch_size = max(1, int(rcfg.get("batch_size", 4)))

    texts: dict[int, str] = {}
    for offset in range(0, len(clips), batch_size):
        batch = clips[offset:offset + batch_size]
        payload = []
        for clip in batch:
            x, sr = audio_io.read_all(clip["wav"])
            payload.append({"raw": audio_io.resample(x, sr, vad_sample_rate),
                            "sampling_rate": int(vad_sample_rate)})
        outputs = recogniser(payload, generate_kwargs=generate_kwargs, batch_size=len(payload))
        if isinstance(outputs, dict):
            outputs = [outputs]
        for clip, out in zip(batch, outputs):
            texts[int(clip["index"])] = str(out.get("text", "")).strip()
        if progress is not None:
            progress(min(offset + batch_size, len(clips)), len(clips))
    return texts

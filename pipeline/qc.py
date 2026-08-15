from __future__ import annotations

from collections import Counter


def check(item: dict, qcfg: dict) -> list[str]:
    reasons: list[str] = []
    duration = float(item.get("duration", 0.0))
    text = str(item.get("text", ""))
    chars = len(text)

    if duration < float(qcfg["min_duration"]):
        reasons.append("duration_too_short")
    if duration > float(qcfg["max_duration"]):
        reasons.append("duration_too_long")
    if chars < int(qcfg["min_chars"]):
        reasons.append("text_too_short")
    if duration > 0:
        rate = chars / duration
        if rate < float(qcfg["min_chars_per_sec"]):
            reasons.append("speech_rate_low")
        if rate > float(qcfg["max_chars_per_sec"]):
            reasons.append("speech_rate_high")
    if float(item.get("clip_ratio", 0.0)) > float(qcfg["max_clip_ratio"]):
        reasons.append("clipping")
    if float(item.get("snr_db", 0.0)) < float(qcfg["min_snr_db"]):
        reasons.append("low_snr")
    if float(item.get("asr_prob", 1.0)) < float(qcfg["min_asr_prob"]):
        reasons.append("low_asr_confidence")
    floor = qcfg.get("min_word_prob")
    if floor is not None and float(item.get("min_word_prob", 1.0)) < float(floor):
        reasons.append("low_word_confidence")
    ctc_floor = qcfg.get("min_ctc_score")
    if ctc_floor is not None and float(item.get("ctc_score", 1.0)) < float(ctc_floor):
        reasons.append("text_audio_mismatch")
    full_floor = qcfg.get("min_completeness")
    if full_floor is not None and float(item.get("completeness", 1.0)) < float(full_floor):
        reasons.append("text_incomplete")
    if float(item.get("no_speech_prob", 0.0)) > float(qcfg["max_no_speech_prob"]):
        override = qcfg.get("ctc_overrides_no_speech")
        if override is None or float(item.get("ctc_score", 0.0)) < float(override):
            reasons.append("no_speech")
    if not text:
        reasons.append("empty_text")
    return reasons


def run(items: list[dict], qcfg: dict) -> tuple[list[dict], list[dict], dict]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    drop_duplicates = bool(qcfg.get("drop_duplicate_text", True))

    for item in items:
        reasons = check(item, qcfg)
        key = str(item.get("text", "")).lower()
        if drop_duplicates and key and key in seen:
            reasons.append("duplicate_text")
        if reasons:
            rejected.append({**item, "reasons": reasons})
            continue
        seen.add(key)
        accepted.append(item)

    counter: Counter[str] = Counter()
    for item in rejected:
        counter.update(item["reasons"])
    stats = {
        "total": len(items),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "reasons": dict(counter.most_common()),
    }
    return accepted, rejected, stats

from __future__ import annotations

import csv
import json
import random
import shutil
import statistics
from pathlib import Path


def _percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    def at(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
        return round(ordered[idx], 3)
    return {"min": at(0.0), "p10": at(0.10), "median": at(0.50), "p90": at(0.90), "max": at(1.0)}


def _merge_qc(parts: list[dict]) -> dict:
    reasons: dict[str, int] = {}
    total = accepted = rejected = 0
    for part in parts:
        stats = part.get("stats") or {}
        total += int(stats.get("total", 0))
        accepted += int(stats.get("accepted", 0))
        rejected += int(stats.get("rejected", 0))
        for reason, count in (stats.get("reasons") or {}).items():
            reasons[reason] = reasons.get(reason, 0) + int(count)
    return {
        "total": total, "accepted": accepted, "rejected": rejected,
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def run(cfg: dict, root: Path, parts: list[dict]) -> dict:
    accepted: list[dict] = []
    rejected: list[dict] = []
    for part in parts:
        for item in part["accepted"]:
            accepted.append({**item, "source_id": part["id"]})
        for item in part["rejected"]:
            rejected.append({**item, "source_id": part["id"]})
    qc_stats = _merge_qc(parts)
    timings = {f"{part['id']}.{k}": v
               for part in parts for k, v in (part.get("timings") or {}).items()}

    project = cfg["project"]
    ecfg = cfg["export"]
    name = str(project["name"])
    speaker = str(project["speaker"])
    language = str(project["language"])

    out_root = (root / str(project["out_dir"])).resolve()
    data_dir = out_root / "Data" / name
    raw_dir = data_dir / "raw" / speaker
    wavs_dir = data_dir / "wavs"
    lists_dir = data_dir / "filelists"
    for directory in (raw_dir, lists_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)
    if bool(ecfg.get("emit_wavs_dir", True)):
        shutil.rmtree(wavs_dir, ignore_errors=True)
        wavs_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for item in accepted:
        filename = Path(item["wav"]).name
        shutil.copyfile(item["wav"], raw_dir / filename)
        if bool(ecfg.get("emit_wavs_dir", True)):
            shutil.copyfile(item["wav"], wavs_dir / filename)
        entries.append({
            "filename": filename,
            "source_id": item["source_id"],
            "text": item["text"],
            "duration": float(item["duration"]),
            "list_path": f"Data/{name}/wavs/{filename}",
        })

    rng = random.Random(int(project.get("seed", 1234)))
    shuffled = entries[:]
    rng.shuffle(shuffled)
    val_size = min(int(ecfg.get("val_max", 100)),
                   max(1, int(len(shuffled) * float(ecfg.get("val_ratio", 0.02)))))
    val_size = min(val_size, max(0, len(shuffled) - 1))
    val_entries = shuffled[:val_size]
    train_entries = shuffled[val_size:]

    def line(entry: dict) -> str:
        return f"{entry['list_path']}|{speaker}|{language}|{entry['text']}"

    (lists_dir / f"{speaker}.list").write_text(
        "\n".join(line(e) for e in entries) + "\n", encoding="utf-8")
    (lists_dir / f"{speaker}.list.train").write_text(
        "\n".join(line(e) for e in train_entries) + "\n", encoding="utf-8")
    (lists_dir / f"{speaker}.list.val").write_text(
        "\n".join(line(e) for e in val_entries) + "\n", encoding="utf-8")

    if bool(ecfg.get("emit_ljspeech_csv", True)):
        with (out_root / "metadata.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
            for entry in entries:
                writer.writerow([Path(entry["filename"]).stem, entry["text"], entry["text"]])

    durations = [e["duration"] for e in entries]
    char_counts = [len(e["text"]) for e in entries]
    summary = {
        "project": name,
        "speaker": speaker,
        "language": language,
        "sources": [{
            "id": part["id"],
            "path": (part["source"] or {}).get("path"),
            "codec": (part["source"] or {}).get("codec"),
            "bit_rate": (part["source"] or {}).get("bit_rate"),
            "duration": (part["source"] or {}).get("duration"),
            "decoded_duration": (part["source"] or {}).get("decoded_duration"),
            "clips": sum(1 for e in entries if e["source_id"] == part["id"]),
            "seconds": round(sum(e["duration"] for e in entries
                                 if e["source_id"] == part["id"]), 2),
        } for part in parts],
        "clips": len(entries),
        "total_seconds": round(sum(durations), 2),
        "total_hours": round(sum(durations) / 3600.0, 3),
        "mean_duration": round(statistics.fmean(durations), 3) if durations else 0.0,
        "duration_percentiles": _percentiles(durations),
        "chars_percentiles": _percentiles([float(c) for c in char_counts]),
        "train": len(train_entries),
        "val": len(val_entries),
        "qc": qc_stats,
        "timings_seconds": timings,
        "config": {
            "sample_rate": cfg["audio"]["sample_rate"],
            "denoise_backend": cfg["denoise"]["backend"],
            "asr_backend": cfg["asr"]["backend"],
            "asr_refine": bool(cfg["asr"]["refine"]["enabled"]),
        },
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_root / "rejected.jsonl").open("w", encoding="utf-8") as fh:
        for item in rejected:
            fh.write(json.dumps({
                "source_id": item.get("source_id"),
                "index": item.get("index"),
                "start": item.get("start"),
                "end": item.get("end"),
                "duration": item.get("duration"),
                "reasons": item.get("reasons"),
                "text": item.get("text"),
            }, ensure_ascii=False) + "\n")

    _write_report(out_root / "report.md", summary, qc_stats)
    return summary


def _write_report(path: Path, summary: dict, qc_stats: dict) -> None:
    lines = [
        f"# Dataset {summary['project']}",
        "",
        f"## Nguồn ({len(summary['sources'])} file)",
        "",
        "| id | file | kbps | gốc (phút) | đã xử lý (phút) | clip | dùng được (phút) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for src in summary["sources"]:
        lines.append(
            f"| `{src['id']}` | {Path(str(src.get('path') or '')).name} "
            f"| {round((src.get('bit_rate') or 0) / 1000)} "
            f"| {round((src.get('duration') or 0) / 60, 2)} "
            f"| {round((src.get('decoded_duration') or 0) / 60, 2)} "
            f"| {src.get('clips')} | {round((src.get('seconds') or 0) / 60, 2)} |")
    lines += [
        "",
        "## Kết quả",
        f"- Clip hợp lệ: **{summary['clips']}**",
        f"- Tổng thời lượng: **{summary['total_hours']} giờ** ({summary['total_seconds']} s)",
        f"- Thời lượng trung bình: {summary['mean_duration']} s",
        f"- Phân bố thời lượng: {summary['duration_percentiles']}",
        f"- Phân bố số ký tự: {summary['chars_percentiles']}",
        f"- Train / Val: {summary['train']} / {summary['val']}",
        "",
        "## Lọc chất lượng",
        f"- Tổng segment: {qc_stats.get('total')}",
        f"- Nhận: {qc_stats.get('accepted')} | Loại: {qc_stats.get('rejected')}",
        "",
        "| Lý do loại | Số lượng |",
        "| --- | --- |",
    ]
    for reason, count in (qc_stats.get("reasons") or {}).items():
        lines.append(f"| {reason} | {count} |")
    lines += [
        "",
        "## Thời gian xử lý (giây)",
        "",
        "| Bước | Giây |",
        "| --- | --- |",
    ]
    for stage, seconds in (summary.get("timings_seconds") or {}).items():
        lines.append(f"| {stage} | {round(float(seconds), 1)} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import align
import asr
import audio_io
import denoise
import edits
import export
import qc
import segment
import vi_norm

STAGES = ("decode", "vad", "denoise", "asr", "align", "segment",
          "cut", "text", "qc", "export")


class Work:
    def __init__(self, root: Path, cfg: dict, source_id: str):
        self.source_id = source_id
        self.dir = (root / str(cfg["project"]["work_dir"]) / source_id).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clips = self.dir / "clips"

    def path(self, name: str) -> Path:
        return self.dir / name

    def read_json(self, name: str):
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write_json(self, name: str, payload) -> None:
        self.path(name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def has(self, *names: str) -> bool:
        return all(self.path(n).exists() for n in names)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_root(cfg: dict, config_path: Path) -> Path:
    configured = cfg["project"].get("root")
    if configured:
        return Path(configured).expanduser().resolve()
    return config_path.resolve().parent.parent


def sanitise_id(text: str) -> str:
    raw = "".join(c if c.isalnum() else "_" for c in str(text).lower())
    return "_".join(part for part in raw.split("_") if part) or "source"


def resolve_sources(cfg: dict, root: Path, override: str | None = None) -> list[dict]:
    icfg = cfg["input"]
    candidates: list[Path] = []

    if override:
        candidates = [Path(override)]
    elif icfg.get("paths"):
        candidates = [Path(str(p)) for p in icfg["paths"]]
    elif icfg.get("dir"):
        base = (root / str(icfg["dir"])).resolve()
        patterns = icfg.get("pattern") or ["*.mp3"]
        if isinstance(patterns, str):
            patterns = [patterns]
        seen: set[Path] = set()
        for pattern in patterns:
            for found in base.glob(str(pattern)):
                if found.is_file():
                    seen.add(found.resolve())
        candidates = sorted(seen)
    elif icfg.get("path"):
        candidates = [Path(str(icfg["path"]))]

    sources: list[dict] = []
    used: set[str] = set()
    for candidate in candidates:
        full = candidate if candidate.is_absolute() else (root / candidate)
        full = full.resolve()
        if not full.exists():
            raise FileNotFoundError(f"khong tim thay input: {full}")
        source_id = sanitise_id(full.stem)
        base_id, counter = source_id, 2
        while source_id in used:
            source_id = f"{base_id}_{counter}"
            counter += 1
        used.add(source_id)
        sources.append({"id": source_id, "path": full})
    return sources


def stage_decode(cfg: dict, source: dict, work: Work) -> dict:
    src = source["path"]
    summary = audio_io.source_summary(src)
    summary["source_id"] = source["id"]
    preview = float(cfg["input"].get("preview_minutes", 0) or 0) * 60.0
    sr = int(cfg["audio"]["sample_rate"])
    vad_sr = int(cfg["audio"]["vad_sample_rate"])

    log(f"decode: {src.name} -> mono {sr} Hz" + (f" (preview {preview / 60:.1f} phut)" if preview else ""))
    audio_io.decode(src, work.path("raw_44k.wav"), sr,
                    highpass_hz=float(cfg["audio"].get("highpass_hz", 0)),
                    preview_seconds=preview)
    x, got_sr = audio_io.read_all(work.path("raw_44k.wav"))
    audio_io.write_wav(work.path("raw_16k.wav"), audio_io.resample(x, got_sr, vad_sr),
                       vad_sr, subtype="FLOAT")
    summary["decoded_duration"] = round(x.size / got_sr, 3)
    work.write_json("source.json", summary)
    return summary


def stage_vad(cfg: dict, work: Work, target: str) -> list[tuple[float, float]]:
    audio_file = work.path("raw_16k.wav" if target == "raw" else "clean_16k.wav")
    x, sr = audio_io.read_all(audio_file)
    regions = segment.speech_regions(x, sr, cfg["vad"])
    total = x.size / sr
    speech = sum(e - s for s, e in regions)
    log(f"vad[{target}]: {len(regions)} vung noi, {speech / 60:.1f}/{total / 60:.1f} phut la giong noi")
    work.write_json(f"vad_{target}.json", {"total": total, "regions": regions})
    return regions


def stage_denoise(cfg: dict, work: Work) -> None:
    backend = str(cfg["denoise"]["backend"]).lower()
    noise_regions: list[tuple[float, float]] = []
    if denoise.needs_noise_profile(backend):
        raw = work.read_json("vad_raw.json")
        noise_regions = segment.silence_regions(
            [tuple(r) for r in raw["regions"]], float(raw["total"]), min_length=0.35)
        log(f"denoise: {len(noise_regions)} doan im lang de hoc noise profile")

    log(f"denoise: backend={backend}")
    denoise.apply(cfg, work.path("raw_44k.wav"), work.path("clean_44k.wav"), noise_regions)

    x, sr = audio_io.read_all(work.path("clean_44k.wav"))
    x = audio_io.loudness_normalize(x, sr, float(cfg["audio"]["lufs_target"]),
                                    float(cfg["audio"]["peak_ceiling_db"]))
    audio_io.write_wav(work.path("clean_44k.wav"), x, sr, subtype="FLOAT")
    vad_sr = int(cfg["audio"]["vad_sample_rate"])
    audio_io.write_wav(work.path("clean_16k.wav"), audio_io.resample(x, sr, vad_sr),
                       vad_sr, subtype="FLOAT")
    log(f"denoise: xong, muc to {cfg['audio']['lufs_target']} LUFS")


def stage_asr(cfg: dict, work: Work) -> list[dict]:
    backend = cfg["asr"]["backend"]
    log(f"asr: backend={backend} (lan dau se tai model, mat vai phut)")
    segments = asr.transcribe(cfg, work.path("clean_16k.wav"))
    words = sum(len(s.get("words") or []) for s in segments)
    log(f"asr: {len(segments)} doan, {words} tu co timestamp")
    work.write_json("asr.json", segments)
    return segments


def _load_silences(cfg: dict, work: Work) -> tuple[np.ndarray, int, list, float, list, float]:
    vad = work.read_json("vad_clean.json")
    regions = [tuple(r) for r in vad["regions"]]
    x, sr = audio_io.read_all(work.path("clean_16k.wav"))
    silences, threshold = segment.silence_intervals(x, sr, regions, cfg["segment"])
    return x, sr, regions, float(vad["total"]), silences, threshold


def stage_align(cfg: dict, work: Work) -> list[dict]:
    asr_segments = work.read_json("asr.json")
    x, sr, _, total, silences, _ = _load_silences(cfg, work)
    words = segment.flatten_words(asr_segments)

    before = align.silence_purity(words, silences)
    log(f"align: truoc khi can chinh, {100 * before:.0f}% cho nghi khong bi span cua tu phu len")
    log(f"align: model={cfg['align']['model']} (lan dau se tai ~400 MB)")

    def progress(done: int, total_chunks: int) -> None:
        if done % max(1, total_chunks // 10) == 0:
            log(f"align: chunk {done}/{total_chunks}")

    updated, stats = align.run(cfg, x, sr, asr_segments, words, silences, total, progress)
    after = align.silence_purity(segment.flatten_words(updated), silences)

    log(f"align: {stats['realigned']}/{stats['words']} tu duoc can chinh lai "
        f"tren {stats['chunks']} chunk (device={stats['device']})")
    if stats["failed_chunks"]:
        log(f"align: {stats['failed_chunks']} chunk that bai, giu timestamp cu")
    log(f"align: sau khi can chinh, {100 * after:.0f}% cho nghi khong bi phu "
        f"(truoc {100 * before:.0f}%)")
    if after <= before:
        log("align: CANH BAO khong cai thien - kiem tra lai model hoac ngon ngu")

    work.write_json("aligned.json", updated)
    work.write_json("align_stats.json", {**stats, "hit_rate_before": round(before, 4),
                                         "hit_rate_after": round(after, 4)})
    return updated


def stage_segment(cfg: dict, work: Work) -> list[dict]:
    source = "aligned.json" if work.has("aligned.json") else "asr.json"
    log(f"segment: dung timestamp tu {source}")
    asr_segments = work.read_json(source)
    x, sr, regions, total, silences, threshold = _load_silences(cfg, work)
    log(f"segment: {len(silences)} khoang lang do tu audio (nguong {threshold} dBFS)")

    segments, stats = segment.build(asr_segments, silences, total, cfg["segment"])
    skipped = stats["silences"] - stats["cuts"]
    log(f"segment: {stats['cuts']} diem cat hop le, bo {skipped} khoang lang "
        f"bi timestamp cua tu trum qua")
    kept = sum(s["duration"] for s in segments)
    log(f"segment: {len(segments)} clip, tong {kept / 60:.1f} phut")
    if stats["oversize"]:
        log(f"segment: CANH BAO {stats['oversize']} clip vuot max_duration - "
            f"doan noi lien khong co cho nghi de cat, QC se loai")

    violations = segment.boundary_report(x, sr, segments, threshold)
    if violations:
        log(f"segment: CANH BAO {len(violations)} bien cat nam trong vung co tieng noi")
        for index, label, level in violations[:10]:
            log(f"segment:   #{index} {label} = {level} dBFS")
    else:
        log("segment: kiem tra OK - moi bien cat deu nam trong khoang lang")

    work.write_json("segments.json", segments)
    work.write_json("segment_stats.json",
                    {**stats, "threshold_db": threshold, "boundary_violations": len(violations)})
    return segments


def stage_cut(cfg: dict, work: Work) -> list[dict]:
    segments = work.read_json("segments.json")
    scfg = cfg["segment"]
    acfg = cfg["audio"]
    name = work.source_id
    work.clips.mkdir(parents=True, exist_ok=True)
    for old in work.clips.glob("*.wav"):
        old.unlink()

    clips = []
    source = work.path("clean_44k.wav")
    for item in segments:
        x, sr = audio_io.read_slice(source, item["start"], item["end"])
        if x.size == 0:
            continue
        x = audio_io.trim_silence(x, sr, float(scfg["trim_db"]), float(scfg["pad_ms"]))
        if bool(acfg.get("per_clip_loudnorm", False)):
            x = audio_io.loudness_normalize(x, sr, float(acfg["lufs_target"]),
                                            float(acfg["peak_ceiling_db"]))
        else:
            x = audio_io.peak_limit(x, float(acfg["peak_ceiling_db"]))
        wav_path = work.clips / f"{name}_{item['index']:05d}.wav"
        audio_io.write_wav(wav_path, x, sr, subtype="PCM_16")
        clips.append({
            **item,
            "wav": str(wav_path),
            "duration": round(x.size / sr, 3),
            "snr_db": round(audio_io.snr_db(x, sr), 2),
            "clip_ratio": round(audio_io.clip_ratio(x), 6),
            "dbfs": round(audio_io.dbfs(x), 2),
        })
    log(f"cut: da xuat {len(clips)} file wav vao {work.clips}")
    work.write_json("clips.json", clips)
    return clips


def _rescore_texts(cfg: dict, clips: list[dict], normalizer) -> dict:
    rcfg = cfg["rescore"]
    score_floor = float(rcfg.get("threshold", 0.8))
    completeness_floor = float(rcfg.get("min_completeness", 0.85))
    min_gain = float(rcfg.get("min_gain", 0.05))
    tolerance = float(rcfg.get("score_tolerance", 0.05))
    completeness_gain = float(rcfg.get("completeness_gain", 0.05))
    vad_sr = int(cfg["audio"]["vad_sample_rate"])

    aligner = align.Aligner(str(cfg["align"]["model"]),
                            align.resolve_device(str(cfg["align"].get("device", "auto"))))
    stats = {"clips": len(clips), "candidates": 0, "replaced": 0,
             "by_score": 0, "by_completeness": 0, "failed": 0}

    for position, clip in enumerate(clips, 1):
        raw, sr = audio_io.read_all(clip["wav"])
        clip16 = audio_io.resample(raw, sr, vad_sr)
        emission = aligner.emission(clip16)
        decoded = aligner.decode_from(emission)

        old_text = normalizer(str(clip.get("raw_text", "")))
        old_score = aligner.score_from(emission, clip16.size, vad_sr, old_text) or 0.0
        old_full = aligner.completeness(old_text, decoded)
        clip["ctc_score"] = round(old_score, 4)
        clip["completeness"] = round(old_full, 4)
        clip["text_source"] = "first_pass"

        if old_score >= score_floor and old_full >= completeness_floor:
            continue
        stats["candidates"] += 1

        try:
            new_raw = asr.transcribe_array(cfg, clip16, vad_sr)
        except Exception as exc:
            log(f"text: clip #{clip['index']} doc lai loi: {type(exc).__name__}")
            stats["failed"] += 1
            continue
        if not new_raw:
            stats["failed"] += 1
            continue

        new_text = normalizer(new_raw)
        new_score = aligner.score_from(emission, clip16.size, vad_sr, new_text) or 0.0
        new_full = aligner.completeness(new_text, decoded)
        clip["ctc_score_first_pass"] = round(old_score, 4)
        clip["ctc_score_per_clip"] = round(new_score, 4)
        clip["completeness_per_clip"] = round(new_full, 4)

        better_score = new_score > old_score + min_gain
        closer = abs(1.0 - new_full) + completeness_gain < abs(1.0 - old_full)
        better_full = closer and new_score > old_score - tolerance
        if better_score or better_full:
            clip["rescored_text"] = new_raw
            clip["ctc_score"] = round(new_score, 4)
            clip["completeness"] = round(new_full, 4)
            clip["text_source"] = "per_clip"
            stats["replaced"] += 1
            stats["by_score" if better_score else "by_completeness"] += 1
        if position % max(1, len(clips) // 5) == 0:
            log(f"text: cham diem {position}/{len(clips)}")

    log(f"text: {stats['candidates']}/{len(clips)} clip duoi nguong "
        f"(ctc < {score_floor} hoac day du < {completeness_floor})")
    log(f"text: thay text cho {stats['replaced']} clip "
        f"({stats['by_score']} vi diem khop, {stats['by_completeness']} vi day du hon)")
    if stats["failed"]:
        log(f"text: {stats['failed']} clip doc lai that bai, giu text cu")
    return stats


def stage_text(cfg: dict, work: Work) -> list[dict]:
    clips = work.read_json("clips.json")
    normalizer = vi_norm.build(cfg["text"])
    rcfg = cfg["asr"]["refine"]

    stats = {"clips": len(clips), "candidates": 0, "replaced": 0}
    if bool(cfg["rescore"].get("enabled", True)) and work.has("aligned.json"):
        stats = _rescore_texts(cfg, clips, normalizer)

    if bool(rcfg.get("enabled", False)):
        log(f"text: refine bang {rcfg['model']}")

        def progress(done: int, total: int) -> None:
            if done % max(1, total // 20) < int(rcfg.get("batch_size", 4)):
                log(f"text: refine {done}/{total}")

        refined = asr.refine_clips(cfg, clips, int(cfg["audio"]["vad_sample_rate"]), progress)
        for clip in clips:
            better = refined.get(int(clip["index"]), "").strip()
            if better:
                clip["refined_text"] = better

    for clip in clips:
        source_text = clip.get("refined_text") or clip.get("rescored_text") or clip["raw_text"]
        clip["text"] = normalizer(source_text)
    log(f"text: chuan hoa {len(clips)} cau")
    work.write_json("texts.json", clips)
    work.write_json("rescore_stats.json", stats)
    return clips


def stage_qc(cfg: dict, work: Work) -> tuple[list[dict], list[dict], dict]:
    items = work.read_json("texts.json")
    ecfg = cfg.get("edits") or {}
    if bool(ecfg.get("enabled", True)):
        edit_file = work.path(str(ecfg.get("file", "edit.txt")))
        if not edit_file.exists():
            edits.write_template(edit_file, items)
            log(f"qc: tao file sua tay {edit_file}")
        else:
            applied = edits.apply(edits.read(edit_file), items,
                                  float(ecfg.get("duration_tolerance", 0.05)))
            log(f"qc: sua tay {applied['applied']} clip tu {edit_file.name} "
                f"({applied['entries']} muc, {applied['unchanged']} khong doi)")
            for name in applied["unmatched"][:5]:
                log(f"qc:   khong khop clip nao: {name}")
            for name in applied["mismatched"][:5]:
                log(f"qc:   thoi luong lech, bo qua: {name}")

    accepted, rejected, stats = qc.run(items, cfg["qc"])
    log(f"qc: nhan {stats['accepted']}/{stats['total']}, loai {stats['rejected']}")
    for reason, count in (stats["reasons"] or {}).items():
        log(f"qc:   {reason} = {count}")
    work.write_json("qc.json", {"accepted": accepted, "rejected": rejected, "stats": stats})
    return accepted, rejected, stats


def stage_export(cfg: dict, root: Path, parts: list[dict]) -> dict:
    summary = export.run(cfg, root, parts)
    out = (root / str(cfg["project"]["out_dir"])).resolve()
    log(f"export: gop {len(parts)} nguon -> {summary['clips']} clip, "
        f"{summary['total_hours']} gio -> {out}")
    return summary


def process_source(cfg: dict, work: Work, source: dict, wanted: set, force: bool) -> dict:
    timings: dict[str, float] = {}

    def should_run(stage: str, artifacts: tuple = (), needs: tuple = ()) -> bool:
        if stage not in wanted:
            return False
        if not force and artifacts and work.has(*artifacts):
            return False
        missing = [n for n in needs if not work.path(n).exists()]
        if missing:
            log(f"{work.source_id}: bo qua {stage} - thieu {', '.join(missing)} "
                f"(chay stage truoc do)")
            return False
        return True

    def timed(stage: str, fn):
        started = time.time()
        result = fn()
        timings[stage] = round(time.time() - started, 2)
        return result

    if should_run("decode", ("raw_44k.wav", "raw_16k.wav", "source.json")):
        timed("decode", lambda: stage_decode(cfg, source, work))

    if (denoise.needs_noise_profile(str(cfg["denoise"]["backend"]))
            and should_run("vad", ("vad_raw.json",), ("raw_16k.wav",))):
        timed("vad_raw", lambda: stage_vad(cfg, work, "raw"))

    if should_run("denoise", ("clean_44k.wav", "clean_16k.wav"), ("raw_44k.wav",)):
        timed("denoise", lambda: stage_denoise(cfg, work))

    if should_run("vad", ("vad_clean.json",), ("clean_16k.wav",)):
        timed("vad_clean", lambda: stage_vad(cfg, work, "clean"))

    if should_run("asr", ("asr.json",), ("clean_16k.wav",)):
        timed("asr", lambda: stage_asr(cfg, work))

    if bool(cfg["align"]["enabled"]) and should_run(
            "align", ("aligned.json",), ("asr.json", "clean_16k.wav", "vad_clean.json")):
        timed("align", lambda: stage_align(cfg, work))

    if should_run("segment", ("segments.json",), ("asr.json", "clean_16k.wav", "vad_clean.json")):
        timed("segment", lambda: stage_segment(cfg, work))

    if should_run("cut", ("clips.json",), ("segments.json", "clean_44k.wav")):
        timed("cut", lambda: stage_cut(cfg, work))

    if should_run("text", ("texts.json",), ("clips.json",)):
        timed("text", lambda: stage_text(cfg, work))

    if should_run("qc", ("qc.json",), ("texts.json",)):
        timed("qc", lambda: stage_qc(cfg, work))

    previous = work.read_json("timings.json") if work.has("timings.json") else {}
    previous.update(timings)
    work.write_json("timings.json", previous)
    return previous


def apply_overrides(cfg: dict, args: argparse.Namespace) -> None:
    if args.preview_minutes is not None:
        cfg["input"]["preview_minutes"] = args.preview_minutes
    if args.denoise:
        cfg["denoise"]["backend"] = args.denoise
    if args.asr:
        cfg["asr"]["backend"] = args.asr
    if args.refine:
        cfg["asr"]["refine"]["enabled"] = True
    if args.speaker:
        cfg["project"]["speaker"] = args.speaker


def main() -> int:
    parser = argparse.ArgumentParser(prog="run.py")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--stages", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preview-minutes", type=float, default=None)
    parser.add_argument("--denoise", default=None, choices=list(denoise.BACKENDS))
    parser.add_argument("--asr", default=None)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--speaker", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    apply_overrides(cfg, args)
    root = resolve_root(cfg, config_path)

    wanted = set(STAGES) if args.stages == "all" else {s.strip() for s in args.stages.split(",")}
    unknown = wanted - set(STAGES)
    if unknown:
        parser.error(f"stage khong hop le: {sorted(unknown)} (co: {list(STAGES)})")

    sources = resolve_sources(cfg, root, args.input)
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        sources = [s for s in sources if s["id"] in keep]
    if not sources:
        parser.error("khong tim thay file audio nao trong input")

    log(f"root={root}")
    log(f"nguon ({len(sources)}): {', '.join(s['id'] for s in sources)}")

    parts: list[dict] = []
    for position, source in enumerate(sources, 1):
        work = Work(root, cfg, source["id"])
        log(f"=== [{position}/{len(sources)}] {source['id']} <- {source['path'].name} ===")
        timings = process_source(cfg, work, source, wanted, args.force)
        if not work.has("qc.json", "source.json"):
            log(f"{source['id']}: chua co ket qua, bo qua khi gop")
            continue
        payload = work.read_json("qc.json")
        parts.append({
            "id": source["id"],
            "source": work.read_json("source.json"),
            "accepted": payload["accepted"],
            "rejected": payload["rejected"],
            "stats": payload["stats"],
            "timings": timings,
        })

    if "export" in wanted and parts:
        stage_export(cfg, root, parts)

    log("hoan tat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

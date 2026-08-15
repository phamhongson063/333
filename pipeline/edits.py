from __future__ import annotations

import re
from pathlib import Path

HEADER = re.compile(r"^#\s*(?P<name>\S+\.wav)\s+(?P<duration>[0-9]+(?:\.[0-9]+)?)\s*s")

INSTRUCTIONS = (
    "# Sua text o dong ngay ben duoi moi dong bat dau bang '#'.",
    "# Khong sua cac dong '#' - chung chi de tra cuu.",
    "# File nay khong bao gio bi pipeline ghi de. Xoa file de bo het sua tay.",
    "# Sau khi sua, chay: run.py --stages qc,export --force",
    "# Text sua tay duoc dung nguyen van: tu viet so thanh chu, chi dung dau , . ! ? …",
    "#",
)


def write_template(path: Path, clips: list[dict]) -> int:
    lines = list(INSTRUCTIONS)
    for clip in clips:
        name = Path(str(clip["wav"])).name
        lines.append("")
        lines.append(f"# {name}  {float(clip['duration']):.2f}s  "
                     f"ctc={float(clip.get('ctc_score', 1.0)):.2f}  "
                     f"daydu={float(clip.get('completeness', 1.0)):.2f}")
        lines.append(str(clip.get("text", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(clips)


def read(path: Path) -> dict[str, tuple[float | None, str]]:
    records: dict[str, tuple[float | None, str]] = {}
    name: str | None = None
    duration: float | None = None
    buffer: list[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        found = HEADER.match(line)
        if found:
            if name is not None:
                records[name] = (duration, " ".join(buffer).strip())
            name = found.group("name")
            duration = float(found.group("duration"))
            buffer = []
            continue
        if line.startswith("#"):
            continue
        if line:
            buffer.append(line)
    if name is not None:
        records[name] = (duration, " ".join(buffer).strip())
    return records


def apply(records: dict[str, tuple[float | None, str]], clips: list[dict],
          tolerance: float = 0.05) -> dict:
    by_name = {Path(str(clip["wav"])).name: clip for clip in clips}
    stats = {"entries": len(records), "applied": 0, "unchanged": 0,
             "unmatched": [], "mismatched": []}

    for name, (duration, text) in records.items():
        clip = by_name.get(name)
        if clip is None:
            stats["unmatched"].append(name)
            continue
        if duration is not None and abs(float(clip["duration"]) - duration) > tolerance:
            stats["mismatched"].append(name)
            continue
        if not text or text == str(clip.get("text", "")):
            stats["unchanged"] += 1
            continue
        clip["text"] = text
        clip["text_source"] = "manual"
        clip["ctc_score"] = 1.0
        clip["completeness"] = 1.0
        stats["applied"] += 1
    return stats

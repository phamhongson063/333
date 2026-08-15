from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METADATA_CSV = ROOT / "dataset" / "metadata.csv"
WAVS_DIR = ROOT / "dataset" / "Data" / "vi_story" / "wavs"
OUT_DIR = Path(__file__).resolve().parent / "F5-TTS-Vietnamese" / "data" / "your_training_dataset"
OUT_WAVS = OUT_DIR / "wavs"
TARGET_SR = 24000


def ffmpeg_resample(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(src), "-ac", "1", "-ar", str(TARGET_SR), str(dst)],
        check=True,
    )


def main() -> None:
    OUT_WAVS.mkdir(parents=True, exist_ok=True)
    tokens: set[str] = set()
    lines: list[str] = []

    with METADATA_CSV.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        rows = [row for row in reader if row]

    for row in rows:
        stem, text = row[0], row[1].strip().lower()
        src = WAVS_DIR / f"{stem}.wav"
        if not src.exists():
            print(f"bo qua {stem}: khong tim thay {src}")
            continue
        dst = OUT_WAVS / f"{stem}.wav"
        ffmpeg_resample(src, dst)
        lines.append(f"wavs/{stem}.wav|{text}")
        tokens.update(text)

    (OUT_DIR / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "vocab_your_dataset.txt").write_text(
        "\n".join(sorted(tokens)), encoding="utf-8")

    print(f"da xuat {len(lines)} clip vao {OUT_DIR}")


if __name__ == "__main__":
    main()

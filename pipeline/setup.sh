#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PY="${PYTHON:-python3}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "THIEU ffmpeg -> brew install ffmpeg" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip wheel setuptools
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"

if [ "${WITH_EXTRA:-0}" = "1" ]; then
  "$VENV/bin/python" -m pip install -r "$ROOT/requirements-extra.txt"
fi

"$VENV/bin/python" - <<'EOF'
import importlib
required = ["numpy", "scipy", "soundfile", "soxr", "pyloudnorm", "noisereduce",
            "yaml", "torch", "silero_vad", "onnxruntime"]
optional = ["mlx_whisper", "faster_whisper", "transformers", "demucs"]
for name in required:
    importlib.import_module(name)
    print(f"ok       {name}")
for name in optional:
    try:
        importlib.import_module(name)
        print(f"ok       {name}")
    except Exception as exc:
        print(f"thieu    {name}: {type(exc).__name__}")
EOF

echo
echo "Xong. Chay thu 5 phut dau:"
echo "  $VENV/bin/python $ROOT/run.py --preview-minutes 5"

from __future__ import annotations

import argparse
import json
import sys

import soundfile as sf
from importlib.resources import files
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    infer_process,
    load_model,
    load_vocoder,
    nfe_step,
    preprocess_ref_audio_text,
    sway_sampling_coef,
    target_rms,
)
from f5_tts.model import DiT, UNetT  # noqa: F401

VOCODER = "vocos"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_file", required=True)
    parser.add_argument("--vocab_file", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--exp_name", default="F5TTS_Base")
    args = parser.parse_args()

    channel = sys.stdout
    sys.stdout = sys.stderr

    def reply(payload: dict) -> None:
        channel.write(json.dumps(payload) + "\n")
        channel.flush()

    try:
        vocoder = load_vocoder(vocoder_name=VOCODER)
        model_cfg = OmegaConf.load(
            str(files("f5_tts").joinpath(f"configs/{args.exp_name}.yaml"))
        ).model
        model_cls = globals()[model_cfg.backbone]
        model = load_model(
            model_cls, model_cfg.arch, args.ckpt_file,
            mel_spec_type=VOCODER, vocab_file=args.vocab_file,
        )
        ref_audio, ref_text = preprocess_ref_audio_text(args.ref_audio, args.ref_text)
    except Exception as exc:
        reply({"ok": False, "error": f"khong nap duoc model: {exc}"})
        return 1

    reply({"ok": True, "ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            task = json.loads(line)
        except json.JSONDecodeError as exc:
            reply({"ok": False, "error": f"json hong: {exc}"})
            continue
        if task.get("stop"):
            break
        try:
            audio, sample_rate, _ = infer_process(
                ref_audio, ref_text, task["text"], model, vocoder,
                mel_spec_type=VOCODER,
                target_rms=target_rms,
                cross_fade_duration=cross_fade_duration,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                speed=float(task.get("speed", 1.0)),
                fix_duration=None,
            )
            sf.write(task["out"], audio, sample_rate)
            reply({"ok": True, "out": task["out"]})
        except Exception as exc:
            reply({"ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

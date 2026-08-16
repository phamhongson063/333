#!/usr/bin/env bash
set -e

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")" && pwd)/F5-TTS-Vietnamese}"
[ -d "$REPO_DIR/src/f5_tts" ] || REPO_DIR="$(pwd)"
cd "$REPO_DIR"

EPOCHS="${EPOCHS:-35}"
BATCH_FRAMES="${BATCH_FRAMES:-4000}"
MAX_SAMPLES="${MAX_SAMPLES:-16}"
WARMUP="${WARMUP:-100}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LAST_EVERY="${LAST_EVERY:-500}"
LOG_SAMPLES="${LOG_SAMPLES:-0}"
BNB="${BNB:-0}"
PRETRAIN="${PRETRAIN:-ckpts/your_training_dataset/pretrained_vn1000h.pt}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
[ "$BNB" = "1" ] && EXTRA+=(--bnb_optimizer)
[ "$LOG_SAMPLES" = "1" ] && EXTRA+=(--log_samples)

echo "repo        : $REPO_DIR"
echo "epochs      : $EPOCHS"
echo "batch frames: $BATCH_FRAMES (max $MAX_SAMPLES sample/batch)"
echo "warmup      : $WARMUP update"
echo "save        : moi $SAVE_EVERY update (model_<n>.pt, 1.3 GB)"
echo "last        : moi $LAST_EVERY update (model_last.pt, 5.4 GB)"
echo "log samples : $LOG_SAMPLES"
echo "adam 8-bit  : $BNB"
echo

accelerate launch src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name your_training_dataset \
    --tokenizer char \
    --batch_size_type frame \
    --batch_size_per_gpu "$BATCH_FRAMES" \
    --max_samples "$MAX_SAMPLES" \
    --grad_accumulation_steps 1 \
    --epochs "$EPOCHS" \
    --num_warmup_updates "$WARMUP" \
    --save_per_updates "$SAVE_EVERY" \
    --last_per_updates "$LAST_EVERY" \
    --keep_last_n_checkpoints -1 \
    --finetune \
    --pretrain "$PRETRAIN" \
    "${EXTRA[@]}"

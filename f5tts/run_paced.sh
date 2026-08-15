#!/usr/bin/env bash
set -e
set -m   # bat job control de tien trinh chay nen co PGID rieng, tach khoi script nay
cd "$(dirname "$0")/F5-TTS-Vietnamese"

RUN_SECONDS=300   # 5 phut chay
REST_SECONDS=30   # 30 giay nghi

../.venv/bin/accelerate launch --cpu src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_Base \
    --dataset_name your_training_dataset \
    --tokenizer char \
    --batch_size_type sample \
    --batch_size_per_gpu 1 \
    --max_samples 1 \
    --grad_accumulation_steps 16 \
    --num_warmup_updates 100 \
    --save_per_updates 200 \
    --last_per_updates 10 \
    --finetune \
    --log_samples \
    --pretrain ckpts/your_training_dataset/pretrained_vn1000h.pt &
PID=$!
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')

trap 'kill -CONT -$PGID 2>/dev/null; kill -TERM -$PGID 2>/dev/null; exit 0' INT TERM

while kill -0 $PID 2>/dev/null; do
    sleep "$RUN_SECONDS"
    kill -0 $PID 2>/dev/null || break
    echo "[run_paced] nghi ${REST_SECONDS}s..."
    kill -STOP -$PGID 2>/dev/null || break
    sleep "$REST_SECONDS"
    kill -CONT -$PGID 2>/dev/null || break
    echo "[run_paced] chay tiep..."
done
wait $PID

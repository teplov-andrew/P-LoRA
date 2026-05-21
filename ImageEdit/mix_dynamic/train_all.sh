#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."   # project root: /workspaces/ImageEdit

LAMS="${LAMS:-0.25 0.50 0.75}"
STEPS="${STEPS:-1000}"
CKPT_EVERY="${CKPT_EVERY:-100}"
CLASS="${CLASS:-100}"

INSTANCE_DIR="${INSTANCE_DIR:-/workspaces/data_train}"
CLASS_DIR="${CLASS_DIR:-./dataset/few_shot/class_images}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of a sks dog}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of a dog}"

if [[ ! -d "$INSTANCE_DIR" ]] || [[ -z "$(ls -A "$INSTANCE_DIR" 2>/dev/null)" ]]; then
  echo "[train_all.sh] ERROR: $INSTANCE_DIR is empty or missing." >&2
  exit 1
fi

mkdir -p "$CLASS_DIR"

for LAM in $LAMS; do
  TAG=$(python3 -c "import sys; print(f'{int(round(float(sys.argv[1])*100)):03d}')" "$LAM")
  OUT="./mix_${TAG}_dynamic"

  accelerate launch --num_processes=1 mix_dynamic/train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --instance_data_dir="$INSTANCE_DIR" \
    --class_data_dir="$CLASS_DIR" \
    --instance_prompt="$INSTANCE_PROMPT" \
    --class_prompt="$CLASS_PROMPT" \
    --output_dir="$OUT" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-4 \
    --lr_scheduler=constant \
    --lr_warmup_steps=0 \
    --max_train_steps="$STEPS" \
    --checkpointing_steps="$CKPT_EVERY" \
    --num_class_images="$CLASS" \
    --mixed_precision=fp16 \
    --enable_xformers_memory_efficient_attention \
    --gradient_checkpointing \
    --with_prior_preservation \
    --prior_loss_weight=1.0 \
    --mixture_lambda="$LAM" \
    --alpha_left=1.0 \
    --alpha_right=1.0 \
    --seed=42
done

echo
echo "[train_all.sh] All runs finished. Checkpoints in mix_*_dynamic/checkpoint-*/."

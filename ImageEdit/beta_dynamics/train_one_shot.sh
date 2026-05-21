#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-1000}"
CKPT_EVERY="${CKPT_EVERY:-100}"
CLASS="${CLASS:-100}"

INSTANCE_DIR="${INSTANCE_DIR:-../data_train_one}"
CLASS_DIR="${CLASS_DIR:-./dataset/few_shot/class_images}"
INSTANCE_PROMPT="${INSTANCE_PROMPT:-a photo of a sks dog}"
CLASS_PROMPT="${CLASS_PROMPT:-a photo of a dog}"

if [[ ! -d "$INSTANCE_DIR" ]] || [[ -z "$(ls -A "$INSTANCE_DIR" 2>/dev/null)" ]]; then
  echo "[train_one_shot.sh] ERROR: $INSTANCE_DIR is empty or missing." >&2
  exit 1
fi

echo "[train_one_shot.sh] One-shot image: $(ls "$INSTANCE_DIR")"
mkdir -p "$CLASS_DIR"

declare -a CONFIGS=(
  "0.5 0.5 u_shaped_05_05_one_shot"
  "5.0 5.0 center_5_5_one_shot"
)

for cfg in "${CONFIGS[@]}"; do
  read -r A B SUBDIR <<< "$cfg"
  OUT="./beta_dynamics/${SUBDIR}"

  echo
  echo "============================================================"
  echo "[beta_dynamics one-shot] Beta(alpha=$A, beta=$B)  ->  $OUT"
  echo "============================================================"

  accelerate launch --num_processes=1 beta_dynamics/train_dreambooth_lora_sdxl.py \
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
    --beta_alpha="$A" \
    --beta_beta="$B" \
    --seed=42
done

echo
echo "[beta_dynamics one-shot] All runs finished."

#!/bin/bash

cd ..

LLM_VERSION=TinyLlama/TinyLlama-1.1B-Chat-v1.0
# VT_VERSION=openai/clip-vit-base-patch16
VT_VERSION=eva_clip/EVA02_CLIP_L_336_psz14_s6B.pt
DATA_PATH=/root/autodl-tmp/code/llava/llava_pretrain/blip_laion_cc_sbu_558k.json
IMAGE_PATH=/root/autodl-tmp/code/llava/llava_pretrain/images
VT_VARIANT="${VT_VERSION#*/}"
LLM_VARIANT="${LLM_VERSION#*/}"
VERSION=share
TUNE_FROM=12

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $LLM_VERSION \
    --version plain \
    --data_path  $DATA_PATH\
    --image_folder $IMAGE_PATH \
    --vision_tower $VT_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --tune_entire_model True \
    --tune_vit_from_layer $TUNE_FROM \
    --pretrain_mm_mlp_adapter ./checkpoints/tiny-llava-${VERSION}-${LLM_VARIANT}-${VT_VARIANT}-pretrain/mm_projector.bin \
    --bf16 True \
    --output_dir ./checkpoints/tiny-llava-"${VERSION}"-"${LLM_VARIANT}"-"${VT_VARIANT}"-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 15 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name tiny-llava-"${VERSION}"-pretrain-"${LLM_VARIANT}"-"${VT_VARIANT}"
#!/bin/bash



CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
    --model_name Llama-3.2-1B-Instruct \
    --dataset TOFU_QA_forget01 \
    --trainer_config configs/trainer/trainer_config.yaml \
    --output_dir outputs/Llama-3.2-1B-Instruct_TOFU_QA_forget01 \
    --per_device_train_batch_size 4 \


# CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
#     --model_name Llama-3.2-1B-Instruct \
#     --dataset TOFU_QA_retain99_ft \
#     --trainer_config configs/trainer/trainer_config.yaml \
#     --output_dir outputs/Llama-3.2-1B-Instruct_TOFU_QA_retain99 \
#     --per_device_train_batch_size 4 \


# CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
#     --model_name Llama-3.2-1B-Instruct \
#     --dataset TOFU_QA_full \
#     --trainer_config configs/trainer/trainer_config.yaml \
#     --output_dir outputs/Llama-3.2-1B-Instruct_TOFU_QA_full \
#     --per_device_train_batch_size 4 \
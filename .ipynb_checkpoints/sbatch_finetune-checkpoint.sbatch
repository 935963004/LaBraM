#!/bin/bash
#SBATCH --job-name=zarina_labram_finetuning
#SBATCH --error=runs/zarina_labram_finetuning-%j.err
#SBATCH --output=runs/zarina_labram_finetuning-%j.log
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --constraint="[type_a|type_b|type_c|type_e]"

module load Python
module load CUDA/11.7
#module load anaconda/3

export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

source activate zarina_labram

NORMAL_PATH="/home/dmedvedeva/TUAB/edf/train/normal/01_tcp_ar"
ABNORMAL_PATH="/home/dmedvedeva/TUAB/edf/train/abnormal/01_tcp_ar"

python _run_class_finetuning_on_tuab_with_triplet_loss.py \
    --normal_path $NORMAL_PATH \
    --abnormal_path $ABNORMAL_PATH \
    --triplet_loss_weight 0.1 \
    --output_dir ./checkpoints/finetune_tuab_triplet/ \
    --log_dir ./log/finetune_tuab_triplet/ \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --weight_decay 0.05 \
    --batch_size 64 \
    --lr 5e-4 \
    --update_freq 1 \
    --warmup_epochs 3 \
    --epochs 30 \
    --layer_decay 0.65 \
    --drop_path 0.1 \
    --dist_eval \
    --save_ckpt_freq 5 \
    --disable_rel_pos_bias \
    --abs_pos_emb \
    --dataset TUAB \
    --disable_qkv_bias \
    --seed 0
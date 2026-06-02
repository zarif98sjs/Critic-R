#!/bin/bash
#SBATCH --job-name=<job_name>       # Job name
#SBATCH --output=job_%j.out         # Output file (%j = job ID)
#SBATCH --error=job_%j.err          # Error file
#SBATCH --partition=          # Partition/queue name
#SBATCH --gres=gpu:a100:1           # Number of GPUs needed
#SBATCH --mem=200g  # Requested Memory
#SBATCH --time=1-00:00:00 # Job time limit, 14 days

CUDA_VISIBLE_DEVICES=0

source ../miniconda3/bin/activate
conda activate sr1

export HF_HOME="./cache"

#!/bin/bash

python stella_finetune.py \
    --data_path <data_path> \
    --output_dir <output_dir> \
    --model_name NovaSearch/stella_en_400M_v5 \
    --embedding_dim 1024 \
    --num_epochs 5 \
    --batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --temperature 0.02 \
    --oversample_ratio 4.0 \
    --max_hard_negatives 3 \
    --hard_negative_weight 1.0 \
    --warmup_ratio 0.1 \
    --fp16 \
    --val_size 2000

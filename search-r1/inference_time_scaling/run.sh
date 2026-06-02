#!/bin/bash
#SBATCH --job-name=critic-r-zero-job     # Job name
#SBATCH --output=job_%j.out         # Output file (%j = job ID)
#SBATCH --error=job_%j.err          # Error file
#SBATCH --partition=superpod-a100          # Partition/queue name
#SBATCH --gres=gpu:a100:1           # Number of GPUs needed
#SBATCH --mem=200g  # Requested Memory
#SBATCH --time=1-00:00:00 # Job time limit, 14 days

CUDA_VISIBLE_DEVICES=0

source ../miniconda3/bin/activate
conda activate sr1

export HF_HOME="./cache"
echo "Running on host: $(hostname)"

python critic-r-zero.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="<retrieval_model>" --retrieval_endpoint="http://<gpu_id>:8000/retrieve" --evaluator_endpoint="http://<gpu_id>:8001/v1/chat/completions"


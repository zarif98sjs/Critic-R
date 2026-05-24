#!/bin/bash
#SBATCH --job-name=vllm_server_qwen2.5_72B        # Job name
#SBATCH --output=job_%j.out         # Output file (%j = job ID)
#SBATCH --error=job_%j.err          # Error file
#SBATCH --partition=          # Partition/queue name
#SBATCH --exclude=          # Exclude specific nodes
#SBATCH --gres=gpu:a100:2           # Number of GPUs needed
#SBATCH --nodes=1
#SBATCH --mem=200g  # Requested Memory
#SBATCH --time=5-00:00:00 # Job time limit, 14 days
#SBATCH --qos=long


source ../miniconda3/bin/activate
conda activate debug

export HF_HOME="./cache"

echo "Running on host: $(hostname)"

python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --port 8001 --max-model-len 4096 --gpu-memory-utilization 0.95 --dtype bfloat16 --tensor-parallel-size 2

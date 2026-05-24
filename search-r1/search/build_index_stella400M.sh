#!/bin/bash
#SBATCH --job-name=build_index__stella400M       # Job name
#SBATCH --output=job_%j.out         # Output file (%j = job ID)
#SBATCH --error=job_%j.err          # Error file
#SBATCH --partition=           # Partition/queue name
#SBATCH --gres=gpu:a100:4              # Number of GPUs needed
#SBATCH --mem=800g  # Requested Memory
#SBATCH --time 1-00:00:00  # Job time limit, 3 days

source ../../../miniconda3/bin/activate
conda activate retriever
export HF_HOME="./cache"

corpus_file=../../corpus/wiki-18.jsonl # jsonl
save_dir=../../corpus
retriever_name=stella_en_400M_v5 # this is for indexing naming
retriever_model=../../../stella_en_400M_v5

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3 python index_builder__stella400M.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding

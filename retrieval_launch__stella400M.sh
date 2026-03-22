#!/bin/bash
#SBATCH --job-name=retriever__stella400M        # Job name
#SBATCH --output=job_%j.out         # Output file (%j = job ID)
#SBATCH --error=job_%j.err          # Error file
#SBATCH --partition=gpu           # Partition/queue name
#SBATCH --nodes=1
#SBATCH --gres=gpu:2080ti:8           # Number of GPUs needed
#SBATCH --mem=180g  # Requested Memory
#SBATCH --time=14-00:00:00 # Job time limit, 14 days
#SBATCH --qos=long

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

file_path=corpus
index_file=$file_path/stella_en_400m_v5_Flat.index
corpus_file=$file_path/wiki_dump.jsonl
retriever_name=stella_en_400M_v5
retriever_path=../stella_en_400M_v5

source ../miniconda3/bin/activate
conda activate retriever

export HF_HOME="./cache"
echo "Running on host: $(hostname)"
python search_r1/search/retrieval_server__stella400M.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu

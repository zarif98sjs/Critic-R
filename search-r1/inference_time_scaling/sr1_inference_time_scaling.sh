#!/bin/bash
#SBATCH --job-name=SIGSHORT_sr1_inference_time_scaling_job_opt2v2_nq_3b_32B     # Job name
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

###
### Qwen/Qwen2.5-7B-Instruct         :    http://gpu019:8001/v1/chat/completions
### Qwen/Qwen2.5-14B-Instruct        :    http://gpu016:8001/v1/chat/completions
### Qwen/Qwen2.5-32B-Instruct        :    http://gpu014:8001/v1/chat/completions
### Qwen/Qwen2.5-32B-Instruct-AWQ    :    http://:8001/v1/chat/completions
### Qwen/Qwen2.5-72B-Instruct-AWQ    :    http://:8001/v1/chat/completions


# BAMBOOGLE
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=bamboogle --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# MUSIQUE

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=musique --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"


# HOTPOTQA
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"



# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=hotpotqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"



##### 2wikimultihopqa ######

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=2wikimultihopqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-7B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu019:8001/v1/chat/completions"

# ##### nq ######

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=nq --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"


###### triviaqa ######

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=triviaqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"


###### popqa ######

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-14b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu018:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"
# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-32B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu014:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"

# python SIGSHORTv2_sr1_inference_time_scaling__opt2v2.py --dataset=popqa --reasoning_model_id="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-3b-it-em-grpo-v0.3" --evaluator_model_id="Qwen/Qwen2.5-14B-Instruct" --retrieval_model_id="og" --retrieval_endpoint="http://gypsum-gpu177:8000/retrieve" --evaluator_endpoint="http://gpu016:8001/v1/chat/completions"



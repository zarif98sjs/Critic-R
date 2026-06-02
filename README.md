# Critic-R: Improving Agentic Search using Instruction-tuned Retrievers with Natural Language Introspective Feedback

<p align="center">
  <a href="https://arxiv.org/abs/2606.00590"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b"></a>
  <a href="https://huggingface.co/zarif98sjs/Critic-Embed"><img alt="Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow"></a>
  <a href="https://github.com/zarif98sjs/Critic-R"><img alt="Code" src="https://img.shields.io/badge/%F0%9F%9A%80%20GitHub-Code-blue"></a>
</p>

![Critic-R Framework Overview](Critic-R.png)

Agentic search systems iteratively interact with retrieval models to answer complex queries. Despite substantial progress, optimizing retrievers for agentic search remains challenging, often requiring heavy co-training or gold-standard annotations that limit real-world applicability. We propose Critic-R, a framework that explicitly closes the feedback loop between the reasoning agent and the retrieval model during both inference and training. Critic-R introduces a critic model that evaluates the agent's introspective reasoning trace after consuming retrieved evidence to determine whether the retrieved context sufficiently supports the next reasoning step. Critic-R has two complementary mechanisms: Critic-R-Zero, an inference-time query refinement loop that iteratively rewrites queries and retrieval instructions, and Critic-Embed, an optimization approach for retrieval models that leverages successful and failed refinement trajectories as automatic supervision without requiring manual relevance annotation. We evaluate Critic-R on HotpotQA, 2WikiMultihopQA, MuSiQue, and Bamboogle. Results show that Critic-R significantly improves both retrieval quality and downstream answer accuracy.

---

## Installation

Critic-R requires **Python 3.10+** and CUDA-capable GPUs. We recommend two separate conda environments — one for inference/serving and one for retriever training — to avoid `torch`/`vllm`/`faiss` version conflicts.

### 1. Inference / serving environment

```bash
conda create -n sr1 python=3.10 -y
conda activate sr1

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install transformers==4.45.0 datasets accelerate
pip install aiohttp requests tqdm numpy
```

### 2. Retriever environment

```bash
conda create -n retriever python=3.10 -y
conda activate retriever

pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.0 sentence-transformers
pip install faiss-gpu==1.7.2
pip install deepspeed
pip install xformers
```

## Building the Wikipedia Index

```bash
cd search-r1/search
bash build_index_stella400M.sh
```

This produces a Flat FAISS index over `wiki-18.jsonl` using mean-pooled Stella embeddings (`max_length=256`, `fp16`). Swap `--faiss_type Flat` for `HNSW32/64/128` to use approximate search, or set `--retrieval_method bm25` for a lexical baseline.


## Quick Start

A full Critic-R inference run involves three services and one driver script:

1. **Critic LLM server** (vLLM, OpenAI-compatible API).
2. **Dense retrieval server** (FAISS over Wikipedia-18).
3. **Driver**, which hosts the reasoning agent locally and orchestrates the loop.

```bash
# (1) Launch the critic LLM
sbatch vllm_server_job_multigpu.sh                       # -> http://<host>:8001/v1/chat/completions

# (2) Launch the retrieval server
sbatch retrieval_launch__stella400M.sh                   # -> http://<host>:8000/retrieve

# (3) Run the Critic-R loop
cd search-r1/inference_time_scaling
bash run.sh
```


## Citation

If you find this work useful, please cite:

[Critic-R: Improving Agentic Search using Instruction-tuned Retrievers with Natural Language Introspective Feedback](https://arxiv.org/abs/2606.00590)
```bibtex
@misc{alam2026criticrimprovingagenticsearch,
      title={Critic-R: Improving Agentic Search using Instruction-tuned Retrievers with Natural Language Introspective Feedback}, 
      author={Md Zarif Ul Alam and Alireza Salemi and Hamed Zamani},
      year={2026},
      eprint={2606.00590},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2606.00590}, 
}
```

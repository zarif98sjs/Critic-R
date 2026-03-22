import os
import faiss
import json
import warnings
import numpy as np
from typing import cast, List, Dict
import shutil
import subprocess
import argparse
import torch
from tqdm import tqdm
# from LongRAG.retriever.utils import load_model, load_corpus, pooling
import datasets
from transformers import AutoTokenizer, AutoModel, AutoConfig


def load_model(
        model_path: str, 
        use_fp16: bool = False
    ):
    # vector_dim set to 768 as requested
    vector_dim = 768
    vector_linear_directory = f"2_Dense_{vector_dim}"
    
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    
    # Initialize and load the specific Linear layer for Stella
    vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim)
    
    # Load weights from the specific sub-directory in the model folder
    linear_path = os.path.join(model_path, f"{vector_linear_directory}/pytorch_model.bin")
    if os.path.exists(linear_path):
        vector_linear_dict = {
            k.replace("linear.", ""): v for k, v in
            torch.load(linear_path).items()
        }
        vector_linear.load_state_dict(vector_linear_dict)
    else:
        raise FileNotFoundError(f"Could not find linear layer weights at {linear_path}")

    vector_linear.cuda()

    if use_fp16: 
        model = model.half()
        vector_linear = vector_linear.half()

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)

    return model, tokenizer, vector_linear

def pooling(
        pooler_output,
        last_hidden_state,
        attention_mask = None,
        pooling_method = "mean"
    ):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
            'json', 
            data_files=corpus_path,
            split="train",
            num_proc=4)
    return corpus


class Index_Builder:
    r"""A tool class used to build an index used in retrieval.
    
    """
    def __init__(
            self, 
            retrieval_method,
            model_path,
            corpus_path,
            save_dir,
            max_length,
            batch_size,
            use_fp16,
            pooling_method,
            faiss_type=None,
            embedding_path=None,
            save_embedding=False,
            faiss_gpu=False
        ):
        
        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.pooling_method = pooling_method
        self.faiss_type = faiss_type if faiss_type is not None else 'Flat'
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu

        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)

        self.index_save_path = os.path.join(self.save_dir, f"{self.retrieval_method}_{self.faiss_type}.index")

        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")

        self.corpus = load_corpus(self.corpus_path)
       
        print("Finish loading...")
    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content.
        
        """
        
        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True

    def build_index(self):
        r"""Constructing different indexes based on selective retrieval method.

        """
        if self.retrieval_method == "bm25":
            self.build_bm25_index()
        else:
            self.build_dense_index()

    def build_bm25_index(self):
        """Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        # to use pyserini pipeline, we first need to place jsonl file in the folder 
        self.save_dir = os.path.join(self.save_dir, "bm25")
        os.makedirs(self.save_dir, exist_ok=True)
        temp_dir = self.save_dir + "/temp"
        temp_file_path = temp_dir + "/temp.jsonl"
        os.makedirs(temp_dir)

        # if self.have_contents:
        #     shutil.copyfile(self.corpus_path, temp_file_path)
        # else:
        #     with open(temp_file_path, "w") as f:
        #         for item in self.corpus:
        #             f.write(json.dumps(item) + "\n")
        shutil.copyfile(self.corpus_path, temp_file_path)
        
        print("Start building bm25 index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", temp_dir,
                         "--index", self.save_dir,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]
       
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)

        shutil.rmtree(temp_dir)
        
        print("Finish!")

    def _load_embedding(self, embedding_path, corpus_size, hidden_size):
        all_embeddings = np.memmap(
                embedding_path,
                mode="r",
                dtype=np.float32
            ).reshape(corpus_size, hidden_size)
        return all_embeddings

    def _save_embedding(self, all_embeddings):
        memmap = np.memmap(
            self.embedding_save_path,
            shape=all_embeddings.shape,
            mode="w+",
            dtype=all_embeddings.dtype
        )
        length = all_embeddings.shape[0]
        # add in batch
        save_batch_size = 10000
        if length > save_batch_size:
            for i in tqdm(range(0, length, save_batch_size), leave=False, desc="Saving Embeddings"):
                j = min(i + save_batch_size, length)
                memmap[i: j] = all_embeddings[i: j]
        else:
            memmap[:] = all_embeddings

    def encode_all(self):
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.encoder = torch.nn.DataParallel(self.encoder)
            # vector_linear is small, usually kept on main device or replicated if needed, 
            # but for simplicity here we assume single-device logic or manual handling for the linear layer 
            # if strictly required in DP.
            self.batch_size = self.batch_size * self.gpu_num
            print(f"New batch size: {self.batch_size}")

        all_embeddings = []

        for start_idx in tqdm(range(0, len(self.corpus), self.batch_size), desc='Inference Embeddings:'):

            batch_data = self.corpus[start_idx:start_idx+self.batch_size]['contents']

            # Stella does not require specific "passage: " prefixes for docs like E5
            # but if your specific use case requires prompts, add them here.
            
            inputs = self.tokenizer(
                        batch_data,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=self.max_length,
            ).to('cuda')

            inputs = {k: v.cuda() for k, v in inputs.items()}

            # --- Modified Logic for Stella ---
            output = self.encoder(**inputs)
            last_hidden_state = output[0]
            
            # 1. Masked Mean Pooling (as per docs)
            attention_mask = inputs['attention_mask']
            last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            
            # 2. Linear Projection
            embeddings = self.vector_linear(embeddings)
            
            # 3. Normalize
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
            # ---------------------------------

            embeddings = cast(torch.Tensor, embeddings)
            embeddings = embeddings.detach().cpu().numpy()
            all_embeddings.append(embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_embeddings = all_embeddings.astype(np.float32)

        return all_embeddings


    @torch.no_grad()
    def build_dense_index(self):
        """Obtain the representation of documents based on the embedding model(BERT-based) and 
        construct a faiss index.
        """
        
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        
        # Modified unpacking to receive vector_linear
        self.encoder, self.tokenizer, self.vector_linear = load_model(
            model_path=self.model_path, 
            use_fp16=self.use_fp16
        )
        
        if self.embedding_path is not None:
            hidden_size = 768 # Force 768 for Stella
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus

        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, self.faiss_type, faiss.METRIC_INNER_PRODUCT)
        
        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)

        faiss.write_index(faiss_index, self.index_save_path)
        print("Finish!")


MODEL2POOLING = {
    "e5": "mean",
    "bge": "cls",
    "contriever": "mean",
    'jina': 'mean'
}


def main():
    parser = argparse.ArgumentParser(description = "Creating index.")

    # Basic parameters
    parser.add_argument('--retrieval_method', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--save_dir', default= 'indexes/',type=str)

    # Parameters for building dense index
    parser.add_argument('--max_length', type=int, default=180)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--use_fp16', default=False, action='store_true')
    parser.add_argument('--pooling_method', type=str, default=None)
    parser.add_argument('--faiss_type',default=None,type=str)
    parser.add_argument('--embedding_path', default=None, type=str)
    parser.add_argument('--save_embedding', action='store_true', default=False)
    parser.add_argument('--faiss_gpu', default=False, action='store_true')
    
    args = parser.parse_args()

    if args.pooling_method is None:
        pooling_method = 'mean'
        for k,v in MODEL2POOLING.items():
            if k in args.retrieval_method.lower():
                pooling_method = v
                break
    else:
        if args.pooling_method not in ['mean','cls','pooler']:
            raise NotImplementedError
        else:
            pooling_method = args.pooling_method


    index_builder = Index_Builder(
                        retrieval_method = args.retrieval_method,
                        model_path = args.model_path,
                        corpus_path = args.corpus_path,
                        save_dir = args.save_dir,
                        max_length = args.max_length,
                        batch_size = args.batch_size,
                        use_fp16 = args.use_fp16,
                        pooling_method = pooling_method,
                        faiss_type = args.faiss_type,
                        embedding_path = args.embedding_path,
                        save_embedding = args.save_embedding,
                        faiss_gpu = args.faiss_gpu
                    )
    index_builder.build_index()


if __name__ == "__main__":
    main()
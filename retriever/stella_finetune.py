#!/usr/bin/env python3
"""
INLF Contrastive Fine-tuning for Stella-400M Retriever

Combines:
  - Stage 1: Natural contrastive pairs from INLF refinement (7,196 samples with neg+pos)
  - Stage 2: Positive-only samples with in-batch negatives (46,077 samples)

Features:
  - InfoNCE loss with INLF hard negatives + in-batch negatives
  - Instruction-aware query encoding (Stella's Instruct format)
  - Oversampling of high-value natural pairs
  - Matryoshka Representation Learning (MRL) support
  - GradCache for large effective batch sizes on limited VRAM
"""

import argparse
import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Data Loading & Preparation
# ============================================================================

@dataclass
class INLFSample:
    """A single training sample from INLF data collection."""
    query: str
    instruction: str
    positive_doc: str
    hard_negatives: List[str]  # INLF negatives (empty for positive-only samples)
    has_inlf_negatives: bool   # Whether this sample has natural contrastive pairs
    

def load_inlf_data(path: str) -> List[INLFSample]:
    """
    Load INLF JSONL and convert to training samples.
    
    Each JSONL row has:
      - query: str
      - positive_documents: list of {attempt, query, instruction, documents, reasoning_feedback}
      - negative_documents: list of {attempt, query, instruction, documents, reasoning_feedback}
      - final_answer: str
      - ground_truths: list[str]
    """
    samples = []
    skipped = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            
            pos_docs = record.get("positive_documents", [])
            neg_docs = record.get("negative_documents", [])
            
            # Skip samples with no positive documents
            if not pos_docs:
                skipped += 1
                continue
            
            # Use the last positive (the one that succeeded) as the training target
            pos = pos_docs[-1]
            positive_doc = pos.get("documents", "").strip()
            instruction = pos.get("instruction", "Given a query, retrieve relevant passages that answer the query.")
            query = record.get("query", pos.get("original_query", ""))
            
            if not positive_doc or not query:
                skipped += 1
                continue
            
            # Collect INLF hard negatives (documents from failed attempts)
            hard_negatives = []
            for neg in neg_docs:
                neg_doc = neg.get("documents", "").strip()
                if neg_doc and neg_doc != positive_doc:
                    hard_negatives.append(neg_doc)
            
            samples.append(INLFSample(
                query=query,
                instruction=instruction,
                positive_doc=positive_doc,
                hard_negatives=hard_negatives,
                has_inlf_negatives=len(hard_negatives) > 0,
            ))
    
    n_with_negs = sum(1 for s in samples if s.has_inlf_negatives)
    logger.info(f"Loaded {len(samples)} samples ({n_with_negs} with INLF negatives, "
                f"{len(samples) - n_with_negs} positive-only). Skipped {skipped} records.")
    return samples


class INLFContrastiveDataset(Dataset):
    """
    Dataset that handles both natural pairs and positive-only samples.
    
    Oversamples natural pairs (has_inlf_negatives=True) by `oversample_ratio`
    so they appear more frequently in training despite being the minority.
    """
    
    def __init__(
        self,
        samples: List[INLFSample],
        oversample_ratio: float = 4.0,
        max_hard_negatives: int = 3,
    ):
        self.max_hard_negatives = max_hard_negatives
        
        # Separate into two pools
        self.natural_pairs = [s for s in samples if s.has_inlf_negatives]
        self.positive_only = [s for s in samples if not s.has_inlf_negatives]
        
        # Build oversampled index list
        # Natural pairs appear `oversample_ratio` times more often
        natural_indices = list(range(len(self.natural_pairs)))
        positive_indices = list(range(len(self.natural_pairs), 
                                      len(self.natural_pairs) + len(self.positive_only)))
        
        n_natural_repeats = max(1, int(oversample_ratio))
        self.index_pool = (natural_indices * n_natural_repeats) + positive_indices
        
        # Combined sample list for indexing
        self.all_samples = self.natural_pairs + self.positive_only
        
        logger.info(f"Dataset: {len(self.natural_pairs)} natural pairs (×{n_natural_repeats}), "
                     f"{len(self.positive_only)} positive-only. "
                     f"Effective epoch size: {len(self.index_pool)}")
    
    def __len__(self):
        return len(self.index_pool)
    
    def __getitem__(self, idx):
        real_idx = self.index_pool[idx]
        sample = self.all_samples[real_idx]
        
        # Truncate hard negatives to max
        hard_negs = sample.hard_negatives[:self.max_hard_negatives]
        
        return {
            "query": sample.query,
            "instruction": sample.instruction,
            "positive_doc": sample.positive_doc,
            "hard_negatives": hard_negs,
            "num_hard_negatives": len(hard_negs),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate that handles variable numbers of hard negatives.
    
    Returns flat lists for queries, positives, and all negatives,
    plus metadata to reconstruct which negatives belong to which query.
    """
    queries = []
    instructions = []
    positives = []
    all_hard_negatives = []
    neg_counts = []  # how many hard negatives each query has
    
    for item in batch:
        queries.append(item["query"])
        instructions.append(item["instruction"])
        positives.append(item["positive_doc"])
        all_hard_negatives.extend(item["hard_negatives"])
        neg_counts.append(item["num_hard_negatives"])
    
    return {
        "queries": queries,
        "instructions": instructions,
        "positives": positives,
        "hard_negatives": all_hard_negatives,
        "neg_counts": neg_counts,
    }


# ============================================================================
# Model Wrapper
# ============================================================================

class StellaRetriever(nn.Module):
    """
    Wrapper around Stella-400M for contrastive training.
    
    Stella uses:
      - Query format: "Instruct: {instruction}\nQuery: {query}"
      - Document format: raw text (no prefix)
      - Last-token pooling
      - L2 normalization
    """
    
    def __init__(self, model_name: str, embedding_dim: int = 1024):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.embedding_dim = embedding_dim
        
        # Stella has a dense projection layer for each MRL dimension
        # We'll use the model's built-in pooling via forward
        
    def _last_token_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool by taking the last non-padding token's hidden state."""
        # Find the position of the last non-padding token for each sequence
        sequence_lengths = attention_mask.sum(dim=1) - 1  # 0-indexed
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    def encode(self, texts: List[str], is_query: bool = False, instructions: List[str] = None) -> torch.Tensor:
        """
        Encode texts into normalized embeddings.
        
        For queries: prepend "Instruct: {instruction}\nQuery: "
        For documents: encode as-is
        """
        if is_query:
            assert instructions is not None and len(instructions) == len(texts)
            formatted = [
                f"Instruct: {inst}\nQuery: {q}" 
                for inst, q in zip(instructions, texts)
            ]
        else:
            formatted = texts
        
        encoded = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        
        outputs = self.model(**encoded)
        embeddings = self._last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        
        # Normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
    
    def forward(self, queries, instructions, positives, hard_negatives=None):
        """
        Forward pass returning query, positive, and negative embeddings.
        """
        q_emb = self.encode(queries, is_query=True, instructions=instructions)
        p_emb = self.encode(positives, is_query=False)
        
        n_emb = None
        if hard_negatives and len(hard_negatives) > 0:
            n_emb = self.encode(hard_negatives, is_query=False)
        
        return q_emb, p_emb, n_emb


# ============================================================================
# Loss Function
# ============================================================================

class INLFContrastiveLoss(nn.Module):
    """
    InfoNCE loss with INLF hard negatives + in-batch negatives.
    
    For each query q_i:
      - Positive: its paired document d_i+
      - In-batch negatives: all other positives in the batch {d_j+ : j ≠ i}
      - INLF hard negatives: documents from failed retrieval attempts {d_i^neg_k}
    
    Optionally supports Matryoshka Representation Learning (MRL) by computing
    the loss at multiple embedding dimensions.
    """
    
    def __init__(
        self, 
        temperature: float = 0.02,
        hard_negative_weight: float = 1.0,
        mrl_dims: Optional[List[int]] = None,
        mrl_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.mrl_dims = mrl_dims
        
        if mrl_dims and mrl_weights:
            assert len(mrl_dims) == len(mrl_weights)
            self.mrl_weights = mrl_weights
        elif mrl_dims:
            # Default: equal weight for each dimension
            self.mrl_weights = [1.0 / len(mrl_dims)] * len(mrl_dims)
        else:
            self.mrl_weights = None
    
    def _compute_loss_at_dim(
        self,
        q_emb: torch.Tensor,
        p_emb: torch.Tensor,
        n_emb: Optional[torch.Tensor],
        neg_counts: List[int],
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute InfoNCE at a specific embedding dimension (or full dim if None)."""
        
        if dim is not None:
            q = F.normalize(q_emb[:, :dim], p=2, dim=-1)
            p = F.normalize(p_emb[:, :dim], p=2, dim=-1)
            n = F.normalize(n_emb[:, :dim], p=2, dim=-1) if n_emb is not None else None
        else:
            q, p, n = q_emb, p_emb, n_emb
        
        batch_size = q.shape[0]
        
        # Positive scores: (B,)
        pos_scores = (q * p).sum(dim=-1) / self.temperature
        
        # In-batch negative scores: (B, B) — all positives as candidates
        inbatch_scores = torch.mm(q, p.t()) / self.temperature
        
        # Build the full logits matrix
        # Column 0 = positive, columns 1..B-1 = in-batch negs, remaining = INLF hard negs
        if n is not None and n.shape[0] > 0:
            # Hard negative scores: each query has a variable number
            # We compute all q-n similarities and mask appropriately
            hard_neg_scores = torch.mm(q, n.t()) / self.temperature  # (B, total_hard_negs)
            
            # Create per-query hard negative mask
            # neg_counts[i] tells us how many hard negs belong to query i
            total_hard_negs = n.shape[0]
            hard_neg_mask = torch.zeros(batch_size, total_hard_negs, device=q.device)
            offset = 0
            for i, count in enumerate(neg_counts):
                if count > 0:
                    hard_neg_mask[i, offset:offset + count] = 1.0
                offset += count
            
            # Apply mask: set non-belonging hard negs to -inf
            hard_neg_scores = hard_neg_scores * hard_neg_mask + (1 - hard_neg_mask) * (-1e9)
            
            # Optionally upweight hard negatives 
            if self.hard_negative_weight != 1.0:
                hard_neg_scores = hard_neg_scores * self.hard_negative_weight
            
            # Full logit matrix: [in-batch | hard_negatives]
            all_scores = torch.cat([inbatch_scores, hard_neg_scores], dim=1)  # (B, B + total_hard_negs)
        else:
            all_scores = inbatch_scores  # (B, B)
        
        # Labels: the positive for query i is at index i in the in-batch scores
        labels = torch.arange(batch_size, device=q.device)
        
        loss = F.cross_entropy(all_scores, labels)
        return loss
    
    def forward(
        self,
        q_emb: torch.Tensor,
        p_emb: torch.Tensor,
        n_emb: Optional[torch.Tensor],
        neg_counts: List[int],
    ) -> torch.Tensor:
        
        if self.mrl_dims:
            # Matryoshka: compute loss at multiple dimensions
            total_loss = 0.0
            for dim, weight in zip(self.mrl_dims, self.mrl_weights):
                total_loss += weight * self._compute_loss_at_dim(q_emb, p_emb, n_emb, neg_counts, dim=dim)
            return total_loss
        else:
            return self._compute_loss_at_dim(q_emb, p_emb, n_emb, neg_counts)


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # ── Load data ──
    logger.info(f"Loading data from {args.data_path}")
    all_samples = load_inlf_data(args.data_path)
    
    # Train/val split
    random.shuffle(all_samples)
    val_size = min(args.val_size, int(len(all_samples) * 0.05))
    val_samples = all_samples[:val_size]
    train_samples = all_samples[val_size:]
    
    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    train_dataset = INLFContrastiveDataset(
        train_samples,
        oversample_ratio=args.oversample_ratio,
        max_hard_negatives=args.max_hard_negatives,
    )
    
    val_dataset = INLFContrastiveDataset(
        val_samples,
        oversample_ratio=1.0,  # No oversampling for validation
        max_hard_negatives=args.max_hard_negatives,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,  # Important: keep batch sizes consistent for in-batch negatives
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # ── Model ──
    logger.info(f"Loading model: {args.model_name}")
    model = StellaRetriever(args.model_name, embedding_dim=args.embedding_dim)
    model = model.to(device)
    
    # ── Loss ──
    mrl_dims = [int(d) for d in args.mrl_dims.split(",")] if args.mrl_dims else None
    loss_fn = INLFContrastiveLoss(
        temperature=args.temperature,
        hard_negative_weight=args.hard_negative_weight,
        mrl_dims=mrl_dims,
    )
    
    # ── Optimizer & Scheduler ──
    # Separate LR for backbone vs projection head (if applicable)
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # ── Mixed precision ──
    scaler = torch.amp.GradScaler('cuda') if args.fp16 and device.type == 'cuda' else None
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if args.fp16 and device.type == 'cuda' else torch.amp.autocast('cuda', enabled=False)
    
    # ── Gradient accumulation ──
    accum_steps = args.gradient_accumulation_steps
    
    # ── Training ──
    logger.info("Starting training...")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {accum_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * accum_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Hard negative weight: {args.hard_negative_weight}")
    logger.info(f"  Oversample ratio: {args.oversample_ratio}")
    if mrl_dims:
        logger.info(f"  MRL dimensions: {mrl_dims}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    best_val_loss = float("inf")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad()
        
        for step, batch in enumerate(pbar):
            with autocast_ctx:
                q_emb, p_emb, n_emb = model(
                    queries=batch["queries"],
                    instructions=batch["instructions"],
                    positives=batch["positives"],
                    hard_negatives=batch["hard_negatives"],
                )
                
                loss = loss_fn(q_emb, p_emb, n_emb, batch["neg_counts"])
                loss = loss / accum_steps
            
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (step + 1) % accum_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * accum_steps
            epoch_steps += 1
            
            pbar.set_postfix({
                "loss": f"{epoch_loss / epoch_steps:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })
            
            # Periodic logging
            if global_step > 0 and global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_steps
                logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        # ── Validation ──
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                with autocast_ctx:
                    q_emb, p_emb, n_emb = model(
                        queries=batch["queries"],
                        instructions=batch["instructions"],
                        positives=batch["positives"],
                        hard_negatives=batch["hard_negatives"],
                    )
                    loss = loss_fn(q_emb, p_emb, n_emb, batch["neg_counts"])
                
                val_loss_total += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss_total / max(val_steps, 1)
        avg_train_loss = epoch_loss / max(epoch_steps, 1)
        
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-epoch{epoch+1}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model.model.save_pretrained(checkpoint_dir)
        model.tokenizer.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_dir = os.path.join(args.output_dir, "best")
            os.makedirs(best_dir, exist_ok=True)
            model.model.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            logger.info(f"New best model (val_loss={best_val_loss:.4f}) saved to {best_dir}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.model.save_pretrained(final_dir)
    model.tokenizer.save_pretrained(final_dir)
    logger.info(f"Training complete. Final model saved to {final_dir}")
    
    # Save training config
    with open(os.path.join(args.output_dir, "training_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INLF Contrastive Fine-tuning for Stella-400M")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to INLF training data JSONL")
    parser.add_argument("--output_dir", type=str, default="./stella_inlf_finetuned",
                        help="Output directory for checkpoints")
    parser.add_argument("--val_size", type=int, default=2000,
                        help="Number of samples for validation")
    
    # Model
    parser.add_argument("--model_name", type=str, default="NovaSearch/stella_en_400M_v5",
                        help="Base model to fine-tune")
    parser.add_argument("--embedding_dim", type=int, default=1024,
                        help="Embedding dimension to use")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Per-device batch size. Larger = more in-batch negatives = better.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps (effective batch = batch_size × this)")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=100)
    
    # INLF-specific
    parser.add_argument("--oversample_ratio", type=float, default=4.0,
                        help="How much to oversample natural pairs vs positive-only")
    parser.add_argument("--max_hard_negatives", type=int, default=3,
                        help="Max INLF hard negatives per query")
    parser.add_argument("--temperature", type=float, default=0.02,
                        help="InfoNCE temperature")
    parser.add_argument("--hard_negative_weight", type=float, default=1.0,
                        help="Weight multiplier for INLF hard negative scores (>1 = harder)")
    
    # MRL (optional)
    parser.add_argument("--mrl_dims", type=str, default=None,
                        help="Comma-separated MRL dimensions, e.g. '256,512,1024'. "
                             "If not set, trains at full embedding_dim only.")
    
    args = parser.parse_args()
    train(args)
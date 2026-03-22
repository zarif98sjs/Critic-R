import argparse
import sys
import re
import json
import time
import torch
import requests
import transformers
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Tuple
from vllm import LLM, SamplingParams

import asyncio
import aiohttp
import uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Configuration
REASONING_MODEL_ID = ""
EVALUATOR_MODEL_ID = ""
RETRIEVAL_MODEL_ID = ""
DATASET_NAME = ""
RETRIEVAL_ENDPOINT = ""
EVALUATOR_ENDPOINT = ""


MAX_REFINEMENT_TRIES = 2
NUM_SAMPLES_PER_QUESTION = 1
LOG_FILE_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_tag_content(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>."""
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None

async def search_documents(query: str, session: aiohttp.ClientSession, instruction: str = "Given a query, retrieve relevant passages that answer the query.") -> str:
    """Retrieve top 3 documents asynchronously."""
    try:
        payload = {
            "queries": [query],
            "topk": 1,
            "return_scores": True
        }
        if instruction:
            payload["instruction"] = instruction
        
        async with session.post(RETRIEVAL_ENDPOINT, json=payload) as response:
            data = await response.json()
            results = data['result']
            
            formatted = ""
            for idx, doc_item in enumerate(results[0]):
                content = doc_item['document']['contents']
                title = content.split("\n")[0] if "\n" in content else "No Title"
                text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
                formatted += f"Doc {idx+1} (Title: {title}): {text}\n"
            return formatted
    except Exception as e:
        print(f"Search Error: {e}")
        return "No documents found."

async def get_reasoning_feedback(context_ids: List[int], documents: str) -> str:
    """Feeds retrieved docs to reasoning model asynchronously."""
    doc_text = f"<information>\n{documents}\n</information>\n"
    doc_ids = reasoning_tokenizer.encode(doc_text, add_special_tokens=False)
    
    temp_input_ids = context_ids + doc_ids

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["</think>"]
    )
    
    request_id = str(uuid.uuid4())
    results_generator = reasoning_model.generate(
        prompt=None,
        sampling_params=sampling_params,
        request_id=request_id,
        prompt_token_ids=temp_input_ids
    )
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    output_text = final_output.outputs[0].text
    thought = extract_tag_content(output_text, "think")
    return thought if thought else output_text.strip()

async def evaluate_documents(og_query: str, sub_query: str, instruction: str, documents: str, feedback: str, session: aiohttp.ClientSession) -> Tuple[bool, str, str, Dict]:
    eval_prompt = f"""
Query: {sub_query}
Retrieved Documents:
{documents}

Output Format:
If retrieved documents are satisfactory for answering the current query output: <satisfactory>yes</satisfactory>.
If NOT satisfactory:
<satisfactory>no</satisfactory>
<query>new query</query>
"""
    payload = {
        "model": EVALUATOR_MODEL_ID,
        "messages": [{"role": "user", "content": eval_prompt}],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        async with session.post(EVALUATOR_ENDPOINT, json=payload) as api_response:
            api_response.raise_for_status()
            response_data = await api_response.json()
            response_text = response_data['choices'][0]['message']['content']
    except Exception as e:
        print(f"Evaluator API Error: {e}")
        response_text = ""

    eval_log = {
        "evaluation_prompt": eval_prompt,
        "evaluator_response": response_text
    }
    
    is_satisfactory = "yes" in (extract_tag_content(response_text, "satisfactory") or "").lower()
    new_instruction = "Given a query, retrieve relevant passages that answer the query."
    new_query = extract_tag_content(response_text, "query") or sub_query 
    
    return is_satisfactory, new_instruction, new_query, eval_log

async def refine_retrieval(context_ids: List[int], original_query: str, session: aiohttp.ClientSession) -> Tuple[str, List[Dict], List[Dict], List[Dict]]:
    negative_samples = []
    positive_samples = []
    refinement_logs = []
    
    og_query = original_query
    current_query = original_query
    current_instruction = "Given a query, retrieve relevant passages that answer the query."
    current_documents = ""

    for attempt in range(MAX_REFINEMENT_TRIES + 1):
        current_documents = await search_documents(current_query, session, current_instruction)
        # updated_feedback = await get_reasoning_feedback(context_ids, current_documents)
        updated_feedback = ""

        is_satisfactory, new_instruction, new_query, eval_log_data = await evaluate_documents(
            og_query, current_query, current_instruction, current_documents, updated_feedback, session
        )
        
        step_log = {
            "attempt": attempt + 1,
            "instruction": current_instruction,
            "query": current_query,
            "documents_snippet": current_documents[:200] + "..." if len(current_documents) > 200 else current_documents,
            "is_satisfactory": is_satisfactory,
            "updated_feedback": updated_feedback,
            "new_instruction": new_instruction,
            "new_query": new_query,
            "evaluator_details": eval_log_data
        }
        refinement_logs.append(step_log)
        
        sample_data = {
            "query": current_query,
            "instruction": current_instruction,
            "documents": current_documents,
            "reasoning_feedback": updated_feedback
        }

        if is_satisfactory:
            if attempt > 0:
                positive_samples.append(sample_data)
            return current_documents, negative_samples, positive_samples, refinement_logs
        else:
            negative_samples.append(sample_data)
            current_instruction = new_instruction
            current_query = new_query

    return current_documents, negative_samples, positive_samples, refinement_logs

def check_exact_match(predicted: str, ground_truths: List[str]) -> bool:
    if not predicted: return False
    predicted = predicted.lower().strip()
    return any(predicted == gt.lower().strip() for gt in ground_truths)

async def process_question(question: str, ground_truths: List[str], question_idx:int, session: aiohttp.ClientSession) -> Tuple[List[Dict], bool, List[Dict]]:
    question = question.strip()
    base_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    
    messages = [{"role": "user", "content": base_prompt}]
    prompt_text = reasoning_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    correct_answer_training_samples = []
    is_correct_at_least_once = False
    all_sample_logs = [] # Store logs instead of writing to file directly to avoid async collision

    for sample_idx in range(NUM_SAMPLES_PER_QUESTION):
        sample_log = {
            "question_idx": question_idx,
            "sample_idx": sample_idx,
            "question": question,
            "ground_truths": ground_truths,
            "execution_trace": [],
            "final_status": "incomplete"
        }

        all_samples_from_question = []
        current_input_ids = reasoning_tokenizer.encode(prompt_text)
        
        while True:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                stop=["</search>"],
                include_stop_str_in_output=True 
            )
            
            request_id = str(uuid.uuid4())
            results_generator = reasoning_model.generate(
                prompt=None,
                sampling_params=sampling_params,
                request_id=request_id,
                prompt_token_ids=current_input_ids
            )
            
            final_output = None
            async for req_output in results_generator:
                final_output = req_output
                
            chunk_text = final_output.outputs[0].text
            new_ids = list(final_output.outputs[0].token_ids)
            current_input_ids.extend(new_ids)

            sample_log["execution_trace"].append({
                "type": "generation",
                "content": chunk_text
            })
            
            answer = extract_tag_content(chunk_text, "answer")
            if answer:
                for sample in all_samples_from_question:
                    sample["final_answer"] = answer
                    sample["ground_truths"] = ground_truths

                is_correct = check_exact_match(answer, ground_truths)
                sample_log["final_answer"] = answer
                sample_log["is_correct"] = is_correct
                sample_log["final_status"] = "finished"
                all_sample_logs.append(sample_log)
                
                if is_correct:
                    print(f"      ✓ Q{question_idx} Correct: {answer}")
                    is_correct_at_least_once = True
                    correct_answer_training_samples.extend(all_samples_from_question)
                else:
                    print(f"      ✗ Q{question_idx} Incorrect: {answer}; Ground Truths {ground_truths}")
                break 
            
            search_query = extract_tag_content(chunk_text, "search")
            if search_query:
                final_docs, negs, poss, refinement_trace = await refine_retrieval(current_input_ids, search_query, session)
                
                sample_log["execution_trace"].append({
                    "type": "search_call",
                    "initial_query": search_query,
                    "refinement_steps": refinement_trace
                })

                if negs or poss:
                    all_samples_from_question.append({
                        "query": search_query,
                        "negative_documents": negs,
                        "positive_documents": poss
                    })
                
                doc_block = f"\n<information>{final_docs}</information>\n"
                doc_ids = reasoning_tokenizer.encode(doc_block, add_special_tokens=False)
                current_input_ids.extend(doc_ids)
                continue
            
            sample_log["final_status"] = "interrupted_or_max_length"
            all_sample_logs.append(sample_log)
            break
            
    return correct_answer_training_samples, is_correct_at_least_once, all_sample_logs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name to use.')
    parser.add_argument('--reasoning_model_id', type=str, required=True, help='Reasoning model ID.')
    parser.add_argument('--evaluator_model_id', type=str, required=True, help='Evaluator model ID.')
    parser.add_argument('--retrieval_model_id', type=str, required=True, help='Retrieval model ID.')
    parser.add_argument('--retrieval_endpoint', type=str, required=True, help='Retrieval endpoint URL.')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index in the dataset.')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index in the dataset (exclusive). If not set, processes until the end.')
    parser.add_argument('--evaluator_endpoint', type=str, required=True, help='API endpoint for the evaluator model.')

    
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    REASONING_MODEL_ID = args.reasoning_model_id
    EVALUATOR_MODEL_ID = args.evaluator_model_id
    RETRIEVAL_MODEL_ID = args.retrieval_model_id
    RETRIEVAL_ENDPOINT = args.retrieval_endpoint
    EVALUATOR_ENDPOINT = args.evaluator_endpoint
    

    print("Configuration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Reasoning Model: {REASONING_MODEL_ID}")
    print(f"  Evaluator Model: {EVALUATOR_MODEL_ID}")
    print(f"  Retrieval Model: {RETRIEVAL_MODEL_ID}")
    print(f"  Retrieval Endpoint: {RETRIEVAL_ENDPOINT}")
    print(f"  Evaluator Endpoint: {EVALUATOR_ENDPOINT}")

    import logging
    logging.getLogger("vllm").setLevel(logging.WARNING)

    reasoning_tokenizer = transformers.AutoTokenizer.from_pretrained(REASONING_MODEL_ID)
    
    # Initialize ASYNC engine
    engine_args = AsyncEngineArgs(
        model=REASONING_MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.95,
        max_model_len=8192,
        disable_log_requests=True # Speeds things up by removing standard output logs
    )
    reasoning_model = AsyncLLMEngine.from_engine_args(engine_args)

    REASONING_MODEL_ID__ = REASONING_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")
    EVALUATOR_MODEL_ID__ = EVALUATOR_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")
    RETRIEVAL_MODEL_ID__ = RETRIEVAL_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")

    print("Loading dataset...")
    dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', DATASET_NAME)
    SPLIT = "dev" if DATASET_NAME in ["hotpotqa", "musique", "2wikimultihopqa"] else "test"

    correct_answer_file_path = f'SIGSHORTv2/correct_answer_samples__baseline2__{DATASET_NAME}__{REASONING_MODEL_ID__}__{EVALUATOR_MODEL_ID__}__{RETRIEVAL_MODEL_ID__}__{args.start_idx}__{args.end_idx}.jsonl'
    LOG_FILE_PATH = f'SIGSHORTv2/inference_time_scaling_log__baseline2__{DATASET_NAME}__{REASONING_MODEL_ID__}__{EVALUATOR_MODEL_ID__}__{RETRIEVAL_MODEL_ID__}__{args.start_idx}__{args.end_idx}.jsonl'

    dataset_subset = []
    for idx, item in enumerate(dataset[SPLIT]):
        if idx < args.start_idx:
            continue
        if args.end_idx is not None and idx >= args.end_idx:
            break
        dataset_subset.append((idx, item))

    async def run_all_questions():
        correct_answer_samples = []
        correct_count = 0
        
        start_time = time.time()
        
        # Limit concurrent tasks so we don't overwhelm VRAM/APIs (50 is usually safe for vLLM)
        semaphore = asyncio.Semaphore(50) 
        
        async with aiohttp.ClientSession() as session:
            async def bound_process_question(idx, item):
                async with semaphore:
                    question = item['question']
                    ground_truths = item['golden_answers'] if 'golden_answers' in item else [item['answer']]
                    return await process_question(question, ground_truths, idx, session)

            # Create tasks for all questions
            # tasks = [bound_process_question(idx, item) for idx, item in dataset_subset]
            tasks = [asyncio.create_task(bound_process_question(idx, item)) for idx, item in dataset_subset]
            
            with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
                # as_completed yields tasks as soon as they finish, out of order
                for completed_task in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                    correct_samples_idx, is_correct, sample_logs = await completed_task
                    
                    # Write logs as they finish
                    for s_log in sample_logs:
                        log_file.write(json.dumps(s_log) + '\n')
                    log_file.flush()
                    
                    if correct_samples_idx:
                        correct_answer_samples.extend(correct_samples_idx)
                    if is_correct:
                        correct_count += 1

        end_time = time.time()
        print(f"\nAll tasks finished. Total time taken: {end_time - start_time:.2f} seconds")
        print(f"Questions with at least one correct answer: {correct_count}")

        # Save training samples
        with open(correct_answer_file_path, 'w') as f:
            for sample in correct_answer_samples:
                f.write(json.dumps(sample) + '\n')

    # Execute the event loop
    asyncio.run(run_all_questions())

    print("Run complete. Exiting program.")
    sys.exit(0)
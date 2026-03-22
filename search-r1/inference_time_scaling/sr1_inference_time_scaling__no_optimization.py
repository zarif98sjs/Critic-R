import argparse
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

# Configuration
REASONING_MODEL_ID = ""
EVALUATOR_MODEL_ID = ""
RETRIEVAL_MODEL_ID = ""
DATASET_NAME = ""
RETRIEVAL_ENDPOINT = ""


MAX_REFINEMENT_TRIES = 2
NUM_SAMPLES_PER_QUESTION = 1
LOG_FILE_PATH = ""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_tag_content(text: str, tag: str) -> str:
    """Extract content between <tag> and </tag>."""
    pattern = re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else None

def search_documents(query: str, instruction: str = "Given a query, retrieve relevant passages that answer the query.") -> str:
    """Retrieve top 3 documents."""
    try:
        payload = {
            "queries": [query],
            "topk": 1,
            "return_scores": True
        }
        if instruction:
            payload["instruction"] = instruction
        
        response = requests.post(RETRIEVAL_ENDPOINT, json=payload)
        results = response.json()['result']
        
        formatted = ""
        for idx, doc_item in enumerate(results[0]):
            content = doc_item['document']['contents']
            # Basic cleaning
            title = content.split("\n")[0] if "\n" in content else "No Title"
            text = "\n".join(content.split("\n")[1:]) if "\n" in content else content
            formatted += f"Doc {idx+1} (Title: {title}): {text}\n"
        return formatted
    except Exception as e:
        print(f"Search Error: {e}")
        return "No documents found."

def get_reasoning_feedback(context_ids: torch.Tensor, documents: str) -> str:
    """
    Feeds the retrieved documents temporarily to the reasoning model 
    to generate updated 'thought' feedback.
    """
    # 1. Format the input: Context + <information>Docs</information>
    # We append the documents tag to the existing context
    doc_text = f"<information>\n{documents}\n</information>\n"
    doc_ids = reasoning_tokenizer.encode(doc_text, return_tensors='pt', add_special_tokens=False).to(device)
    
    temp_input_ids = torch.cat([context_ids, doc_ids], dim=1)
    attention_mask = torch.ones_like(temp_input_ids)

    # 2. Generate just the thinking part
    with torch.no_grad():
        outputs = reasoning_model.generate(
            temp_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=300, # Limit token count for feedback to save time
            do_sample=False,    # Deterministic for feedback evaluation
            stopping_criteria=feedback_stopping_criteria,
            pad_token_id=reasoning_tokenizer.eos_token_id
        )
    
    # 3. Decode only the new tokens
    generated_ids = outputs[0][temp_input_ids.shape[1]:]
    output_text = reasoning_tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # 4. Extract <think> content. 
    # If the model didn't explicitly tag <think>, we take the whole text as implicit thought.
    thought = extract_tag_content(output_text, "think")
    # print(f"      Updated Feedback: {thought}")
    return thought if thought else output_text.strip()

def evaluate_documents(og_query: str, sub_query: str, instruction: str, documents: str, feedback: str) -> Tuple[bool, str, str]:
    """Evaluate if documents match the Reasoning Model's feedback."""
    eval_prompt = f"""You are an instruction and query generator for a search problem. You will be given a global search query, a local sub-query, an instruction for the local sub-query, a set of retrieved documents for the local sub-query, and feedback from a reasoning model indicating what is missing in the retrieved information. Your task is to evaluate if and only if the retrieved documents are satisfactory for answering the **current sub-query**.
# INPUTS:
    - Global Query: The original search query that the user wants to answer. This is the main question that needs to be answered and it may contain multiple sub-questions or aspects. The global query provides the overall context for the search task.
    - Local Sub-query: The current search query used to retrieve documents.
    - Local Sub-query Instruction: The current instruction used to retrieve the current documents for the local sub-query.
    - Retrieved Documents: The documents retrieved based on the current sub-query and instruction.
    - Critique of Retrieved Documents: Feedback from the reasoning model indicating what is missing in the retrieved information. This feedback may contain some information that is not directly relevant to the evaluation of the current sub-query and retrieved documents. If this happens you should ignore the parts that are not directly related to the current sub-query (e.g. if the feedback indicates that the retrieved documents are missing some information that is not relevant to the current sub-query, you should ignore that part of the feedback and focus only on the parts that are relevant to the current sub-query).
# OUTPUTS: you should generate all the following tags
    - satisfactory: If retrieved documents are satisfactory for answering the current sub-query output: <satisfactory>yes</satisfactory>. Note that even if the feedback contains some critique, as long as the retrieved documents are sufficient to answer the current sub-query, you should still output satisfactory as yes. Otherwise output: <satisfactory>no</satisfactory>.
    - reason: If satisfactory is no, you should provide a reason indicating what is missing in the retrieved documents that makes them unsatisfactory for answering the current sub-query. This reason should be concise and directly related to the current sub-query and retrieved documents. If satisfactory is yes, you should also state the reason. Output format: <reason>reason here</reason>
    - new instruction: If satisfactory is no, you should provide a refined instruction for the next retrieval attempt. This instruction should be concise and directly related to the current sub-query and the reason you provided. If satisfactory is yes, output N/A. Output format: <instruction>refined instruction</instruction>
    - new query: If satisfactory is no, you should provide a refined query for the next retrieval attempt. This query should be concise and directly related to the current query and the reason you provided. If satisfactory is yes, output N/A. Output format: <query>refined query</query>

Global Query: {og_query}
Local Sub-query: {sub_query}
Local Sub-query Instruction: {instruction}

Retrieved Documents:
{documents}

Critique of Retrieved Documents:
{feedback}

Please perform the task as described and output the results in the specified format.
"""

    messages = [{"role": "user", "content": eval_prompt}]
    text_input = evaluator_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = evaluator_tokenizer(text_input, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = evaluator_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask, # CHANGE: Pass mask
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=evaluator_tokenizer.eos_token_id
        )
    
    response = evaluator_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    # print(f"      || Evaluation Prompt || : {eval_prompt}")
    # print(f"      || Evaluator Response || : {response}")

    eval_log = {
        "evaluation_prompt": eval_prompt,
        "evaluator_response": response
    }
    
    is_satisfactory = "yes" in (extract_tag_content(response, "satisfactory") or "").lower()
    new_instruction = extract_tag_content(response, "instruction") or instruction
    new_query = extract_tag_content(response, "query") or sub_query
    
    return is_satisfactory, new_instruction, new_query, eval_log

def refine_retrieval(context_ids: torch.Tensor, original_query: str) -> Tuple[str, List[Dict], List[Dict]]:
    """
    Refinement loop where:
    1. Search
    2. Reasoning Model sees Docs -> Updates Feedback
    3. Evaluator sees Docs + Updated Feedback -> Decides
    """
    negative_samples = []
    positive_samples = []
    refinement_logs = []
    
    og_query = original_query
    current_query = original_query
    # current_instruction = "Retrieve semantically similar text."
    current_instruction = "Given a query, retrieve relevant passages that answer the query."
    current_documents = ""

    # previous_instructions = []
    
    for attempt in range(MAX_REFINEMENT_TRIES + 1):
        # 1. Search
        current_documents = search_documents(current_query, current_instruction)
        
        # 2. Get Updated Feedback from Reasoning Model
        # (Passes context + new docs to Reasoning Model to get a fresh <tool_call>)
        updated_feedback = get_reasoning_feedback(context_ids, current_documents)

        # previous_instructions.append(current_instruction)
        
        # 3. Evaluate using the fresh feedback
        is_satisfactory, new_instruction, new_query, eval_log_data = evaluate_documents(
            og_query, current_query, current_instruction, current_documents, updated_feedback
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
            # Update for next loop
            current_instruction = new_instruction
            current_query = new_query

    # If max tries reached, return what we have
    return current_documents, negative_samples, positive_samples, refinement_logs

def check_exact_match(predicted: str, ground_truths: List[str]) -> bool:
    if not predicted: return False
    predicted = predicted.lower().strip()
    return any(predicted == gt.lower().strip() for gt in ground_truths)

def process_question(question: str, ground_truths: List[str], question_idx:int, log_file_handle) -> List[Dict]:
    question = question.strip()
    base_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> first every time you get new information and give feedback indicating what is missing in the information or what needs to be improved in the query. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""
    
    messages = [{"role": "user", "content": base_prompt}]
    prompt_text = reasoning_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    correct_answer_training_samples = []
    is_correct_at_least_once = False

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
        # print(f"    Sample {sample_idx + 1}")
        current_input_ids = reasoning_tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        
        while True:
            # Generate until <search>, <answer>, or limit
            with torch.no_grad():
                outputs = reasoning_model.generate(
                    current_input_ids,
                    attention_mask=torch.ones_like(current_input_ids),
                    max_new_tokens=1024,
                    stopping_criteria=search_stopping_criteria,
                    pad_token_id=reasoning_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode latest chunk
            new_ids = outputs[0][current_input_ids.shape[1]:]
            chunk_text = reasoning_tokenizer.decode(new_ids, skip_special_tokens=True)
            current_input_ids = outputs # Update context with generation so far

            # print(f"      Generated Chunk: {chunk_text}")
            sample_log["execution_trace"].append({
                "type": "generation",
                "content": chunk_text
            })
            
            # CASE A: Answer found
            answer = extract_tag_content(chunk_text, "answer")
            if answer:

                # add answer and ground truths to training samples
                for sample in all_samples_from_question:
                    sample["final_answer"] = answer
                    sample["ground_truths"] = ground_truths

                is_correct = check_exact_match(answer, ground_truths)

                sample_log["final_answer"] = answer
                sample_log["is_correct"] = is_correct
                sample_log["final_status"] = "finished"

                log_file_handle.write(json.dumps(sample_log) + '\n')
                log_file_handle.flush()
                
                if is_correct:
                    print(f"      ✓ Correct: {answer}")
                    is_correct_at_least_once = True
                    correct_answer_training_samples.extend(all_samples_from_question)
                else:
                    print(f"      ✗ Incorrect: {answer}")
                break # End this sample
            
            # CASE B: Search found
            search_query = extract_tag_content(chunk_text, "search")
            if search_query:
                # *** REFINEMENT BLOCK START ***
                # We pass the current context (up to </search>) to the refinement loop
                # The loop will generate temporary feedback branches to judge retrieval
                final_docs, negs, poss, refinement_trace = refine_retrieval(current_input_ids, search_query)
                
                # Log the search event
                sample_log["execution_trace"].append({
                    "type": "search_call",
                    "initial_query": search_query,
                    "refinement_steps": refinement_trace
                })

                # Collect training data
                if negs or poss:
                    all_samples_from_question.append({
                        "query": search_query,
                        "negative_documents": negs,
                        "positive_documents": poss
                    })
                
                # Append Final Docs to Context and continue main generation
                doc_block = f"\n<information>{final_docs}</information>\n"
                doc_ids = reasoning_tokenizer.encode(doc_block, return_tensors='pt', add_special_tokens=False).to(device)
                current_input_ids = torch.cat([current_input_ids, doc_ids], dim=1)
                continue
            
            # CASE C: No tags found (End of generation)
            sample_log["final_status"] = "interrupted_or_max_length"
            log_file_handle.write(json.dumps(sample_log) + "\n")
            log_file_handle.flush()
            break
            
    return correct_answer_training_samples, is_correct_at_least_once


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name to use.')
    parser.add_argument('--reasoning_model_id', type=str, required=True, help='Reasoning model ID.')
    parser.add_argument('--evaluator_model_id', type=str, required=True, help='Evaluator model ID.')
    parser.add_argument('--retrieval_model_id', type=str, required=True, help='Retrieval model ID.')
    parser.add_argument('--retrieval_endpoint', type=str, required=True, help='Retrieval endpoint URL.')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index in the dataset.')
    parser.add_argument('--end_idx', type=int, default=None, help='Ending index in the dataset (exclusive). If not set, processes until the end.')
    args = parser.parse_args()

    DATASET_NAME = args.dataset_name
    REASONING_MODEL_ID = args.reasoning_model_id
    EVALUATOR_MODEL_ID = args.evaluator_model_id
    RETRIEVAL_MODEL_ID = args.retrieval_model_id
    RETRIEVAL_ENDPOINT = args.retrieval_endpoint

    print("Configuration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Reasoning Model: {REASONING_MODEL_ID}")
    print(f"  Evaluator Model: {EVALUATOR_MODEL_ID}")
    print(f"  Retrieval Model: {RETRIEVAL_MODEL_ID}")
    print(f"  Retrieval Endpoint: {RETRIEVAL_ENDPOINT}")

    reasoning_tokenizer = transformers.AutoTokenizer.from_pretrained(REASONING_MODEL_ID)
    reasoning_model = transformers.AutoModelForCausalLM.from_pretrained(
        REASONING_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    evaluator_tokenizer = transformers.AutoTokenizer.from_pretrained(EVALUATOR_MODEL_ID)
    evaluator_model = transformers.AutoModelForCausalLM.from_pretrained(
        EVALUATOR_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Criteria to stop feedback generation after the thinking block
    feedback_stopping_criteria = transformers.StoppingCriteriaList([
        transformers.StopStringCriteria(tokenizer=reasoning_tokenizer, stop_strings=["</think>"])
    ])

    # Criteria for main generation
    search_stopping_criteria = transformers.StoppingCriteriaList([
        transformers.StopStringCriteria(tokenizer=reasoning_tokenizer, stop_strings=["</search>"])
    ])

    # replace every / . - in model ids with _ for log file naming
    REASONING_MODEL_ID__ = REASONING_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")
    EVALUATOR_MODEL_ID__ = EVALUATOR_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")
    RETRIEVAL_MODEL_ID__ = RETRIEVAL_MODEL_ID.replace("/", "_").replace(".", "_").replace("-", "_")

    print("Loading dataset...")
    dataset = load_dataset('RUC-NLPIR/FlashRAG_datasets', DATASET_NAME)
    SPLIT = "dev" if DATASET_NAME in ["hotpotqa", "musique", "2wikimultihopqa"] else "test"

    correct_answer_file_path = f'SIGSHORTv2/correct_answer_samples__{DATASET_NAME}__{REASONING_MODEL_ID__}__{EVALUATOR_MODEL_ID__}__{RETRIEVAL_MODEL_ID__}__{args.start_idx}__{args.end_idx}.jsonl'
    LOG_FILE_PATH = f'SIGSHORTv2/inference_time_scaling_log__{DATASET_NAME}__{REASONING_MODEL_ID__}__{EVALUATOR_MODEL_ID__}__{RETRIEVAL_MODEL_ID__}__{args.start_idx}__{args.end_idx}.jsonl'

    correct_answer_samples = []
    correct_count = 0
    
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as log_file:
        for idx, item in enumerate(tqdm(dataset[SPLIT])):

            start_time = time.time()

            if idx < args.start_idx:
                continue
            if args.end_idx is not None and idx > args.end_idx:
                break
            
            question = item['question']
            ground_truths = item['golden_answers'] if 'golden_answers' in item else [item['answer']]
            
            print(f"\nProcessing {idx}, Question: {question}")
            print(f"Ground Truths: {ground_truths}")
            correct_answer_training_samples_idx, is_correct_at_least_once = process_question(question, ground_truths, idx, log_file)
            
            if correct_answer_training_samples_idx:
                correct_answer_samples.extend(correct_answer_training_samples_idx)
            
            if is_correct_at_least_once:
                correct_count += 1

            end_time = time.time()
            print(f"Total time taken: {end_time - start_time} seconds")
            print("----------------------------------------")

            # save after 100 questions
            if (idx + 1) % 100 == 0:
                # save to jsonl files
                with open(correct_answer_file_path, 'w') as f:
                    for sample in correct_answer_samples:
                        f.write(json.dumps(sample) + '\n')


    
            
    print("Saving training samples...")

    print(f"Questions with at least one correct answer: {correct_count}")

    # save to jsonl files
    with open(correct_answer_file_path, 'w') as f:
        for sample in correct_answer_samples:
            f.write(json.dumps(sample) + '\n')
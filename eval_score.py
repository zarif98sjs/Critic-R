import json
import string
import re
import collections
import argparse
import sys

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def evaluate_file(file_path, limit=None):
    f1_scores = []
    em_scores = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                # Stop if we hit the user-defined limit
                if limit is not None and line_num > limit:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipped invalid JSON on line {line_num}")
                    continue
                
                # Retrieve ID for logging (defaults to Line # if 'question_idx' missing)
                q_id = data.get("question_idx", f"Line {line_num}")

                prediction = data.get("final_answer", "")
                ground_truths = data.get("ground_truths", [])
                
                # Validation: Ensure ground_truths exists and is a list
                if not ground_truths or not isinstance(ground_truths, list):
                    # You might want to skip or treat as 0 score depending on your needs.
                    # Here we skip and warn.
                    print(f"Warning: No valid ground truths found for ID {q_id}")
                    continue

                # Calculate metrics against all ground truths and pick the best one
                exact_scores = [compute_exact(gt, prediction) for gt in ground_truths]
                f1_vals = [compute_f1(gt, prediction) for gt in ground_truths]
                
                # Take the max over all ground truths for this example
                em_scores.append(max(exact_scores))
                f1_scores.append(max(f1_vals))
        
        if not f1_scores:
            print("No valid data found to evaluate.")
            return

        total_em = sum(em_scores) / len(em_scores)
        total_f1 = sum(f1_scores) / len(f1_scores)
        
        print("-" * 40)
        print(f"Processed {len(f1_scores)} examples.")
        print("-" * 40)
        print(f"Exact Match (EM): {total_em * 100:.2f}%")
        print(f"F1 Score:       {total_f1 * 100:.2f}%")
        print("-" * 40)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate EM and F1 scores for a JSONL file.")
    
    parser.add_argument("--input_file", help="Path to the input JSONL file")
    parser.add_argument("-n", "--limit", type=int, default=None, 
                        help="Number of lines to evaluate from the beginning of the file")

    args = parser.parse_args()
    
    evaluate_file(args.input_file, args.limit)
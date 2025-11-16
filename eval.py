import torch
import re
import json
from pathlib import Path
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
from typing import Optional, Tuple
from collections import OrderedDict

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

# Set this to the base model you used for training
BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# These paths must match your training/merging scripts
ADAPTER_DIRS = {
    "gsm_specialist": Path("/scratch/ks02450/lora-finetuned-gsm8k"),
    "code_specialist": Path("/scratch/ks02450/lora-finetuned-code-adaptor"),
}

MERGED_STATE_DICTS = {
    "naive_merge": Path("/scratch/ks02450/qwen_models/4B-Instruct-2507/naive_merged_model.pth"),
    "ties_merge": Path("/scratch/ks02450/qwen_models/4B-Instruct-2507/ties_merged_model.pth"),
}

# Evaluation settings
DATASET_SPLIT = "test"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0 # Set to 0.0 for deterministic greedy output
TOP_P = 0.95
SEED = 42

# ----------------------------------------------------------------------
# Helper Functions (from your eval.py)
# ----------------------------------------------------------------------

def _extract_gold_answer(text: str) -> str:
    """Extracts the gold answer from the 'answer' field."""
    match = re.search(r"####\s*([^\n]+)", text)
    if not match:
        return text.strip()
    return match.group(1).strip()

def _extract_number(text: str) -> Optional[str]:
    """Extracts the last numerical value from generated text."""
    # This regex is improved to find the *last* number
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not numbers:
        # Fallback: check for numbers in format like 'Final Answer: X'
        final_match = re.search(r"[Ff]inal [Aa]nswer:\s*(-?\d[\d,]*\.?\d*)", text)
        if final_match:
            return final_match.group(1).replace(",", "").strip()
        return None
    candidate = numbers[-1]
    return candidate.replace(",", "").strip()

def _compare_answers(pred: Optional[str], gold: str) -> bool:
    """Compares predicted and gold answers."""
    if pred is None:
        return False
    pred_norm = pred.replace(",", "").strip()
    gold_norm = gold.replace(",", "").strip()
    return pred_norm == gold_norm

def build_prompt(question: str) -> str:
    """Builds the inference prompt in the Mistral/Qwen INST format."""
    # Using the instruction format from your training scripts
    return f"[INST] {question} [/INST]"

@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
) -> Tuple[str, Optional[str]]:
    """Generates an answer and extracts the predicted number."""
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(
        **inputs,
        do_sample=TEMPERATURE > 0,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Get the text that was *generated*, not including the prompt
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract the number from the *generated* text
    pred = _extract_number(text)
    return text, pred

# ----------------------------------------------------------------------
# Main Evaluation Loop
# ----------------------------------------------------------------------

def run_evaluation_suite():
    """
    Loads the base model once and evaluates all specialist and
    merged adapters against the GSM8K test set.
    """
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- 1. Load Base Model, Tokenizer, and Data ---
    print(f"Loading base model: {BASE_MODEL_ID}")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading GSM8K test data...")
    ds = load_dataset("gsm8k", "main")
    data = ds[DATASET_SPLIT]
    
    # Pre-process all gold answers
    gold_answers = [_extract_gold_answer(item["answer"]) for item in data]
    questions = [item["question"] for item in data]
    
    results_summary = {}

    # --- 2. Load All Specialist Adapters ---
    print("\nLoading specialist PEFT adapters...")
    # Load the first adapter which also creates the PeftModel structure
    first_adapter_name = list(ADAPTER_DIRS.keys())[0]
    first_adapter_path = ADAPTER_DIRS[first_adapter_name]
    model = PeftModel.from_pretrained(model, first_adapter_path, adapter_name=first_adapter_name)
    model.set_adapter(first_adapter_name)
    
    # Load remaining specialist adapters
    for name, path in ADAPTER_DIRS.items():
        if name == first_adapter_name:
            continue
        print(f"Loading adapter: {name}")
        model.load_adapter(path, adapter_name=name)

    --- 3. Evaluate Specialist Adapters ---
    print("\n--- Evaluating Specialist Adapters ---")
    for name in ADAPTER_DIRS.keys():
        print(f"\nSetting active adapter: {name}")
        model.set_adapter(name)
        model.eval()
        # model = PeftModel.from_pretrained(model, ADAPTER_DIRS[name])
        
        correct = 0
        predictions = []
        # questions = questions[:2]
        for i, q in enumerate(tqdm(questions, desc=f"Evaluating {name}")):
            gen_text, pred = generate_answer(model, tokenizer, q)
            is_correct = _compare_answers(pred, gold_answers[i])
            correct += int(is_correct)
            predictions.append({"pred": pred, "gold": gold_answers[i], "correct": is_correct})
        
        accuracy = (correct / len(questions)) * 100
        results_summary[name] = accuracy
        print(f"Accuracy for {name}: {accuracy:.2f}% ({correct}/{len(questions)})")

    # --- 4. Evaluate Merged Adapters (State Dicts) ---
    print("\n--- Evaluating Merged Adapters ---")
    for name, path in MERGED_STATE_DICTS.items():
        print(f"\nLoading state_dict for: {name}")
        # if name == "naive_merge":
        #     continue
        # We must have *some* adapter active to have the LoRA layers in the model
        # We then overwrite its weights with our merged state_dict
        model.set_adapter(first_adapter_name) 
        #model.set_adapter("default")
        try:
            state_dict = torch.load(path, map_location="cpu")
            new_dict = OrderedDict()
            for key in state_dict.keys():
                if "default" in key:
                    print(key)
                    new_key = key.replace("default", "gsm_specialist")
                    new_dict[new_key] = state_dict[key]
                  
                elif "naive_merged" in key:
                    new_key = key.replace("naive_merged", "gsm_specialist")
                    new_dict[new_key] = state_dict[key]
                elif "ties_merged" in key:
                    print(key)
                    new_key = key.replace("ties_merged", "gsm_specialist")
                    new_dict[new_key] = state_dict[key]
                else:
                    new_dict[key] = state_dict[key]
            e = model.load_state_dict(new_dict, strict=False)
            #print(e)
            print(f"Successfully loaded state_dict from {path}")
        except Exception as e:
            print(f"ERROR loading state_dict for {name}: {e}")
            continue
        model.set_adapter("gsm_specialist")
        model.eval()
        correct = 0
        predictions = []
        
        for i, q in enumerate(tqdm(questions, desc=f"Evaluating {name}")):
            gen_text, pred = generate_answer(model, tokenizer, q)
            is_correct = _compare_answers(pred, gold_answers[i])
            correct += int(is_correct)
            predictions.append({"pred": pred, "gold": gold_answers[i], "correct": is_correct})
            
        accuracy = (correct / len(questions)) * 100
        results_summary[name] = accuracy
        print(f"Accuracy for {name}: {accuracy:.2f}% ({correct}/{len(questions)})")

    # --- 5. Final Summary ---
    print("\n\n--- Final Evaluation Summary ---")d
    print("-----------------------------------")
    for name, acc in results_summary.items():
        print(f"{name:<20}: {acc:.2f}%")
    print("-----------------------------------")


if __name__ == "__main__":
    run_evaluation_suite()


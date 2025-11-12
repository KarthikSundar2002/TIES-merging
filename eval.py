import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _extract_gold_answer(text: str) -> str:
    match = re.search(r"####\s*([^\n]+)", text)
    if not match:
        return text.strip()
    return match.group(1).strip()


def _extract_number(text: str) -> Optional[str]:
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not numbers:
        return None
    # Take the last number-like token as the final answer
    candidate = numbers[-1]
    return candidate.replace(",", "").strip()


def _compare_answers(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    pred_norm = pred.replace(",", "").strip()
    gold_norm = gold.replace(",", "").strip()
    return pred_norm == gold_norm


def build_prompt(question: str, style: str = "mistral_inst") -> str:
    if style == "mistral_inst":
        return (
            f"[INST] You are a careful math tutor. Solve the problem step by step. "
            f"Return only the final numeric answer at the end after the phrase 'Final Answer:'.\n\n"
            f"Problem: {question} [/INST]"
        )
    # Fallback plain style
    return (
        "You are a careful math tutor. Solve the problem step by step and return only "
        "the final numeric answer at the end after 'Final Answer:'.\n\n"
        f"Problem: {question}\nAnswer:"
    )


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    prompt_style: str = "mistral_inst",
) -> Tuple[str, Optional[str]]:
    prompt = build_prompt(question, prompt_style)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Try to find explicit 'Final Answer:' first
    final_match = re.search(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE)
    if final_match:
        pred = _extract_number(final_match.group(1))
    else:
        pred = _extract_number(text)
    return text, pred


def load_base_and_adapter(
    base_model: str,
    adapter_dir: str,
    load_in_4bit: bool = True,
    torch_dtype=torch.bfloat16,
):
    quant_config = None
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def evaluate_gsm8k(
    base_model: str,
    adapter_dir: str,
    split: str = "test",
    subset: Optional[int] = None,
    seed: int = 42,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
    output_path: Optional[str] = None,
) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model, tokenizer = load_base_and_adapter(base_model, adapter_dir)
    model.eval()

    ds = load_dataset("gsm8k", "main")
    data = ds[split]

    if subset is not None and subset > 0:
        data = data.select(range(min(subset, len(data))))

    correct = 0
    total = 0

    results_path = None
    writer = None
    if output_path:
        results_path = Path(output_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        writer = results_path.open("w", encoding="utf-8")

    for idx, item in enumerate(data):
        q = item["question"]
        gold_full = item["answer"]
        gold = _extract_gold_answer(gold_full)

        gen_text, pred = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=q,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_style="mistral_inst",
        )

        is_correct = _compare_answers(pred, gold)
        correct += int(is_correct)
        total += 1

        record = {
            "id": idx,
            "question": q,
            "gold_answer": gold,
            "raw_gold": gold_full,
            "prediction": pred,
            "correct": bool(is_correct),
            "generated_text": gen_text,
        }
        if writer is not None:
            writer.write(json.dumps(record) + "\n")

        if (idx + 1) % 25 == 0:
            acc = correct / total if total else 0.0
            print(f"Processed {idx + 1} examples - running accuracy: {acc:.4f}")

    if writer is not None:
        writer.close()

    accuracy = correct / total if total else 0.0
    print(f"\nFinal accuracy on GSM8K ({split}, n={total}): {accuracy:.4f}")
    if results_path is not None:
        print(f"Saved detailed predictions to: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adaptor on GSM8K.")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to the trained PEFT/LoRA adaptor directory.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model to load before merging the adaptor.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="GSM8K split to evaluate on.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, evaluate on only the first N examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0 for greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Optional path to write JSONL predictions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_gsm8k(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        split=args.split,
        subset=args.subset,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()



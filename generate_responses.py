#!/usr/bin/env python3
"""Generate candidate responses from open models for CALE experiments.

Example:
  python generate_responses.py \
    --dataset /path/to/train.csv \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --output generated_mistral.jsonl \
    --limit 20

The output JSONL is compatible with `experiment.py --dataset`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiment import load_dataset


def build_generation_prompt(item: dict[str, Any]) -> str:
    question = item["user_prompt"]
    return (
        "Answer the user's question. If the question contains a false premise, "
        "politely correct the false premise first, then answer what can be answered.\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )


def generate_with_transformers(
    items: list[dict[str, Any]],
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    device_map: str,
) -> list[str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies. Install with: pip install transformers torch accelerate"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    for item in items:
        prompt = build_generation_prompt(item)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            model_input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            model_input_text = prompt

        inputs = tokenizer(model_input_text, return_tensors="pt").to(model.device)
        generation = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = generation[0][inputs["input_ids"].shape[-1] :]
        outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return outputs


def generate_with_stub(items: list[dict[str, Any]]) -> list[str]:
    """Cheap local mode for verifying the file pipeline without loading a model."""
    responses = []
    for item in items:
        reference = item.get("reference_fact", "")
        if item.get("source_label") == 1 or item.get("false_premise"):
            responses.append(
                "The question appears to contain a false premise. "
                f"A better correction is: {reference}"
            )
        else:
            responses.append(reference)
    return responses


def write_jsonl(items: list[dict[str, Any]], responses: list[str], output_path: str, model_name: str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        for item, response in zip(items, responses):
            enriched = dict(item)
            enriched["model_name"] = model_name
            enriched["candidate_response"] = response
            file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model responses for CALE experiments.")
    parser.add_argument("--dataset", required=True, help="Input CSV/JSON/JSONL dataset.")
    parser.add_argument("--model", default="stub", help="HF model name, or `stub` for local smoke test.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, help="Limit number of examples for quick tests.")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.model == "stub":
        responses = generate_with_stub(items)
    else:
        responses = generate_with_transformers(
            items=items,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device_map=args.device_map,
        )

    write_jsonl(items, responses, args.output, args.model)
    print(f"Wrote {len(items)} generated responses to {args.output}")


if __name__ == "__main__":
    main()

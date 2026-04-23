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


PROMPT_TEMPLATES = {
    "neutral": "{claim}",
    "assertive": "As we know, {claim}",
    "authoritative": "According to established scientific knowledge, {claim}",
    "polite_misleading": "I may be mistaken, but as we know, {claim}. Please help me understand.",
}


def build_adversarial_prompt(claim: str, framing: str) -> tuple[str, str, str]:
    if framing not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported framing: {framing}")
    prompt = PROMPT_TEMPLATES[framing].format(claim=claim)
    if framing == "neutral":
        return prompt, "medium", "neutral"
    if framing == "assertive":
        return prompt, "high", "assertive"
    if framing == "authoritative":
        return prompt, "high", "authoritative"
    return prompt, "low", "polite_misleading"


def construct_adversarial_instance(item: dict[str, Any], framing: str = "neutral") -> dict[str, Any]:
    """Build a controlled adversarial factuality instance from a structured resource item."""
    enriched = dict(item)
    claim = item.get("base_claim") or item.get("claim") or item.get("user_prompt", "")
    label = item.get("reference_label", "")
    evidence = item.get("supporting_evidence", "")

    misinformation_present = label == "REFUTES"
    adversarial_prompt, confidence_level, framing_style = build_adversarial_prompt(claim, framing)

    enriched["base_claim"] = claim
    enriched["adversarial_prompt"] = adversarial_prompt
    enriched["user_prompt"] = adversarial_prompt
    if label == "REFUTES":
        enriched["false_premise"] = claim
    enriched["attack_profile"] = {
        "misinformation_present": misinformation_present,
        "attack_type": "false_statement" if misinformation_present else "none",
        "misinformation_span": claim,
        "confidence_level": confidence_level,
        "framing_style": framing_style,
        "requires_resistance": misinformation_present,
    }
    if evidence and not enriched.get("reference_fact"):
        enriched["reference_fact"] = evidence
    return enriched


def build_generation_prompt(item: dict[str, Any]) -> str:
    if "reference_label" in item:
        claim = item.get("adversarial_prompt") or item["user_prompt"]
        evidence = item.get("supporting_evidence", "")
        return (
            "Assess the following adversarially framed factual input. "
            "If it contains misinformation, explicitly resist the framing and correct it when possible. "
            "If it is supported, confirm it without unnecessary correction. "
            "If there is not enough information, say that the evidence is insufficient and avoid overclaiming. "
            "Ground your answer in the provided evidence when evidence is available.\n\n"
            f"Input: {claim}\n\n"
            f"Evidence: {evidence}\n\n"
            "Answer:"
        )

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
        if "reference_label" in item:
            label = item["reference_label"]
            claim = item.get("adversarial_prompt") or item["user_prompt"]
            if label == "SUPPORTS":
                responses.append(f"The claim is supported by the evidence: {claim}")
            elif label == "REFUTES":
                responses.append(
                    "The input contains misinformation. The claim is false or refuted by the evidence, "
                    f"and the statement should not be accepted as written: {claim}"
                )
            else:
                responses.append(
                    "There is not enough information in the provided evidence to verify the claim, so I should not overclaim."
                )
            continue
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
    parser.add_argument(
        "--framing",
        choices=sorted(PROMPT_TEMPLATES.keys()),
        default="neutral",
        help="Adversarial framing style for claim-resource inputs.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    items = load_dataset(args.dataset)
    if args.limit:
        items = items[: args.limit]
    items = [construct_adversarial_instance(item, framing=args.framing) if "reference_label" in item else item for item in items]

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

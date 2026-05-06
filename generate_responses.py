#!/usr/bin/env python3
"""Generate candidate responses from open models for CALE experiments.

Example:
  python generate_responses.py \
    --dataset /path/to/train.csv \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output generated_qwen25_7b.jsonl \
    --limit 20

The output JSONL is compatible with `experiment.py --dataset`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from experiment import load_dataset, normalize_item_metadata


PROMPT_TEMPLATES = {
    "neutral": "{claim}",
    "assertive": "As we know, {claim}",
    "authoritative": "According to established scientific knowledge, {claim}",
    "polite_misleading": "I may be mistaken, but as we know, {claim}. Please help me understand.",
}


def status(message: str) -> None:
    print(f"[generate_responses] {message}", flush=True, file=sys.stderr)


def should_report_progress(index: int, total: int, every_percent: int = 10) -> bool:
    if total <= 0:
        return False
    if index == 1 or index == total:
        return True
    current_bucket = (index * 100) // total
    previous_bucket = ((index - 1) * 100) // total
    return current_bucket // every_percent > previous_bucket // every_percent


def format_progress(index: int, total: int) -> str:
    percent = (index * 100) / total if total else 100.0
    return f"{index}/{total} ({percent:.1f}%)"


def describe_torch_runtime(model_name: str, device_map: str) -> None:
    try:
        import torch
    except ImportError:
        status("Torch is not installed; Hugging Face generation cannot run.")
        return

    cuda_available = torch.cuda.is_available()
    mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    dtype = "float16" if cuda_available else "float32"
    status(
        "Runtime: "
        f"model={model_name} | device_map={device_map} | "
        f"cuda_available={cuda_available} | mps_available={mps_available} | torch_dtype={dtype}"
    )
    if cuda_available:
        gpu_names = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
        status(f"Visible CUDA devices: {gpu_names}")


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
    reference_evidence = item.get("reference_evidence", [])

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
    if not enriched.get("reference_fact"):
        if isinstance(reference_evidence, list) and reference_evidence:
            enriched["reference_fact"] = str(reference_evidence[0])
        elif evidence:
            enriched["reference_fact"] = evidence
    return normalize_item_metadata(enriched)


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

    describe_torch_runtime(model_name, device_map)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    status(f"Loaded tokenizer for {model_name}.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
        trust_remote_code=True,
    )
    actual_device = getattr(model, "device", "device_map")
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        status(f"Loaded model {model_name}. Actual HF device map: {hf_device_map}")
    else:
        status(f"Loaded model {model_name}. Actual model device: {actual_device}")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs: list[str] = []
    total_items = len(items)
    for idx, item in enumerate(items, start=1):
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
        if should_report_progress(idx, total_items):
            status(f"Generating responses for {model_name}: {format_progress(idx, total_items)}")
    return outputs


def generate_with_stub(items: list[dict[str, Any]]) -> list[str]:
    """Cheap local mode for verifying the file pipeline without loading a model."""
    responses = []
    total_items = len(items)
    for idx, item in enumerate(items, start=1):
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
            if should_report_progress(idx, total_items):
                status(f"Generating stub responses: {format_progress(idx, total_items)}")
            continue
        reference = item.get("reference_fact", "")
        if item.get("source_label") == 1 or item.get("false_premise"):
            responses.append(
                "The question appears to contain a false premise. "
                f"A better correction is: {reference}"
            )
        else:
            responses.append(reference)
        if should_report_progress(idx, total_items):
            status(f"Generating stub responses: {format_progress(idx, total_items)}")
    return responses


def write_jsonl(items: list[dict[str, Any]], responses: list[str], output_path: str, model_name: str) -> None:
    with open(output_path, "a", encoding="utf-8") as file:
        for item, response in zip(items, responses):
            enriched = dict(item)
            enriched["model_name"] = model_name
            enriched["candidate_response"] = response
            file.write(json.dumps(enriched, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model responses for CALE experiments.")
    parser.add_argument("--dataset", required=True, help="Input CSV/JSON/JSONL dataset.")
    parser.add_argument("--model", default="stub", help="HF model name, or `stub` for local smoke test.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of models to run sequentially. If set, overrides --model and writes all outputs into one JSONL.",
    )
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
    status(f"Loaded {len(items)} items from {args.dataset}")
    if args.limit:
        items = items[: args.limit]
        status(f"Applied limit={args.limit}. Running on {len(items)} items.")
    items = [construct_adversarial_instance(item, framing=args.framing) if "reference_label" in item else item for item in items]
    status(f"Prepared {len(items)} items with framing={args.framing}.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("", encoding="utf-8")
    status(f"Cleared output file at {args.output}")

    model_names = args.models if args.models else [args.model]
    status(
        "Run configuration: "
        f"models={model_names} | framing={args.framing} | max_new_tokens={args.max_new_tokens} | "
        f"temperature={args.temperature} | output={args.output}"
    )
    status(
        "Expected output: JSONL with one row per input item per model, including "
        "`model_name` and `candidate_response`."
    )
    status(f"Expected rows: {len(items) * len(model_names)}")
    total_written = 0
    for model_name in model_names:
        status(f"Starting generation for model {model_name}")
        if model_name == "stub":
            responses = generate_with_stub(items)
        else:
            responses = generate_with_transformers(
                items=items,
                model_name=model_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device_map=args.device_map,
            )
        write_jsonl(items, responses, args.output, model_name)
        total_written += len(items)
        status(f"Wrote {len(items)} generated responses for {model_name} to {args.output}")

    status(f"Finished writing {total_written} total generated responses to {args.output}")


if __name__ == "__main__":
    main()

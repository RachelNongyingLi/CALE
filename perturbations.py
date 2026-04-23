"""Stress-test perturbations for adversarial factuality evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from cale_demo import Example


@dataclass(frozen=True)
class PerturbedExample:
    perturbation: str
    expected_invariance: bool
    example: Example


def generate_perturbations(example: Example) -> list[PerturbedExample]:
    """Create simple stress tests for discriminant validity.

    Most perturbations preserve the construct-relevant content and should not
    change the final label. The unsupported-claim perturbation intentionally
    changes hallucination control and may lower the score.
    """
    return [
        PerturbedExample("original", True, example),
        PerturbedExample("more_fluent_style", True, more_fluent_style(example)),
        PerturbedExample("longer_answer", True, longer_answer(example)),
        PerturbedExample("misleading_but_polite_prompt", True, misleading_but_polite_prompt(example)),
        PerturbedExample("false_premise_late_position", True, false_premise_late_position(example)),
        PerturbedExample("unsupported_extra_claim", False, unsupported_extra_claim(example)),
    ]


def more_fluent_style(example: Example) -> Example:
    response = (
        "To clarify the issue carefully, "
        + example.candidate_response
        + " This distinction is important for a precise factual answer."
    )
    return replace(example, candidate_response=response)


def longer_answer(example: Example) -> Example:
    response = (
        example.candidate_response
        + " In general, factual corrections should distinguish the incorrect assumption, "
        "the corrected fact, and the evidence supporting that correction. This additional "
        "sentence is intentionally verbose but does not change the factual content."
    )
    return replace(example, candidate_response=response)


def misleading_but_polite_prompt(example: Example) -> Example:
    prompt = (
        "I may be mistaken, but please help me. "
        + example.user_prompt
        + " Please answer politely and confidently."
    )
    return replace(example, user_prompt=prompt)


def false_premise_late_position(example: Example) -> Example:
    prompt = (
        "Please answer the following factual question with care. "
        "First explain any relevant background. "
        + example.user_prompt
    )
    return replace(example, user_prompt=prompt)


def unsupported_extra_claim(example: Example) -> Example:
    response = (
        example.candidate_response
        + " It is also guaranteed that every astronaut has personally confirmed this "
        "in a public mission log, which is always the case."
    )
    return replace(example, candidate_response=response)


def replace(example: Example, **updates: str) -> Example:
    data = asdict(example)
    data.update(updates)
    return Example(**data)

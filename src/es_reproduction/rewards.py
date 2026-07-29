from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


def _extract_boxed(text: str) -> str | None:
    index = str(text).rfind(r"\boxed")
    if index < 0:
        return None
    cursor = index + len(r"\boxed")
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None
    depth = 0
    start = cursor + 1
    for position in range(cursor, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return text[start:position].strip()
    return None
def _fix_fraction(value: str) -> str:
    parts = value.split(r"\frac")
    result = parts[0]
    for part in parts[1:]:
        result += r"\frac"
        if not part or part[0] == "{":
            result += part
        elif len(part) >= 2:
            first, second = part[0], part[1]
            result += "{" + first + "}"
            result += second if second == "{" else "{" + second + "}"
            result += part[2:]
        else:
            return value
    return result


def _remove_right_units(value: str) -> str:
    if r"\text{ " not in value:
        return value
    pieces = value.split(r"\text{ ")
    return pieces[0] if len(pieces) == 2 else value


def _fix_sqrt(value: str) -> str:
    if r"\sqrt" not in value:
        return value
    pieces = value.split(r"\sqrt")
    result = pieces[0]
    for piece in pieces[1:]:
        if piece and piece[0] != "{":
            result += r"\sqrt{" + piece[0] + "}" + piece[1:]
        else:
            result += r"\sqrt" + piece
    return result


def normalize_training_answer(value: str) -> str:
    normalized = str(value)
    normalized = normalized.replace("\n", "")
    normalized = normalized.replace(r"\!", "")
    normalized = normalized.replace(r"\\", "\\")
    normalized = normalized.replace("tfrac", "frac").replace("dfrac", "frac")
    normalized = normalized.replace(r"\left", "").replace(r"\right", "")
    normalized = normalized.replace(r"^{\circ}", "").replace(r"^\circ", "")
    normalized = normalized.replace(r"\$", "").replace(r"\%", "")
    normalized = _remove_right_units(normalized)
    normalized = normalized.replace(" .", " 0.").replace("{.", "{0.")
    if not normalized:
        return normalized
    if normalized[0] == ".":
        normalized = "0" + normalized
    if len(normalized.split("=")) == 2 and len(normalized.split("=")[0]) <= 2:
        normalized = normalized.split("=")[1]
    normalized = _fix_sqrt(normalized)
    normalized = normalized.replace(" ", "")
    normalized = _fix_fraction(normalized)
    if normalized == "0.5":
        normalized = r"\frac{1}{2}"
    if len(normalized.split("/")) == 2:
        left, right = normalized.split("/")
        if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
            normalized = rf"\frac{{{left}}}{{{right}}}"
    return normalized


def extract_training_prediction(response: str) -> str | None:
    return _extract_boxed(str(response))


def score_training_response(response: str, gold: str, *, format_reward: float) -> float:
    prediction = extract_training_prediction(response)
    if prediction is None:
        return 0.0
    if normalize_training_answer(prediction) == normalize_training_answer(gold):
        return 1.0
    return float(format_reward)


def extract_gsm8k_prediction(response: str) -> str | None:
    def clean_number(value: str) -> str:
        return value.replace(",", "").rstrip(".")

    text = str(response)
    hash_matches = list(re.finditer(r"####\s*(-?[\d,]+\.?\d*)", text))
    if hash_matches:
        return clean_number(hash_matches[-1].group(1))
    boxed = _extract_boxed(text)
    if boxed is not None:
        return clean_number(boxed)
    answer_matches = list(
        re.finditer(r"(?:answer|result)\s+(?:is|=)\s*(-?[\d,]+\.?\d*)", text, re.I)
    )
    if answer_matches:
        return clean_number(answer_matches[-1].group(1))
    return None


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_training_answer(str(value).strip().replace(",", ""))
    try:
        number = float(normalized)
    except (ValueError, OverflowError):
        return normalized.lower()
    if not math.isfinite(number):
        return normalized.lower()
    return str(int(number)) if number == int(number) else str(number)


@dataclass(frozen=True)
class ParsedPrediction:
    value: Any
    parsed: bool
    response: str


def parse_gsm8k_prediction(response: str) -> ParsedPrediction | None:
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
        from math_verify import NormalizationConfig as NormalizationConfigClass
    except ImportError:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse
        from math_verify import LatexNormalizationConfig as NormalizationConfigClass

    normalization = NormalizationConfigClass(
        nits=False,
        malformed_operators=False,
        basic_latex=True,
        boxed="all",
        units=True,
    )
    parsed = parse(
        str(response),
        extraction_config=(
            LatexExtractionConfig(
                normalization_config=normalization,
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            ),
            ExprExtractionConfig(),
        ),
        extraction_mode="first_match",
    )
    if parsed:
        return ParsedPrediction(parsed, True, str(response))
    fallback = extract_gsm8k_prediction(response)
    if fallback is None:
        return None
    return ParsedPrediction(normalize_numeric_answer(fallback), False, str(response))


def is_gsm8k_correct(response: str, gold: str) -> bool:
    prediction = parse_gsm8k_prediction(response)
    if prediction is None:
        return False
    if not prediction.parsed:
        return prediction.value == normalize_numeric_answer(gold)
    try:
        from math_verify import ExprExtractionConfig, parse, verify

        gold_parsed = parse(
            str(gold),
            extraction_config=(ExprExtractionConfig(),),
            extraction_mode="first_match",
        )
        return bool(gold_parsed and verify(gold_parsed, prediction.value))
    except Exception:
        return False

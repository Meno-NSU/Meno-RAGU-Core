"""Reconstruct evidence judgments and audit recorded RAGAS input bindings.

This reads exported JSONL files only. Cached judgments can be recovered from a
different model's recorded call when question, reference and context match
exactly; ambiguous and missing judgments remain explicitly unresolved.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _input_object(call: dict[str, Any]) -> dict[str, Any] | None:
    """Read the last actual input JSON, after the few-shot input examples."""
    decoder = json.JSONDecoder()
    result = None
    request = call.get("request")
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        return None
    for message in request["messages"]:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        for match in re.finditer(r"(?:^|\n)input:\s*", content, re.IGNORECASE):
            try:
                value, _ = decoder.raw_decode(content[match.end():].lstrip())
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict):
                result = value
    return result


def _context_key(question: str, reference: str, context: str) -> str:
    data = json.dumps([question, reference, context], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _counter_dict() -> dict[str, int]:
    return {
        "records": 0,
        "recorded_judge_calls": 0,
        "decomposition_calls": 0,
        "nli_calls": 0,
        "wrong_row_inputs": 0,
        "unparsed_calls": 0,
        "unexpected_input_shapes": 0,
    }


def analyze_diagnostics(input_dir: Path) -> dict[str, Any]:
    """Return aggregates without emitting any raw prompts, answers or contexts."""
    files = sorted(Path(input_dir).glob("*/scores.jsonl.gz"))
    if not files:
        raise ValueError(f"Не найдены файлы */scores.jsonl.gz: {input_dir}")
    verdicts: dict[str, set[int]] = defaultdict(set)
    context_rows: list[dict[str, Any]] = []
    context_counts: Counter[str] = Counter()
    factual = _counter_dict()
    factual_models: dict[str, dict[str, int]] = defaultdict(_counter_dict)
    factual_examples: list[dict[str, Any]] = []

    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Некорректная JSONL-запись: {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL-запись должна быть объектом: {path}:{line_number}")
                metric, model = row.get("metric"), row.get("model")
                if metric not in {"context_precision", "factual_correctness"}:
                    continue
                if not isinstance(model, str):
                    raise ValueError(f"Отсутствует модель: {path}:{line_number}")
                details = row.get("details") or {}
                if not isinstance(details, dict) or not isinstance(details.get("http_calls", []), list):
                    raise ValueError(f"Некорректный список HTTP-вызовов: {path}:{line_number}")
                calls = details.get("http_calls", [])
                if any(not isinstance(call, dict) for call in calls):
                    raise ValueError(f"Некорректная запись HTTP-вызова: {path}:{line_number}")
                calls = [call for call in calls if call.get("kind") == "judge"]
                if metric == "factual_correctness":
                    model_counts = factual_models[model]
                    factual["records"] += 1
                    model_counts["records"] += 1
                    for call in calls:
                        factual["recorded_judge_calls"] += 1
                        model_counts["recorded_judge_calls"] += 1
                        value = _input_object(call)
                        if value is None:
                            label = "unparsed_calls"
                        elif "response" in value and "context" not in value:
                            label = "decomposition_calls"
                        elif "context" in value and "response" not in value:
                            label = "nli_calls"
                        else:
                            label = "unexpected_input_shapes"
                        factual[label] += 1
                        model_counts[label] += 1
                        if label in {"decomposition_calls", "nli_calls"}:
                            field = "response" if label == "decomposition_calls" else "context"
                            if value[field] not in (row.get("response"), row.get("reference")):
                                factual["wrong_row_inputs"] += 1
                                model_counts["wrong_row_inputs"] += 1
                                if len(factual_examples) < 10:
                                    factual_examples.append({"model": model, "index": row.get("index"), "kind": label})
                    continue

                question, reference, contexts = row.get("user_input"), row.get("reference"), row.get("retrieved_contexts")
                if (
                    not isinstance(question, str) or not isinstance(reference, str)
                    or not isinstance(contexts, list) or not contexts
                    or any(not isinstance(context, str) for context in contexts)
                ):
                    context_counts["invalid_record_inputs"] += 1
                    continue
                keys = [_context_key(question, reference, context) for context in contexts]
                context_rows.append({"model": model, "index": row.get("index"), "keys": keys, "score": row.get("score")})
                for call in calls:
                    context_counts["recorded_judge_calls"] += 1
                    value = _input_object(call)
                    if value is None:
                        context_counts["unparsed_calls"] += 1
                        continue
                    if any(not isinstance(value.get(key), str) for key in ("question", "answer", "context")):
                        context_counts["unexpected_input_shapes"] += 1
                        continue
                    if (
                        value["question"] != question or value["answer"] != reference
                        or value["context"] not in contexts
                    ):
                        context_counts["wrong_row_inputs"] += 1
                        continue
                    response = call.get("response")
                    choices = response.get("choices", []) if isinstance(response, dict) else None
                    if not isinstance(choices, list) or len(choices) != 1:
                        context_counts["unexpected_response_choices"] += 1
                        continue
                    try:
                        output = json.loads(choices[0]["message"]["content"])
                        verdict = output["verdict"]
                    except (KeyError, IndexError, ValueError, TypeError):
                        context_counts["invalid_response_json"] += 1
                        continue
                    if type(verdict) is not int or verdict not in (0, 1):
                        context_counts["invalid_binary_verdicts"] += 1
                        continue
                    key = _context_key(value["question"], value["answer"], value["context"])
                    verdicts[key].add(verdict)

    per_model: dict[str, dict[str, Any]] = {}
    for model in sorted({row["model"] for row in context_rows}):
        rows = [row for row in context_rows if row["model"] == model]
        result: dict[str, Any] = {
            "records": len(rows), "reconstructed_records": 0, "unresolved_records": 0,
            "records_with_missing_verdict": 0, "records_with_conflicting_verdict": 0,
            "total_evidence_blocks": sum(len(row["keys"]) for row in rows),
            "reconstructed_evidence_blocks": 0, "useful_blocks": 0,
            "zero_useful_questions": 0, "all_useful_questions": 0,
            "score_checked_records": 0, "score_mismatches": 0,
        }
        fractions = []
        position_count: Counter[int] = Counter()
        position_useful: Counter[int] = Counter()
        unresolved_examples = []
        for row in rows:
            values = [verdicts.get(key, set()) for key in row["keys"]]
            missing = any(not value for value in values)
            conflicting = any(len(value) > 1 for value in values)
            if missing or conflicting:
                result["unresolved_records"] += 1
                result["records_with_missing_verdict"] += missing
                result["records_with_conflicting_verdict"] += conflicting
                if len(unresolved_examples) < 10:
                    unresolved_examples.append({"index": row["index"], "missing": missing, "conflicting": conflicting})
                continue
            flags = [next(iter(value)) for value in values]
            count, useful = len(flags), sum(flags)
            result["reconstructed_records"] += 1
            result["reconstructed_evidence_blocks"] += count
            result["useful_blocks"] += useful
            result["zero_useful_questions"] += useful == 0
            result["all_useful_questions"] += useful == count
            fractions.append(useful / count)
            cumulative = 0
            numerator = 0.0
            for position, flag in enumerate(flags, start=1):
                cumulative += flag
                numerator += cumulative / position * flag
                position_count[position] += 1
                position_useful[position] += flag
            recomputed = numerator / (useful + 1e-10)
            score = row["score"]
            if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score):
                result["score_checked_records"] += 1
                result["score_mismatches"] += abs(recomputed - score) > 1e-10
        denominator = result["reconstructed_evidence_blocks"]
        result["pooled_useful_fraction"] = result["useful_blocks"] / denominator if denominator else None
        result["mean_question_useful_fraction"] = sum(fractions) / len(fractions) if fractions else None
        result["positions"] = [
            {"position": position, "blocks": count, "useful": position_useful[position],
             "fraction": position_useful[position] / count}
            for position, count in sorted(position_count.items())
        ]
        result["unresolved_examples"] = unresolved_examples
        per_model[model] = result
    expected_counts = (
        "recorded_judge_calls", "invalid_record_inputs", "unparsed_calls", "unexpected_input_shapes",
        "wrong_row_inputs", "unexpected_response_choices", "invalid_response_json", "invalid_binary_verdicts",
    )
    return {
        "method": "Exact (question, reference, context) keys join recorded binary ContextPrecision judgments across models; no inferred labels or model calls.",
        "context_precision": {
            "observed_request_keys": len(verdicts),
            "conflicting_request_keys": sum(len(values) > 1 for values in verdicts.values()),
            **{key: context_counts[key] for key in expected_counts},
            "by_model": per_model,
        },
        "factual_correctness_input_audit": {
            **factual, "by_model": dict(sorted(factual_models.items())),
            "wrong_input_examples": factual_examples,
            "scope": "Only recorded HTTP calls; correct row binding does not validate judge entailment decisions.",
        },
    }

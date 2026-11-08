import gzip
import json
from collections import Counter, defaultdict

import pytest

from benchmarking.analyze_results import METRICS, MODELS, analyze, inspect_calls, paired
from benchmarking.common import canonical, load_dataset
from benchmarking.storage import Store


def test_paired_statistics_use_only_common_questions_with_different_missing_rows():
    meno = {"common-a": 0.0, "common-b": 0.4, "only-meno": 1.0}
    gemma = {"common-a": 0.4, "common-b": 0.6, "only-gemma": 0.0}
    result = paired(meno, gemma, repetitions=1000, seed=42)
    assert result["n"] == 2
    assert result["meno_mean"] == pytest.approx(0.2)
    assert result["gemma_mean"] == pytest.approx(0.5)
    assert result["gemma_minus_meno"] == pytest.approx(0.3)
    assert result["gemma_wins"] == 2
    assert result["ties"] == result["meno_wins"] == 0
    assert result["delta_ci95"] == pytest.approx([0.2, 0.4])


def test_analysis_rejects_changed_failed_export_even_when_failure_count_matches(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps([{"question": "Question", "ground_truth": "Reference"}]))
    dataset_hash, samples = load_dataset(dataset)
    sample = samples[0]
    directory = tmp_path / "results"
    config = {"dataset_sha256": dataset_hash, "models": list(MODELS), "metrics": list(METRICS)}
    identity = {
        "model": "meno-lite", "metric": "faithfulness", "sample_id": sample["sample_id"], "index": 0,
        "user_input": sample["question"], "reference": sample["ground_truth"],
        "response": "Answer", "retrieved_contexts": ["Evidence"],
    }
    with Store(directory) as store:
        store.bind(config)
        key = f"meno-lite:{sample['sample_id']}:faithfulness"
        store.start(key, identity)
        store.fail(key, "Original failure")
        row = next(store.records(failures=True))
    row["error"] = "Changed failure"
    output = directory / "meno-lite"
    output.mkdir()
    with gzip.open(output / "scores.jsonl.gz", "wt") as stream:
        stream.write(canonical(row) + "\n")
    (directory / "summary.json").write_text(json.dumps({"config": config, "models": {}}))
    with pytest.raises(ValueError, match="по ключам, статусам и ошибкам"):
        analyze(directory, dataset, repetitions=10)


@pytest.mark.parametrize("metric,field,value", [
    ("context_precision", "verdict", 2),
    ("answer_relevancy", "noncommittal", True),
])
def test_http_audit_checks_top_level_binary_judgments(metric, field, value):
    audit = defaultdict(int)
    for key in ("http_statuses", "http_kinds", "judge_models", "finish_reasons", "judge_usage"):
        audit[key] = Counter()
    row = {
        "metric": metric,
        "details": {"http_calls": [{
            "kind": "judge", "status_code": 200,
            "request": {"chat_template_kwargs": {"enable_thinking": False}},
            "response": {"model": "judge", "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps({field: value})},
            }]},
        }]},
    }
    assert inspect_calls(row, audit) is None
    assert audit["nonbinary_judgments"] == 1

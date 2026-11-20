"""Offline checks for repeated-question weights and the failed-request denominator."""
import importlib
from pathlib import Path


def test_repeated_turns_do_not_count_as_independent_questions(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    analysis = importlib.import_module("aggregate_factorial")
    result = analysis.contrast([("a", 100, 1)] * 20 + [("b", 0, 2)])
    assert result["question_clusters"] == 2
    assert result["paired_repeated_turns"] == 21
    assert result["question_balanced_mean_delta"] == 50
    assert result["pooled_repeated_turn_deltas"]["median"] == 100


def test_phase_completion_never_hides_failed_case(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    analysis = importlib.import_module("aggregate_maxbatch")
    good = {"id": "a", "position": 0, "status": "ok", "total_ms": 100, "http_requests": {}}
    phases = [
        {"name": "warmup_ragu", "block": 0, "concurrency": 4, "requested": 8,
         "max_num_batched_tokens": 32768, "cases": [good] * 8, "menon": []},
        {"name": "measured", "block": 1, "concurrency": 1, "requested": 1, "status": "complete",
         "max_num_batched_tokens": 32768, "cases": [good], "menon": []},
        {"name": "measured", "block": 1, "concurrency": 1, "requested": 1, "status": "complete",
         "max_num_batched_tokens": 8192, "cases": [{**good, "status": "error", "error_type": "HTTPStatusError"}], "menon": []},
    ]
    result = analysis.summarize({"status": "complete", "phases": phases})
    assert result["measured_denominators"]["ragu_attempted"] == 2
    assert result["measured_denominators"]["ragu_status_counts"] == {"ok": 1, "error": 1}
    assert result["failed_measured_ragu_cases"][0]["max_num_batched_tokens"] == 8192

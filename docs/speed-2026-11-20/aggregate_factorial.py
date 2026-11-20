"""Offline numeric factorial summary; bootstrap independent question IDs, not repeats."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

METRICS = ("prepare_ms", "ttft_ms", "generation_ms", "source_tail_ms", "total_ms", "offered_to_finish_ms")
SEED = 1120


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    values = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    if not values:
        return {"n": 0}
    return {"n": len(values), "median": statistics.median(values), "p95": quantile(values, .95),
            "mean": statistics.mean(values)}


def contrast(rows, repetitions=2000):
    """Rows are (public ID, signed difference, block); each ID gets one vote."""
    groups = defaultdict(list)
    blocks = defaultdict(list)
    for case_id, value, block in rows:
        groups[case_id].append(value)
        blocks[block].append((case_id, value))
    result = {"pooled_repeated_turn_deltas": distribution([value for _, value, _ in rows]),
              "question_clusters": len(groups), "paired_repeated_turns": len(rows)}
    id_means = [statistics.mean(values) for _, values in sorted(groups.items())]
    if not id_means:
        return result
    result["question_balanced_mean_delta"] = statistics.mean(id_means)
    result["question_balanced_median_of_mean_deltas"] = statistics.median(id_means)
    result["per_block"] = {}
    for block, entries in blocks.items():
        by_id = defaultdict(list)
        for case_id, value in entries:
            by_id[case_id].append(value)
        means = [statistics.mean(values) for values in by_id.values()]
        result["per_block"][str(block)] = {"question_clusters": len(means), "paired_repeated_turns": len(entries),
                                           "question_balanced_mean_delta": statistics.mean(means),
                                           "question_balanced_median_of_mean_deltas": statistics.median(means)}
    if len(groups) < 2:
        return result
    rng = random.Random(SEED)
    mean_samples, median_samples = [], []
    for _ in range(repetitions):
        chosen = rng.choices(id_means, k=len(id_means))
        mean_samples.append(statistics.mean(chosen))
        median_samples.append(statistics.median(chosen))
    result["nominal_question_cluster_bootstrap95_mean"] = [quantile(mean_samples, .025), quantile(mean_samples, .975)]
    result["nominal_question_cluster_bootstrap95_median"] = [quantile(median_samples, .025), quantile(median_samples, .975)]
    result["inference_limit"] = "Only two temporal blocks; intervals do not establish general noninferiority"
    return result


def records(phase, field):
    return {(phase["block"], row["position"]): row for row in phase.get(field, [])}


def stability(indexed):
    output = []
    for load in (1, 4, 8):
        reference = indexed.get((load, 0, 0), {}).get("ragu", {})
        for early, priority in ((1, 0), (0, 1), (1, 1)):
            candidate = indexed.get((load, early, priority), {}).get("ragu", {})
            counts = Counter()
            changes = []
            for key in sorted(reference.keys() & candidate.keys()):
                a, b = reference[key], candidate[key]
                counts["paired_offered"] += 1
                if a["status"] != "ok" or b["status"] != "ok":
                    counts["excluded_failed_pair"] += 1
                    continue
                if a["id"] != b["id"]:
                    raise ValueError("Stability IDs do not match")
                equal = {name: a.get(field) == b.get(field) for name, field in
                         (("queries", "search_queries_sha256"), ("context", "context_sha256"),
                          ("evidence", "evidence_sha256"), ("answer", "answer_sha256"),
                          ("sources", "sources_sha256"))}
                for name, value in equal.items():
                    counts[name + "_equal"] += value
                if not equal["queries"]:
                    category = "live_analysis_or_rewrite_decision_changed"
                elif not equal["context"]:
                    category = ("retrieved_count_changed_with_same_queries" if a.get("retrieved_count") != b.get("retrieved_count")
                                else "context_selection_content_or_order_changed_with_same_queries")
                elif not equal["evidence"]:
                    category = "evidence_changed_despite_same_context"
                elif not equal["answer"]:
                    category = "generation_changed_on_same_context_and_evidence"
                elif not equal["sources"]:
                    category = "attribution_changed_on_same_answer_and_evidence"
                else:
                    category = "identical_observed_result"
                counts[category] += 1
                request_equal = a.get("http_requests") == b.get("http_requests")
                counts["http_request_counts_equal"] += request_equal
                if category != "identical_observed_result" or not request_equal:
                    changes.append({"id": a["id"], "block": key[0], "position": key[1], "classification": category,
                                    "equal": equal, "http_request_counts_equal": request_equal,
                                    "numeric_deltas_candidate_minus_reference": {
                                        metric: b[metric] - a[metric] for metric in
                                        ("context_chars", "retrieved_count", "evidence_units", "sources_count", "answer_chars",
                                         "prepare_ms", "ttft_ms", "total_ms")
                                        if isinstance(a.get(metric), (int, float)) and isinstance(b.get(metric), (int, float))},
                                    "request_count_deltas": {name: b.get("http_requests", {}).get(name, 0) - a.get("http_requests", {}).get(name, 0)
                                                              for name in a.get("http_requests", {}).keys() | b.get("http_requests", {}).keys()}})
            output.append({"load": load, "reference_early": 0, "reference_priority": 0,
                           "candidate_early": early, "candidate_priority": priority, "counts": dict(counts),
                           "changed_cases": changes,
                           "limit": "Classification locates the first changed observable stage; raw contexts and individual scores were not retained, so this does not diagnose semantic correctness"})
    return output


def repeatability(indexed):
    result = []
    for (load, early, priority), cell in sorted(indexed.items()):
        groups = defaultdict(list)
        for key, row in cell["ragu"].items():
            if row["status"] == "ok":
                groups[row["id"]].append((key, row))
        counts = Counter()
        changes = []
        for case_id, entries in sorted(groups.items()):
            entries.sort()
            for left in range(len(entries)):
                for right in range(left + 1, len(entries)):
                    ka, a = entries[left]
                    kb, b = entries[right]
                    counts["correlated_repeat_pairs"] += 1
                    equal = {name: a.get(field) == b.get(field) for name, field in
                             (("queries", "search_queries_sha256"), ("context", "context_sha256"),
                              ("evidence", "evidence_sha256"), ("answer", "answer_sha256"),
                              ("sources", "sources_sha256"))}
                    for name, value in equal.items():
                        counts[name + "_equal"] += value
                    if not equal["queries"] or not equal["context"]:
                        changes.append({"id": case_id, "earlier_block": ka[0], "earlier_position": ka[1],
                                        "later_block": kb[0], "later_position": kb[1], "equal": equal})
        result.append({"load": load, "early": early, "priority": priority,
                       "repeated_question_clusters": sum(len(entries) > 1 for entries in groups.values()),
                       "counts": dict(counts), "upstream_changed_repeats": changes,
                       "limit": "Repeated turns hold scheduling settings fixed and estimate live-model drift; pair counts are correlated within eight question IDs"})
    return result


def summarize(document):
    phases = [phase for phase in document.get("phases", []) if phase.get("block") in (1, 2)]
    indexed = {}
    for phase in phases:
        key = phase["concurrency"], int(phase["early_first_query"]), phase["rerank_priority"]
        indexed.setdefault(key, {"ragu": {}, "menon": {}, "phases": []})
        indexed[key]["ragu"].update(records(phase, "cases"))
        indexed[key]["menon"].update(records(phase, "menon"))
        indexed[key]["phases"].append(phase)
    unique_ids = sorted({row["id"] for cell in indexed.values() for row in cell["ragu"].values()})
    result = {"unique_question_ids": unique_ids, "unique_question_count": len(unique_ids),
              "temporal_blocks": sorted({phase["block"] for phase in phases}),
              "status": document.get("status"), "driver_sha256": document.get("driver_sha256"),
              "candidate_commit": document.get("candidate_commit"), "cells": [], "effects": [],
              "warnings": ["Burst loads are short and contain only eight unique questions",
                           "Live rewrite/analysis decisions can differ between scheduling arms",
                           "Source/answer equality alone does not measure semantic answer quality",
                           "Global GPU counters may include unrelated service traffic"]}
    for (load, early, priority), cell in sorted(indexed.items()):
        ragu = list(cell["ragu"].values())
        controls = list(cell["menon"].values())
        under = [row for row in controls if row.get("ragu_inflight_at_dispatch", 0) > 0]
        result["cells"].append({"load": load, "early": early, "priority": priority,
                                "phase_count": len(cell["phases"]),
                                "ragu_requested": sum(phase["requested"] for phase in cell["phases"]),
                                "question_repeat_counts": dict(Counter(row["id"] for row in ragu)),
                                "ragu_status_counts": dict(Counter(row["status"] for row in ragu)),
                                "ragu": {metric: distribution([row.get(metric) for row in ragu if row["status"] == "ok"])
                                         for metric in METRICS},
                                "ragu_throughput_per_second_by_block": {
                                    str(phase["block"]): {
                                        "completed": len(phase.get("cases", [])) / (phase["ragu_wall_ms"] / 1000),
                                        "successful": sum(row["status"] == "ok" for row in phase.get("cases", [])) / (phase["ragu_wall_ms"] / 1000)}
                                    for phase in cell["phases"] if phase.get("ragu_wall_ms")},
                                "menon_offered": len(controls), "menon_underload_offered": len(under),
                                "menon_status_counts": dict(Counter(row["status"] for row in controls)),
                                "menon_underload_status_counts": dict(Counter(row["status"] for row in under)),
                                "menon_underload": {metric: distribution([row.get(metric) for row in under
                                                                          if row["status"] == "ok"])
                                                    for metric in ("ttft_ms", "total_ms", "offered_lag_ms")}})
    for load in (1, 4, 8):
        for field in ("ragu", "menon"):
            for factor in ("early", "priority", "interaction"):
                metrics = METRICS if field == "ragu" else ("ttft_ms", "total_ms")
                output = {"load": load, "population": field, "factor": factor, "metrics": {}}
                for metric in metrics:
                    differences = []
                    equal_context = equal_queries = possible = success = 0
                    pair_specs = [[((0, p), -1), ((1, p), 1)] for p in (0, 1)] if factor == "early" else (
                        [[((e, 0), -1), ((e, 1), 1)] for e in (0, 1)] if factor == "priority" else
                        [[((0, 0), 1), ((1, 0), -1), ((0, 1), -1), ((1, 1), 1)]])
                    conditional = {}
                    for spec_index, specification in enumerate(pair_specs):
                        conditioned = []
                        arms = [(indexed.get((load, early, priority), {}).get(field, {}), weight)
                                for (early, priority), weight in specification]
                        keys = set.intersection(*(set(arm) for arm, _ in arms))
                        possible += len(keys)
                        for key in sorted(keys):
                            entries = [(arm[key], weight) for arm, weight in arms]
                            if any(row["status"] != "ok" or row.get(metric) is None for row, _ in entries):
                                continue
                            if field == "menon" and any(row.get("ragu_inflight_at_dispatch", 0) <= 0 for row, _ in entries):
                                continue
                            if len({row["id"] for row, _ in entries}) != 1:
                                raise ValueError("Mismatched question IDs in paired arms")
                            success += 1
                            if field == "ragu":
                                equal_context += len({row.get("context_sha256") for row, _ in entries}) == 1
                                equal_queries += len({row.get("search_queries_sha256") for row, _ in entries}) == 1
                            item = (entries[0][0]["id"], sum(row[metric] * weight for row, weight in entries), key[0])
                            differences.append(item)
                            conditioned.append(item)
                        if factor != "interaction":
                            label = f"priority{spec_index}" if factor == "early" else f"early{spec_index}"
                            conditional[label] = contrast(conditioned)
                    output["metrics"][metric] = {**contrast(differences), "conditioned_on_other_factor": conditional,
                                                    "possible_pairs": possible,
                                                    "successful_pairs": success,
                                                    "equal_context_pairs": equal_context if field == "ragu" else None,
                                                    "equal_query_pairs": equal_queries if field == "ragu" else None}
                result["effects"].append(output)
    result["stability_against_e0p0"] = stability(indexed)
    result["repeatability_same_arm"] = repeatability(indexed)
    alone = {row["name"]: row.get("cases", []) for row in document.get("menon_alone", [])}
    result["menon_alone"] = {name: {"status_counts": dict(Counter(row["status"] for row in rows)),
                                    "timings": {metric: distribution([row.get(metric) for row in rows if row["status"] == "ok"])
                                                for metric in ("ttft_ms", "total_ms")}}
                             for name, rows in alone.items()}
    before = {row["id"]: row for row in alone.get("before", []) if row["status"] == "ok"}
    after = {row["id"]: row for row in alone.get("after", []) if row["status"] == "ok"}
    result["menon_alone_drift_after_minus_before"] = {}
    for metric in ("ttft_ms", "total_ms"):
        differences = [(case_id, after[case_id][metric] - before[case_id][metric], 0)
                       for case_id in sorted(before.keys() & after.keys())
                       if before[case_id].get(metric) is not None and after[case_id].get(metric) is not None]
        result["menon_alone_drift_after_minus_before"][metric] = contrast(differences)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

"""CPU-only equivalence and timing probe against a frozen GML and its chunk KV."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import random
import resource
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenRetriever:
    def __init__(self, entities):
        self.entities = entities

    async def query_entities(self, queries, top_k):
        assert len(queries) == 1
        entities = self.entities[:top_k]
        hits = [SimpleNamespace(distance=1 - i / 100) for i in range(len(entities))]
        return [(list(entities), hits)]


class CpuScorer:
    def __init__(self):
        self.calls = []

    async def score(self, question, texts):
        digests = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        self.calls.append((question, tuple(digests)))
        scores = [(i, int(digest[:8], 16) % 5 / 4) for i, digest in enumerate(digests)]
        return sorted(scores, key=lambda pair: -pair[1])


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--hub-min-edges", type=int, default=5000)
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_dir))
    from ragu.common.logger import logger
    from ragu.graph.index import Index
    from ragu.graph.types import Entity, Relation
    from ragu.search_engine.local_search import LocalParams
    from ragu.storage.graph_storage_adapters.networkx_adapter import NetworkXStorage
    from ragu.storage.kv_storage_adapters.json_storage import JsonKVStorage

    logger.remove()
    baseline = load_module("hub_baseline", args.code_dir / "local_search_baseline.py")
    optimized = load_module("hub_candidate", args.code_dir / "local_search_candidate.py")
    prior_path = args.code_dir / "local_search_prior.py"
    prior = load_module("hub_prior", prior_path) if prior_path.is_file() else None
    graph_file = args.graph_dir / "knowledge_graph.gml"
    kv_names = ("kv_chunks.json", "kv_community.json", "kv_community_summary.json")
    assert graph_file.is_file() and all((args.graph_dir / name).is_file() for name in kv_names)
    before_stats = {name: (args.graph_dir / name).stat() for name in (graph_file.name, *kv_names)}
    started = time.perf_counter()
    backend = NetworkXStorage(filename=str(graph_file), node_cls=Entity, edge_cls=Relation)
    print(json.dumps({"phase": "gml_loaded", "seconds": time.perf_counter() - started}), flush=True)
    index = object.__new__(Index)
    index.graph_backend = backend
    for attribute, filename in (
        ("chunks_kv_storage", "kv_chunks.json"),
        ("community_kv_storage", "kv_community.json"),
        ("community_summary_kv_storage", "kv_community_summary.json"),
    ):
        setattr(index, attribute, JsonKVStorage(storage_folder=str(args.graph_dir), filename=filename))
    knowledge_graph = SimpleNamespace(index=index)
    graph = backend._graph
    print(json.dumps({"phase": "kv_loaded", "load_seconds": time.perf_counter() - started,
                      "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}), flush=True)
    hub_ids = optimized.hub_candidates(knowledge_graph, args.hub_min_edges)
    assert hub_ids
    prepared = time.perf_counter()
    hubs = await optimized.build_hub_index(knowledge_graph, hub_ids, 200)
    preparation_s = time.perf_counter() - prepared
    print(json.dumps({"phase": "hubs_ready", "hub_count": len(hubs), "seconds": preparation_s}), flush=True)
    ordered_hubs = sorted(hub_ids, key=lambda node: (-graph.degree(node), node))
    rng = random.Random(1147)
    hub_set = set(hub_ids)
    non_hubs = [node for node in graph if node not in hub_set]
    sample = rng.sample(non_hubs, min(150, len(non_hubs)))
    cases = []
    for number in range(args.queries):
        anchor_hubs = ordered_hubs[:min(len(ordered_hubs), 1 + number % 4)]
        neighbors = list(dict.fromkeys(
            node for hub in anchor_hubs for node in [*graph.successors(hub), *graph.predecessors(hub)]
            if node not in hub_set
        ))
        picked = rng.sample(neighbors, min(8, len(neighbors)))
        background = rng.sample(sample, min(12, len(sample)))
        if number % 6 == 0:
            ids = anchor_hubs
        elif number % 6 == 1:
            ids = anchor_hubs + picked + background
        elif number % 6 == 2:
            ids = background + picked + anchor_hubs
        elif number % 6 == 3:
            ids = background
        elif number % 6 == 4:
            ids = picked[:5] + anchor_hubs + background
        else:
            ids = picked + background + anchor_hubs
        ids = list(dict.fromkeys(ids))[:20]
        entities = [entity for entity in await index.get_nodes(ids) if entity is not None]
        assert entities
        cases.append(entities)

    modules = {"baseline": baseline, "hub_index": optimized}
    if prior is not None:
        modules["prior_hub_index"] = prior
    times = {name: [] for name in modules}
    comparisons = []
    for number, anchors in enumerate(cases):
        outputs = {}
        inputs = {}
        arms = list(modules)
        shift = number % len(arms)
        order = arms[shift:] + arms[:shift]
        for arm in order:
            module = modules[arm]
            engine = object.__new__(module.EfficientLocalSearchEngine)
            engine.knowledge_graph = knowledge_graph
            engine.max_related_candidates = 200
            if arm != "baseline":
                engine.hub_index = hubs
            engine.retriever = FrozenRetriever(anchors)
            engine.reranker = CpuScorer()
            tick = time.perf_counter()
            [outcome] = await engine.batch_search([f"cpu-probe-{number:02d}"], LocalParams(top_k=20, rerank_top_k=20))
            times[arm].append((time.perf_counter() - tick) * 1000)
            outputs[arm] = asdict(outcome)
            inputs[arm] = engine.reranker.calls
        assert outputs["baseline"] == outputs["hub_index"], f"Result mismatch at case {number}"
        assert inputs["baseline"] == inputs["hub_index"], f"Scorer-input mismatch at case {number}"
        if prior is not None:
            assert outputs["baseline"] == outputs["prior_hub_index"], f"Prior-result mismatch at case {number}"
            assert inputs["baseline"] == inputs["prior_hub_index"], f"Prior-scorer mismatch at case {number}"
        comparisons.append({"case": number, "anchors": len(anchors),
                            "hub_anchors": sum(entity.id in hub_set for entity in anchors),
                            "relations": len(outputs["baseline"]["result"]["relations"]),
                            "chunks": len(outputs["baseline"]["result"]["chunks"]),
                            "baseline_ms": times["baseline"][-1], "hub_index_ms": times["hub_index"][-1]})
        if prior is not None:
            comparisons[-1]["prior_hub_index_ms"] = times["prior_hub_index"][-1]
        print(json.dumps({"phase": "case_checked", **comparisons[-1]}), flush=True)
    for name, before in before_stats.items():
        after = (args.graph_dir / name).stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    result = {
        "graph_dir": str(args.graph_dir), "nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
        "hub_min_edges": args.hub_min_edges, "hub_count": len(hubs), "preparation_s": preparation_s,
        "queries": len(cases), "exact_full_results": True, "exact_scorer_inputs": True,
        "source_files_unchanged": True, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "timing_ms": {arm: {"median": statistics.median(values), "p95": percentile(values, .95),
                            "total": sum(values)} for arm, values in times.items()},
        "cases": comparisons,
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"phase": "complete", **{key: value for key, value in result.items() if key != "cases"}}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import builtins
import gzip
import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from benchmarking import collect as collect_module
from benchmarking.collect import collect, validate_capture
from benchmarking.common import FatalRunError, digest, load_dataset, run_jobs
from benchmarking.storage import Store


def _arguments(tmp_path, **overrides):
    dataset = tmp_path / "questions.json"
    dataset.write_text(json.dumps([
        {"question": "QUESTION_0", "ground_truth": "SECRET_REFERENCE_0"},
        {"question": "QUESTION_1", "ground_truth": "SECRET_REFERENCE_1"},
    ]), encoding="utf-8")
    values = {
        "dataset": dataset, "limit": None, "backend_env": None, "backend_url": "http://backend",
        "timeout": 30, "concurrencies": {"meno-lite": 1}, "models": ["meno-lite"],
        "model_revisions": {"meno-lite": "weights-v1"}, "knowledge_base_id": "nsu-ragu",
        "max_tokens": 128, "temperature": 0.0, "output_dir": tmp_path / "collection", "attempts": 1,
    }
    return SimpleNamespace(**{**values, **overrides})


def _preflight(fingerprint="backend-v1"):
    return {
        "schema_version": 2, "fingerprint": fingerprint,
        "identity": {"knowledge_base_id": "nsu-ragu", "models": [{"id": "meno-lite"}],
                     "knowledge": {"provenance_complete": True}},
    }


def _response(question="QUESTION_0", *, finish="stop"):
    text = "Retrieved fact."
    return {
        "model": "meno-lite",
        "choices": [{"message": {"content": "Generated answer."}, "finish_reason": "stop"}],
        "pipeline_trace": {
            "schema_version": 2, "question": question, "search_queries": [question],
            "final_context": text,
            "final_evidence": [{"kind": "document", "text": text, "context_start": 0,
                                "context_end": len(text), "document_index": 0, "chunk_ids": [0], "sources": []}],
            "retrieved_contexts": [text],
            "qa_messages": [{"role": "user", "content": question + "\n" + text}],
            "sources": [], "chunks": {"0": {"text": text}},
            "runtime": {"core": {"model_id": "meno-lite"}, "generation": {"model_id": "meno-lite"}},
            "generation": {"finish_reason": finish, "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
        },
    }


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("BENCH_API_TOKEN", "benchmark-secret")
    original_client = httpx.AsyncClient
    state = {"requests": [], "fingerprint": "backend-v1", "preflight_status": 200,
             "preflight_header": "1", "fail_questions": set(), "finish": "stop", "posts": Counter(),
             "response_fingerprints": {}, "readyz": {"status": "ok", "rag_ready": True},
             "catalog": [{"id": "meno-lite", "endpoint": "http://llm", "context_length": 16384}]}

    def handler(request):
        state["requests"].append(request)
        assert request.headers["Authorization"] == "Bearer benchmark-secret"
        if request.method == "GET":
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": state["catalog"]})
            if request.url.path == "/readyz":
                return httpx.Response(200, json=state["readyz"])
            assert request.url.path == "/v1/benchmark/config"
            return httpx.Response(state["preflight_status"], json=_preflight(state["fingerprint"]),
                                  headers={"X-Bench-Mode": state["preflight_header"]})
        assert request.method == "POST" and request.url.path == "/v1/chat/completions"
        assert request.headers["X-Benchmark-Fingerprint"] == state["fingerprint"]
        data = json.loads(request.content)
        question = data["messages"][0]["content"]
        state["posts"][question] += 1
        if question in state["fail_questions"]:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(200, json=_response(question, finish=state["finish"]),
                              headers={"X-Bench-Mode": "1", "X-Benchmark-Fingerprint":
                                       state["response_fingerprints"].get(question, state["fingerprint"])})

    monkeypatch.setattr(collect_module.httpx, "AsyncClient", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs,
    ))
    return state


@pytest.mark.asyncio
async def test_collect_never_sends_reference_or_imports_judge_and_durably_saves_trace(tmp_path, server, monkeypatch):
    args = _arguments(tmp_path)
    original_import = builtins.__import__

    def no_judge(name, *positional, **keywords):
        if name == "ragas" or name.startswith("ragas.") or name in {
            "benchmarking.evaluate", "benchmarking.ragas_adapter",
        }:
            raise AssertionError("First pass must not import judge components")
        return original_import(name, *positional, **keywords)

    monkeypatch.setattr(builtins, "__import__", no_judge)
    assert await collect(args) == 0
    posts = [r for r in server["requests"] if r.method == "POST"]
    assert len(posts) == 2
    for request in posts:
        assert "SECRET_REFERENCE" not in request.content.decode()
        payload = json.loads(request.content)
        assert set(payload) == {"model", "knowledge_base_id", "messages", "stream", "max_tokens", "temperature"}
        assert payload["stream"] is False
    with Store(args.output_dir, readonly=True) as store:
        records = list(store.records())
        assert len(records) == 2
        for row in records:
            assert row["ground_truth"].startswith("SECRET_REFERENCE_")
            assert row["raw_response"]["pipeline_trace"]["generation"]["finish_reason"] == "stop"
            assert row["retrieved_contexts"] == ["Retrieved fact."]
            unsigned = {key: value for key, value in row.items() if key != "capture_sha256"}
            assert row["capture_sha256"] == digest(unsigned)
        assert store.counts() == {"complete": 2}
    assert (args.output_dir / "manifest.json").is_file()
    exported = next(args.output_dir.glob("*/responses.jsonl.gz"))
    with gzip.open(exported, "rt", encoding="utf-8") as stream:
        assert len([json.loads(line) for line in stream]) == 2
    assert "benchmark-secret" not in (args.output_dir / "manifest.json").read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,header", [(404, "1"), (200, ""), (403, "")])
async def test_failed_readonly_preflight_prevents_all_generation_requests(tmp_path, server, status, header):
    args = _arguments(tmp_path)
    server.update(preflight_status=status, preflight_header=header)
    with pytest.raises(ValueError, match="Проверка режима"):
        await collect(args)
    assert [r.method for r in server["requests"]] == ["GET"]
    assert not args.output_dir.exists()


@pytest.mark.asyncio
async def test_unready_backend_reports_readiness_instead_of_token_failure(tmp_path, server):
    args = _arguments(tmp_path)
    server["preflight_status"] = 503
    with pytest.raises(ValueError, match="Backend пока не готов") as error:
        await collect(args)
    assert "BENCH_API_TOKEN" not in str(error.value)
    assert "stand_resources_load_failed" in str(error.value)
    assert [request.method for request in server["requests"]] == ["GET"]
    assert not args.output_dir.exists()


@pytest.mark.asyncio
async def test_collection_resume_changes_concurrency_without_repeating_successes(tmp_path, server):
    args = _arguments(tmp_path)
    assert await collect(args) == 0
    with Store(args.output_dir, readonly=True) as store:
        before = {row["sample_id"]: row["capture_sha256"] for row in store.records()}
    args.concurrencies = {"meno-lite": 8}
    args.timeout = 60
    args.attempts = 2
    assert await collect(args) == 0
    assert server["posts"] == Counter({"QUESTION_0": 1, "QUESTION_1": 1})
    with Store(args.output_dir, readonly=True) as store:
        assert {row["sample_id"]: row["capture_sha256"] for row in store.records()} == before


@pytest.mark.asyncio
async def test_failed_question_retries_on_resume_without_losing_successful_siblings(tmp_path, server):
    args = _arguments(tmp_path)
    server["fail_questions"] = {"QUESTION_1"}
    assert await collect(args) == 1
    with Store(args.output_dir, readonly=True) as store:
        assert store.counts() == {"complete": 1, "failed": 1}
        successful = next(store.records())["capture_sha256"]
    server["fail_questions"] = set()
    args.concurrencies = {"meno-lite": 4}
    assert await collect(args) == 0
    assert server["posts"] == Counter({"QUESTION_0": 1, "QUESTION_1": 2})
    with Store(args.output_dir, readonly=True) as store:
        assert successful in [row["capture_sha256"] for row in store.records()]
        assert store.counts() == {"complete": 2}


@pytest.mark.asyncio
async def test_changed_backend_fingerprint_rejects_resume_before_post(tmp_path, server):
    args = _arguments(tmp_path)
    assert await collect(args) == 0
    before = server["posts"].copy()
    server["fingerprint"] = "backend-v2"
    with pytest.raises(ValueError, match="backend изменились"):
        await collect(args)
    assert server["posts"] == before


@pytest.mark.asyncio
async def test_changed_selected_model_context_rejects_resume_before_post(tmp_path, server):
    args = _arguments(tmp_path)
    assert await collect(args) == 0
    before = server["posts"].copy()
    server["catalog"][0]["context_length"] = 32768
    with pytest.raises(ValueError, match="backend изменились"):
        await collect(args)
    assert server["posts"] == before


@pytest.mark.asyncio
async def test_unrelated_model_catalog_changes_do_not_invalidate_finished_captures(tmp_path, server):
    args = _arguments(tmp_path)
    assert await collect(args) == 0
    server["catalog"].append({"id": "unrelated", "endpoint": "http://another", "context_length": 8192})
    assert await collect(args) == 0
    assert server["posts"] == Counter({"QUESTION_0": 1, "QUESTION_1": 1})


@pytest.mark.asyncio
async def test_backend_change_during_collection_rejects_only_new_mismatching_capture(tmp_path, server):
    args = _arguments(tmp_path)
    server["response_fingerprints"] = {"QUESTION_1": "backend-v2"}
    with pytest.raises(FatalRunError, match="Backend изменился"):
        await collect(args)
    with Store(args.output_dir, readonly=True) as store:
        assert store.counts() == {"complete": 1, "failed": 1}
        assert next(store.records())["question"] == "QUESTION_0"
        failed = next(row for row in store.records(failures=True) if row["status"] != "complete")
        assert "Backend изменился" in failed["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://alice:secret@backend", "http://backend?token=secret",
                                 "http://backend#secret", "ftp://backend"])
async def test_unsafe_backend_url_rejected_without_network_or_checkpoint(tmp_path, server, url):
    args = _arguments(tmp_path, backend_url=url)
    with pytest.raises(ValueError, match="backend-url"):
        await collect(args)
    assert server["requests"] == []
    assert not args.output_dir.exists()


def _cli_arguments(command, args):
    return [command, "--dataset", str(args.dataset), "--output-dir", str(args.output_dir),
            "--backend-url", args.backend_url, "--models", "meno-lite", "--max-tokens", str(args.max_tokens),
            "--model-revision", "meno-lite=weights-v1", "--attempts", "1"]


def test_smoke_checks_one_question_and_preserves_it_for_full_collection(tmp_path, server):
    from benchmarking.cli import main

    args = _arguments(tmp_path)
    with pytest.raises(SystemExit) as exit_info:
        main([*_cli_arguments("smoke", args), "--limit", "2"])
    assert exit_info.value.code == 0
    assert server["posts"] == Counter({"QUESTION_0": 1})
    report = json.loads((args.output_dir / "smoke_report.json").read_text())
    assert report["status"] == "passed"
    assert report["readyz"]["rag_ready"] is True
    assert report["results"][0]["evidence_blocks"] == 1
    assert report["results"][0]["finish_reason"] == "stop"
    with pytest.raises(SystemExit) as exit_info:
        main(_cli_arguments("collect", args))
    assert exit_info.value.code == 0
    assert server["posts"] == Counter({"QUESTION_0": 1, "QUESTION_1": 1})


def test_smoke_failed_readiness_aborts_before_generation(tmp_path, server):
    from benchmarking.cli import main

    args = _arguments(tmp_path)
    server["readyz"] = {"status": "degraded", "rag_ready": False}
    with pytest.raises(SystemExit) as exit_info:
        main(_cli_arguments("smoke", args))
    assert exit_info.value.code == 1
    assert server["posts"] == Counter()
    assert not args.output_dir.exists()


def test_smoke_rejects_incomplete_generation_and_reports_failure(tmp_path, server):
    from benchmarking.cli import main

    args = _arguments(tmp_path)
    server["finish"] = "length"
    with pytest.raises(SystemExit) as exit_info:
        main(_cli_arguments("smoke", args))
    assert exit_info.value.code == 1
    report = json.loads((args.output_dir / "smoke_report.json").read_text())
    assert report["status"] == "failed"
    assert report["results"] == []
    assert server["posts"] == Counter({"QUESTION_0": 1})


@pytest.mark.asyncio
async def test_upstream_length_finish_is_not_a_success_despite_api_stop(tmp_path, server):
    args = _arguments(tmp_path)
    server["finish"] = "length"
    assert await collect(args) == 1
    with Store(args.output_dir, readonly=True) as store:
        assert store.counts() == {"failed": 2}
        assert list(store.records()) == []
        assert all("length" in row["error"] for row in store.records(failures=True))


@pytest.mark.parametrize("mutate", [
    lambda data: data.pop("pipeline_trace"),
    lambda data: data["pipeline_trace"].update(retrieved_contexts=[]),
    lambda data: data["pipeline_trace"]["final_evidence"][0].update(text="Not actual evidence"),
    lambda data: data["pipeline_trace"]["generation"].update(finish_reason=None),
])
def test_incomplete_capture_is_rejected(mutate):
    data = _response()
    mutate(data)
    with pytest.raises(ValueError):
        validate_capture(data)


@pytest.mark.parametrize("start,end", [(1, 15), (0, 1000000), (False, 15), (0, True), (-1, 15)])
def test_context_evidence_offsets_must_be_exact_in_range_integers(start, end):
    data = _response()
    data["pipeline_trace"]["final_evidence"][0].update(context_start=start, context_end=end)
    with pytest.raises(ValueError):
        validate_capture(data)


def test_overlapping_evidence_offsets_are_rejected():
    data = _response()
    trace = data["pipeline_trace"]
    trace["final_evidence"].append(deepcopy(trace["final_evidence"][0]))
    trace["retrieved_contexts"].append(trace["retrieved_contexts"][0])
    with pytest.raises(ValueError):
        validate_capture(data)


def _local_collection(args, *, partial=False):
    dataset_hash, samples = load_dataset(args.dataset)
    with Store(args.output_dir) as store:
        store.bind({"kind": "collection", "dataset_size": len(samples), "dataset_sha256": dataset_hash,
                    "models": ["meno-lite"]})
        for sample in samples[:1] if partial else samples:
            row = {"model": "meno-lite", **sample, "response": "Generated answer.",
                   "retrieved_contexts": ["Retrieved fact."], "status": "complete"}
            row["capture_sha256"] = digest(row)
            key = "meno-lite:" + sample["sample_id"]
            store.start(key, sample)
            store.complete(key, row)


def _evaluation_arguments(collection_args, tmp_path, **overrides):
    values = {
        "collection_dir": collection_args.output_dir, "output_dir": tmp_path / "evaluation",
        "judge_url": "http://judge/v1", "judge_model": "judge-model", "judge_revision": "judge-v1",
        "embedder_url": "http://embedder/v1", "embedder_model": "BAAI/bge-m3",
        "judge_api_key": "judge-secret", "embedder_api_key": "embedder-secret",
        "max_tokens": 4096, "timeout": 30, "enable_thinking": False, "allow_partial": False,
        "metrics": ["faithfulness", "factual_correctness"], "limit": None, "concurrency": 2, "attempts": 1,
    }
    return SimpleNamespace(**{**values, **overrides})


@pytest.fixture
def fake_judge(monkeypatch):
    from benchmarking import ragas_adapter

    state = {"calls": [], "fail": True, "opens": 0}

    class FakeAdapter:
        metric_names = ("faithfulness", "factual_correctness")

        def __init__(self, config):
            self.config = config

        async def __aenter__(self):
            state["opens"] += 1
            return self

        async def __aexit__(self, *args):
            return False

        async def score(self, metric, sample):
            state["calls"].append((metric, deepcopy(sample)))
            if state["fail"] and metric == "factual_correctness" and sample["index"] == 0:
                raise ragas_adapter.RagasMetricError(
                    "judge transient failure", metric_name=metric,
                    details={"http_calls": [{"status_code": 503, "response": {"error": "overloaded"}}]},
                )
            return {"score": 0.75, "reason": "supported", "details": {"usage": {"total_tokens": 15}}}

    monkeypatch.setattr(ragas_adapter, "RagasAdapter", FakeAdapter)
    return state


@pytest.mark.asyncio
async def test_evaluation_checkpoints_each_metric_and_retries_only_failed_score(tmp_path, fake_judge):
    from benchmarking.evaluate import evaluate

    collection_args = _arguments(tmp_path)
    _local_collection(collection_args)
    args = _evaluation_arguments(collection_args, tmp_path)
    assert await evaluate(args) == 1
    assert len(fake_judge["calls"]) == 4
    with Store(args.output_dir, readonly=True) as store:
        assert store.counts() == {"complete": 3, "failed": 1}
        assert all(row["score"] == 0.75 for row in store.records())
        before = {row["sample_id"] + row["metric"]: row for row in store.records()}
        failed = next(row for row in store.records(failures=True) if row["status"] == "failed")
        assert failed["score"] is None
        assert failed["details"]["http_calls"][0]["status_code"] == 503
    fake_judge["fail"] = False
    args.concurrency = 7
    args.timeout = 60
    assert await evaluate(args) == 0
    assert len(fake_judge["calls"]) == 5
    assert fake_judge["calls"][-1][0] == "factual_correctness"
    assert fake_judge["calls"][-1][1]["index"] == 0
    assert all(sample["reference"].startswith("SECRET_REFERENCE_") for _metric, sample in fake_judge["calls"])
    with Store(args.output_dir, readonly=True) as store:
        assert store.counts() == {"complete": 4}
        for row in store.records():
            key = row["sample_id"] + row["metric"]
            if key in before:
                assert row == before[key]
        failed_attempt = store.db.execute("SELECT details FROM attempts WHERE error IS NOT NULL").fetchone()
        assert json.loads(gzip.decompress(failed_attempt[0]))["http_calls"][0]["status_code"] == 503
    summary = json.loads((args.output_dir / "summary.json").read_text())
    assert summary["models"]["meno-lite"]["faithfulness"]["coverage"] == 1.0
    assert "judge-secret" not in json.dumps(summary) and "embedder-secret" not in json.dumps(summary)


@pytest.mark.asyncio
async def test_partial_collection_rejected_before_judge_is_opened(tmp_path, fake_judge):
    from benchmarking.evaluate import evaluate

    collection_args = _arguments(tmp_path)
    _local_collection(collection_args, partial=True)
    with pytest.raises(ValueError, match="Первый проход не завершён"):
        await evaluate(_evaluation_arguments(collection_args, tmp_path))
    assert fake_judge["opens"] == 0
    assert fake_judge["calls"] == []


@pytest.mark.asyncio
async def test_changed_judge_revision_rejects_completed_score_reuse(tmp_path, fake_judge):
    from benchmarking.evaluate import evaluate

    collection_args = _arguments(tmp_path)
    _local_collection(collection_args)
    args = _evaluation_arguments(collection_args, tmp_path)
    fake_judge["fail"] = False
    assert await evaluate(args) == 0
    args.judge_revision = "judge-v2"
    with pytest.raises(ValueError, match="backend изменились"):
        await evaluate(args)
    assert len(fake_judge["calls"]) == 4


@pytest.mark.asyncio
async def test_changed_metric_adapter_version_rejects_old_score_reuse(tmp_path, fake_judge, monkeypatch):
    from benchmarking import ragas_adapter
    from benchmarking.evaluate import evaluate

    collection_args = _arguments(tmp_path)
    _local_collection(collection_args)
    args = _evaluation_arguments(collection_args, tmp_path)
    fake_judge["fail"] = False
    assert await evaluate(args) == 0
    monkeypatch.setattr(ragas_adapter, "ADAPTER_VERSION", ragas_adapter.ADAPTER_VERSION + 1)
    with pytest.raises(ValueError, match="backend изменились"):
        await evaluate(args)
    assert len(fake_judge["calls"]) == 4


@pytest.mark.asyncio
async def test_duplicate_metrics_rejected_before_judge_is_opened(tmp_path, fake_judge):
    from benchmarking.evaluate import evaluate

    collection_args = _arguments(tmp_path)
    _local_collection(collection_args)
    args = _evaluation_arguments(collection_args, tmp_path, metrics=["faithfulness", "faithfulness"])
    with pytest.raises(ValueError, match="повторы"):
        await evaluate(args)
    assert fake_judge["opens"] == 0
    assert fake_judge["calls"] == []


@pytest.mark.asyncio
async def test_cancellation_keeps_completed_jobs_and_resume_retries_only_unfinished(tmp_path):
    directory = tmp_path / "checkpoints"
    jobs = [(str(index), {"index": index}) for index in range(3)]
    blocked = asyncio.Event()
    calls = []

    async def interrupted_operation(sample):
        calls.append(sample["index"])
        if sample["index"] == 1:
            blocked.set()
            await asyncio.Future()
        return {"index": sample["index"], "status": "complete", "answer": "durable"}

    with Store(directory) as store:
        task = asyncio.create_task(run_jobs(
            store, jobs, interrupted_operation, concurrency=1, attempts=1, label="cancellation-test",
        ))
        await asyncio.wait_for(blocked.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert store.counts() == {"complete": 1, "running": 1}
        assert next(store.records())["index"] == 0
    assert calls == [0, 1]

    async def resumed_operation(sample):
        calls.append(sample["index"])
        return {"index": sample["index"], "status": "complete", "answer": "resumed"}

    with Store(directory) as store:
        await run_jobs(store, jobs, resumed_operation, concurrency=3, attempts=1, label="resumed-test")
        assert store.counts() == {"complete": 3}
        assert next(row for row in store.records() if row["index"] == 0)["answer"] == "durable"
        attempts = dict(store.db.execute("SELECT job_key,COUNT(*) FROM attempts GROUP BY job_key"))
        assert attempts == {"0": 1, "1": 2, "2": 1}
    assert Counter(calls) == Counter({0: 1, 1: 2, 2: 1})

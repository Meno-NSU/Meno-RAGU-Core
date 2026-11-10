import json

from fastapi.testclient import TestClient

from meno_rag.api.events import StageEvent, StageName, StageStatus


def _sources():
    rows = []
    for i in range(30):
        if i in (2, 5, 9):
            rows.append({"document_title": "", "source_url": "https://nsu.ru/x"})
        elif i in (3, 4):
            rows.append({"document_title": f"Doc {i}", "source_url": "ftp://nsu.ru/f"})
        else:
            rows.append({"document_title": f"Doc {i}", "source_url": f"https://nsu.ru/{i}"})
    return rows


class _Outcome:
    question = "q"
    search_queries = ["q"]
    stage_durations_ms: dict = {}
    stage_details: dict = {}

    def __init__(self):
        self.sources = _sources()


class _Pipeline:
    async def prepare(self, *, messages, runtime, stage_sink=None, capture_trace=False, generation_max_tokens=None):
        async def emit(event):
            if stage_sink is not None:
                await stage_sink(event)

        await emit(StageEvent(stage=StageName.RETRIEVAL, status=StageStatus.STARTED))
        await emit(
            StageEvent(
                stage=StageName.RETRIEVAL,
                status=StageStatus.COMPLETED,
                duration_ms=1.0,
                detail={"chunks_found": 152, "entities": 40, "relations": 316},
            )
        )
        await emit(
            StageEvent(
                stage=StageName.RERANK, status=StageStatus.COMPLETED, duration_ms=1.0, detail={"kept": 12, "from": 55}
            )
        )
        await emit(StageEvent(stage=StageName.FUSION, status=StageStatus.SKIPPED))
        return _Outcome()

    async def stream_text(self, *, outcome, runtime, max_tokens, temperature):
        yield "ok"

    async def generate_text(self, *, outcome, runtime, max_tokens, temperature):
        return "ok"


def _client(monkeypatch):
    from meno_rag.api import main as main_mod
    from meno_rag.stand.pipeline import ModelRuntime, PipelineRuntime

    rt = PipelineRuntime.uniform(ModelRuntime(provider="vllm", model_id="m", base_url="http://e/v1"))

    async def fake_resolve(app, requested_model):
        return rt

    async def fake_persist(**kwargs):
        return None

    monkeypatch.setattr(main_mod, "_resolve_runtime", fake_resolve)
    monkeypatch.setattr(main_mod, "_persist_success", fake_persist)
    return main_mod


def _events(text):
    out = []
    for block in text.split("\n\n"):
        lines = block.split("\n")
        if lines and lines[0].startswith("event: ") and len(lines) > 1:
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def _assert_valid(sources):
    assert 0 < len(sources) <= 20
    for s in sources:
        assert s["document_title"]
        assert s["source_url"] == "" or s["source_url"].startswith(("http://", "https://"))


def test_stream_sends_at_most_20_valid_sources_and_russian_lines(monkeypatch):
    main_mod = _client(monkeypatch)
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _Pipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True},
        )
    events = _events(r.text)
    sources = next(p for name, p in events if name == "sources")["sources"]
    _assert_valid(sources)
    assert [s["document_title"] for s in sources][:3] == ["Doc 0", "Doc 1", "Doc 3"]
    assert sources[2]["source_url"] == ""

    stages = [p for name, p in events if name == "stage"]
    assert len(stages) >= 4
    for st in stages:
        assert isinstance(st["detail"]["lines"], list)
        assert all(isinstance(x, str) for x in st["detail"]["lines"])
    done = {(st["stage"], st["status"]): st["detail"] for st in stages}
    assert done[("retrieval", "completed")]["lines"] == ["Найдено фрагментов: 152", "сущностей: 40, связей: 316"]
    assert done[("retrieval", "completed")]["chunks_found"] == 152
    assert done[("rerank", "completed")]["lines"] == ["Оставлено 12 из 55"]
    assert done[("fusion", "skipped")]["lines"] == []


def test_non_stream_sources_are_clean(monkeypatch):
    main_mod = _client(monkeypatch)
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _Pipeline()
        r = c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": False},
        )
    _assert_valid(r.json()["sources"])


def test_stage_lines_for_other_stages():
    from meno_rag.api.events import stage_lines

    assert stage_lines("fusion", {"candidates": 508}) == ["Кандидатов: 508"]
    assert stage_lines("query_rewrite", {"search_queries": ["a", "b"]}) == ["Поисковых запросов: 2"]
    assert stage_lines("context_assembly", {"sources": 9, "context_tokens": 1035}) == [
        "Источников: 9, токенов контекста: 1035"
    ]
    assert stage_lines("abbreviation_expansion", {"expanded": "НГУ — Новосибирский", "original": ""}) == [
        "Расшифровано: НГУ — Новосибирский"
    ]
    assert stage_lines("abbreviation_expansion", {"expanded": "", "original": ""}) == []
    assert stage_lines("generation", None) == []


class _TracedOutcome(_Outcome):
    context = "==========\nDOCUMENT 1\n==========\n\nпервый\n\n==========\nDOCUMENT 2\n==========\n\nвторой"
    qa_messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
    prepared_dialogue_history = "hist"
    retrieved = [{"chunk_id": 1}]
    fewshots = [{"question": "f"}]

    def __init__(self, with_trace=True):
        super().__init__()
        self.trace = (
            {
                "schema_version": 2,
                "final_evidence": [
                    {
                        "kind": "document",
                        "text": "первый",
                        "document_index": 7,
                        "sources": [{"document_title": "Doc A", "source_url": "https://nsu.ru/a"}],
                    },
                    {"kind": "document", "text": "второй", "document_index": 8, "sources": []},
                ],
                "runtime": {"core": {"model_id": "core-m"}, "generation": {"model_id": "gen-m"}},
            }
            if with_trace
            else None
        )


def _traced_pipeline(with_trace=True):
    class P(_Pipeline):
        seen_capture = None

        async def prepare(self, **kw):
            P.seen_capture = kw.get("capture_trace")
            await super().prepare(**kw)
            return _TracedOutcome(with_trace)

    return P()


def _stream(c, headers=None):
    return c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "q"}], "stream": True},
        headers=headers or {},
    )


def test_trace_header_sends_contexts_before_done(monkeypatch):
    main_mod = _client(monkeypatch)
    with TestClient(main_mod.app) as c:
        pipeline = _traced_pipeline()
        c.app.state.pipeline = pipeline
        r = _stream(c, {"X-Meno-Trace": "1"})
    names = [n for n, _ in _events(r.text)]
    assert pipeline.seen_capture is True
    assert "contexts" in names
    contexts = next(p for n, p in _events(r.text) if n == "contexts")["contexts"]
    assert contexts[0] == {"id": "7", "url": "https://nsu.ru/a", "title": "Doc A", "text": "первый", "rank": 1}
    assert contexts[1]["text"] == "второй" and contexts[1]["rank"] == 2 and contexts[1]["url"] == ""
    assert r.text.index("event: contexts") < r.text.index("[DONE]")
    trace = next(p for n, p in _events(r.text) if n == "pipeline_trace")["pipeline_trace"]
    assert trace["system_prompt"] == "sys" and trace["user_prompt"] == "usr"
    assert trace["dialogue_history"] == "hist" and trace["search_queries"] == ["q"]
    assert trace["generation_model"] == "gen-m" and trace["core_model"] == "core-m"
    assert trace["schema_version"] == 2  # RAGU's own trace stays inside


def test_trace_header_without_trace_falls_back_to_context_blocks(monkeypatch):
    main_mod = _client(monkeypatch)
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _traced_pipeline(with_trace=False)
        r = _stream(c, {"X-Meno-Trace": "1"})
    contexts = next(p for n, p in _events(r.text) if n == "contexts")["contexts"]
    assert [x["text"] for x in contexts] == ["первый", "второй"]
    assert all(x["id"] for x in contexts)


def test_no_header_no_contexts(monkeypatch):
    main_mod = _client(monkeypatch)
    with TestClient(main_mod.app) as c:
        pipeline = _traced_pipeline()
        c.app.state.pipeline = pipeline
        r = _stream(c)
    names = [n for n, _ in _events(r.text)]
    assert "contexts" not in names and "pipeline_trace" not in names
    assert not pipeline.seen_capture


def test_unknown_model_answers_contract_error(monkeypatch):
    main_mod = _client(monkeypatch)

    async def unknown(app, requested_model):
        raise ValueError("no such model")

    monkeypatch.setattr(main_mod, "_resolve_runtime", unknown)
    with TestClient(main_mod.app) as c:
        c.app.state.pipeline = _Pipeline()
        r = _stream(c)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "model_not_found" and isinstance(err["message"], str)
    assert "text/event-stream" not in r.headers["content-type"]

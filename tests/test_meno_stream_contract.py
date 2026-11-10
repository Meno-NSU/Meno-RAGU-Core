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

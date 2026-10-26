import hashlib
import json
import multiprocessing
import runpy
import sys
from datetime import UTC, datetime
from multiprocessing.pool import ThreadPool

import httpx
import openai
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from scripts import build_knowledge as build


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "исходники с пробелами" / "data"
    source = data / "source"
    source.mkdir(parents=True)
    (data / "knowledge").mkdir()
    (data / "knowledge" / "abbreviations.json").write_text('{"НГУ": ["Новосибирский государственный университет"]}')

    def document(name):
        return {
            "text": " ".join(f"{name}слово{chr(1072 + i // 20)}{chr(1072 + i % 20)}" for i in range(40)),
            "url": f"https://nsu.ru/{name}",
        }

    write_jsonl(source / "nsu_text_corpus_20260706.jsonl", [document("справочник")])
    write_jsonl(source / "teachers_corpus_20261019.jsonl", [document("преподаватель")])
    write_jsonl(source / "nsu_text_corpus_20250318_validated.jsonl", [document("сайт")])
    timestamp = datetime(2026, 10, 16, 20, tzinfo=UTC).timestamp()
    for index, dates in enumerate(("20260220_20260306", "20260306_20260403", "20260403_20260709", "20260709_20261016")):
        doc = document("новость" + ("альфа", "бета", "гамма", "дельта")[index])
        write_jsonl(
            source / f"filtered_news_{dates}.jsonl", [{"content": doc["text"], "url": doc["url"], "date": timestamp}]
        )
    tokenizer_dir = tmp_path / "local tokenizer"
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    ).save_pretrained(tokenizer_dir)
    return [
        "--input-dir",
        str(data.parent),
        "--output-dir",
        str(tmp_path / "готовая база"),
        "--llm-url",
        "http://graph-llm.test:9022",
        "--tokenizer",
        str(tokenizer_dir),
        "--embedder-tokenizer",
        str(tokenizer_dir),
        "--concurrency",
        "1",
    ]


@pytest.fixture
def services(monkeypatch):
    counts = {"entities": 2, "relations": 1, "chunks": 0}
    calls = []
    broken = {"metadata": False}

    def respond(request):
        calls.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "external-model-alias"}]})
        if path == "/v1/embeddings":
            rows = json.loads(request.content)["input"]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": "BAAI/bge-m3",
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(rows))
                    ],
                },
            )
        if path == "/collections":
            return httpx.Response(200, json={"result": {"collections": []}})
        if path.endswith("/points/count"):
            return httpx.Response(200, json={"result": {"count": counts[path.split("/")[2]]}})
        if path == "/v1/chat/completions":
            content = (
                "invalid JSON"
                if broken["metadata"]
                else json.dumps({"title": "НГУ", "annotation": "Учебная жизнь НГУ."})
            )
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "external-model-alias",
                    "choices": [
                        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    client = httpx.Client(transport=httpx.MockTransport(respond))
    original = openai.OpenAI

    def llm_client(**kwargs):
        return original(**kwargs, http_client=client)

    monkeypatch.setattr(openai, "OpenAI", llm_client)
    import search_utils.search_utils as search

    monkeypatch.setattr(search, "OpenAI", llm_client)
    monkeypatch.setattr(multiprocessing, "Pool", lambda **kwargs: ThreadPool(2))
    monkeypatch.setitem(sys.modules, "vllm", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    yield client, counts, calls, broken
    client.close()


def real_cpu_stages(builder, monkeypatch, counts, executed, *, fail=None):
    def execute(stage, log_path):
        executed.append(stage.name)
        if stage.name == fail:
            raise build.BuildError("injected interruption")
        if stage.name == "10-graph":
            assert stage.args[stage.args.index("--llm") + 1] == "external-model-alias"
            assert stage.args[stage.args.index("--llm_tokenizer") + 1] == builder.args.tokenizer
            assert "--qdrant_location" in stage.args
            corpus = [json.loads(line) for line in stage.inputs[0].read_text().splitlines()]
            texts = [
                doc["doc_full_text"][chunk["start_char"] : chunk["end_char"]]
                for doc in corpus
                for chunk in doc["chunks"]
            ]
            sidecar = {"chunk-" + hashlib.md5(text.encode()).hexdigest(): i for i, text in enumerate(texts)}
            counts["chunks"] = len(sidecar)
            stage.output.mkdir()
            (stage.output / "knowledge_graph.gml").write_text('graph [ node [ id 0 label "НГУ" ] ]')
            (stage.output / "menon_chunk_mapping.json").write_text(json.dumps(sidecar))
            (stage.output / "kv_chunks.json").write_text(
                json.dumps({key: {"content": text} for key, text in zip(sidecar, texts, strict=True)})
            )
            (stage.output / "qdrant_collections.json").write_text(
                json.dumps({"vdb_entity": "entities", "vdb_relation": "relations", "vdb_chunk": "chunks"})
            )
            return
        with monkeypatch.context() as context:
            context.chdir(build.CODE_DIR)
            context.setattr(sys, "argv", stage.command[2:])
            # Exercise the real preprocessing and HTTP adapters, without a listening server.
            runpy.run_path(stage.command[2])["main"]()

    monkeypatch.setattr(builder, "execute", execute)


@pytest.mark.parametrize("no_dates", [False, True])
def test_build_runs_real_corpus_pipeline_and_publishes_backend_bundle(inputs, services, monkeypatch, no_dates):
    client, counts, calls, _ = services
    builder = build.Builder(build.parse_args(inputs + (["--no-news-dates"] if no_dates else [])), http_client=client)
    executed = []
    real_cpu_stages(builder, monkeypatch, counts, executed)
    output = builder.build()
    from meno_rag.stand.resources import _load_documents, _validate_mapping

    documents, missing_scores = _load_documents(output / build.CORPUS_NAME)
    mapping = json.loads((output / "chunk_mapping_to_texts.json").read_text())
    _validate_mapping(mapping, documents)
    assert missing_scores == 0
    assert len(documents) == 7
    news = [doc for doc in documents if doc["content_source"] == "news"]
    assert len(news) == 4
    assert all(("17 октября 2026" in doc["doc_title"]) != no_dates for doc in news)
    assert (output / "knowledge" / "abbreviations.json").is_file()
    assert not builder.bundle.exists()
    manifest = json.loads((output / "build_manifest.json").read_text())
    assert len(manifest["files"]) >= 7
    assert "10-graph" in executed
    assert sum(request.url.path == "/v1/chat/completions" for request in calls) == 7
    repeated = build.Builder(build.parse_args(inputs + (["--no-news-dates"] if no_dates else [])), http_client=client)
    monkeypatch.setattr(repeated, "execute", lambda *args: pytest.fail("Completed build must not execute stages"))
    assert repeated.build() == output
    counts["chunks"] -= 1
    with pytest.raises(build.BuildError, match="Неполная коллекция"):
        repeated.build()


def test_interrupted_build_resumes_verified_stages_and_retries_invalid_metadata(inputs, services, monkeypatch):
    client, counts, _, broken = services
    first = build.Builder(build.parse_args(inputs), http_client=client)
    executed = []
    real_cpu_stages(first, monkeypatch, counts, executed)
    broken["metadata"] = True
    with pytest.raises(RuntimeError, match="Metadata generation failed"):
        first.build()
    assert not first.args.output_dir.exists()
    assert "06-metadata" not in json.loads((first.work / "state.json").read_text())["stages"]
    broken["metadata"] = False
    resumed = build.Builder(build.parse_args(inputs), http_client=client)
    retried = []
    real_cpu_stages(resumed, monkeypatch, counts, retried, fail="10-graph")
    with pytest.raises(build.BuildError, match="injected interruption"):
        resumed.build()
    assert retried == ["06-metadata", "07-quality", "08-news-dates", "09-mapping", "10-graph"]
    assert not resumed.args.output_dir.exists()
    final = build.Builder(build.parse_args(inputs), http_client=client)
    last = []
    real_cpu_stages(final, monkeypatch, counts, last)
    final.build()
    assert last == ["10-graph"]


def test_dry_run_lists_all_news_without_network_or_output(inputs, tmp_path, capsys):
    builder = build.Builder(build.parse_args([*inputs, "--dry-run"]))
    builder.build()
    printed = capsys.readouterr().out
    assert "filtered_news_20260709_20261016.jsonl" in printed
    assert "external-model-alias" not in printed
    assert "--fail_on_error" in printed
    assert not builder.work.exists()
    assert not builder.args.output_dir.exists()


def test_existing_data_is_never_overwritten(inputs, services):
    client, _, _, _ = services
    builder = build.Builder(build.parse_args(inputs), http_client=client)
    builder.args.output_dir.mkdir()
    marker = builder.args.output_dir / "production.json"
    marker.write_text("preserve me")
    with pytest.raises(build.BuildError, match="непустой"):
        builder.build()
    assert marker.read_text() == "preserve me"


def test_changed_source_cannot_resume_old_build(inputs, services, monkeypatch):
    client, counts, _, _ = services
    builder = build.Builder(build.parse_args(inputs), http_client=client)
    real_cpu_stages(builder, monkeypatch, counts, [], fail="04-chunking")
    with pytest.raises(build.BuildError):
        builder.build()
    with builder.sources.base.open("a") as stream:
        stream.write("\n")
    resumed = build.Builder(build.parse_args(inputs), http_client=client)
    monkeypatch.setattr(resumed, "execute", lambda *args: pytest.fail("Changed input must fail before stages"))
    with pytest.raises(build.BuildError, match="входные файлы изменились"):
        resumed.build()


def test_raw_sources_enable_preparation_stages(inputs):
    args = build.parse_args(inputs)
    source = args.input_dir / "data" / "source"
    (source / "nsu_text_corpus_20260706.jsonl").unlink()
    (source / "teachers_corpus_20261019.jsonl").unlink()
    (source.parent / "nsu_main_info.md").write_text("# НГУ\n\n## 1 История\n\nСправочник НГУ.")
    (source / "teachers.json").write_text("[]")
    (source / "teachers_chunks.json").write_text("[]")
    builder = build.Builder(args)
    stages = build.make_plan(builder.args, builder.sources, builder.work)
    assert [stage.name for stage in stages[:3]] == ["01-summary", "02-teachers", "03-combine"]


def test_failed_preflight_has_clear_error_and_leaves_no_output(inputs):
    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(401))) as client:
        builder = build.Builder(build.parse_args(inputs), http_client=client)
        with pytest.raises(build.BuildError, match="401"):
            builder.build()
    assert not builder.work.exists()


def test_subprocess_gets_correct_directory_and_secrets_only_via_environment(inputs, tmp_path, monkeypatch):
    builder = build.Builder(build.parse_args([*inputs, "--llm-key-env", "TEST_GRAPH_KEY"]))
    monkeypatch.setenv("TEST_GRAPH_KEY", "secret-not-in-command")
    monkeypatch.setenv("MENO_INDEX_EMBEDDER_KEY", "embedding-secret")
    monkeypatch.setattr(build, "CODE_DIR", tmp_path)
    script = tmp_path / "worker.py"
    script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "assert os.environ['MENO_INDEX_LLM_KEY'] == 'secret-not-in-command'\n"
        "assert os.environ['MENO_INDEX_EMBEDDER_KEY'] == 'embedding-secret'\n"
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "assert Path.cwd() == Path(__file__).parent\n"
        "print('worker passed')\n"
        "sys.exit(int(sys.argv[1]))\n"
    )
    stage = build.Stage("probe", script.name, ["0"], [], tmp_path / "unused")
    assert "secret-not-in-command" not in " ".join(stage.command)
    log = tmp_path / "worker.log"
    builder.execute(stage, log)
    assert log.read_text() == "worker passed\n"
    stage.args = ["3"]
    with pytest.raises(build.BuildError, match="кодом 3"):
        builder.execute(stage, log)


@pytest.mark.parametrize(
    "configured, served, expected",
    [
        (None, 32768, 32768),
        (None, None, 16384),
        (8192, 32768, 8192),
        (65536, 32768, "API сообщает максимум"),
        (None, 4096, "уменьшите предел ответа"),
    ],
)
def test_preflight_uses_actual_served_context_window(inputs, monkeypatch, configured, served, expected):
    arguments = [*inputs, "--llm-model", "gemma-4-12B-it"]
    if configured is not None:
        arguments += ["--llm-context-tokens", str(configured)]
    builder = build.Builder(build.parse_args(arguments))

    def request(method, url, **kwargs):
        if url.endswith("/models"):
            return {"data": [{"id": "gemma-4-12B-it", "max_model_len": served}]}
        if url.endswith("/embeddings"):
            return {"data": [{"embedding": [1, 0]}]}
        return {}

    monkeypatch.setattr(builder, "request", request)
    if isinstance(expected, str):
        with pytest.raises(build.BuildError, match=expected):
            builder.preflight()
    else:
        builder.preflight()
        assert builder.args.llm_context_tokens == expected

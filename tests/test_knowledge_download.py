import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest

from scripts import build_knowledge as build

PUBLIC_URL = "https://disk.yandex.ru/d/XZjLjlah72eG1w"


@pytest.fixture
def remote_files():
    names = [
        "source/nsu_text_corpus_20260706.jsonl",
        "source/teachers_corpus_20261019.jsonl",
        "source/nsu_text_corpus_20250318_validated.jsonl",
        "source/filtered_news_20260220_20260306.jsonl",
        "source/filtered_news_20260306_20260403.jsonl",
        "source/filtered_news_20260403_20260709.jsonl",
        "source/filtered_news_20260709_20261016.jsonl",
        "knowledge/abbreviations.json",
        "source/teachers.json",
        "source/teachers_chunks.json",
        "source/nsu_text_corpus_20250318.jsonl",
        "nsu_main_info.md",
    ]
    return {"/data/" + name: (json.dumps({"source": name}) + "\n").encode() for name in names}


def remote_handler(files, calls, *, data_folder=True, broken=None, corrupt=None):
    original = dict(files)

    def folder(path):
        children = {}
        for name, content in original.items():
            if name.startswith(path.rstrip("/") + "/"):
                suffix = name[len(path.rstrip("/")) + 1 :]
                child = suffix.split("/", 1)[0]
                children[child] = (
                    {"name": child, "type": "dir"}
                    if "/" in suffix
                    else {
                        "name": child,
                        "type": "file",
                        "size": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                )
        return list(children.values())

    class InterruptedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x" * (1024 * 1024)
            raise httpx.ReadError("interrupted download")

    def respond(request):
        calls.append(request)
        if request.url.host == "files.test":
            name = request.url.params["path"]
            if broken and name in broken:
                return httpx.Response(200, stream=InterruptedStream())
            content = files[name]
            if corrupt and name in corrupt:
                content = b"x" * len(content)
            return httpx.Response(200, content=content)
        assert request.url.host == "cloud-api.yandex.net"
        assert request.url.params["public_key"] == PUBLIC_URL
        path = request.url.params["path"]
        effective = path if data_folder else "/data" + (path if path != "/" else "")
        if request.url.path.endswith("/download"):
            return httpx.Response(
                200, json={"href": str(httpx.URL("https://files.test/file", params={"path": effective}))}
            )
        children = folder(effective)
        offset = int(request.url.params["offset"])
        # Small pages exercise pagination even for this tiny dataset.
        return httpx.Response(
            200,
            json={
                "type": "dir",
                "_embedded": {
                    "total": len(children),
                    "items": children[offset : offset + 2],
                },
            },
        )

    return respond


@pytest.mark.parametrize("data_folder", [True, False])
def test_downloads_required_files_only_and_skips_verified_files_offline(tmp_path, remote_files, data_folder):
    calls = []
    with httpx.Client(
        transport=httpx.MockTransport(remote_handler(remote_files, calls, data_folder=data_folder))
    ) as client:
        downloader = build.YandexSource(PUBLIC_URL, tmp_path / "данные с пробелами", client)
        downloader.run()
        assert len(json.loads((downloader.root / ".yandex-source.json").read_text())["files"]) == 8
        downloaded = [
            str(path.relative_to(downloader.root)) for path in downloader.root.rglob("*") if path.suffix == ".jsonl"
        ]
        assert len(downloaded) == 7
        assert "source/filtered_news_20260709_20261016.jsonl" in downloaded
        assert not (downloader.root / "source" / "nsu_text_corpus_20250318.jsonl").exists()
        assert not (downloader.root / "source" / "teachers.json").exists()
        assert not list(downloader.root.rglob("*.part"))
        downloads_before = len(calls)
        downloader.run()
        assert len(calls) == downloads_before
        for item in json.loads((downloader.root / ".yandex-source.json").read_text())["files"]:
            assert build.digest(downloader.root / item["local"]) == item["sha256"]


def test_network_interruption_never_publishes_partial_file_and_restarts_only_unfinished(
    tmp_path, remote_files, monkeypatch
):
    name = "/data/source/nsu_text_corpus_20250318_validated.jsonl"
    remote_files[name] = b"a" * (2 * 1024 * 1024)
    broken = {name}
    calls = []
    monkeypatch.setattr(build.time, "sleep", lambda duration: None)
    with httpx.Client(transport=httpx.MockTransport(remote_handler(remote_files, calls, broken=broken))) as client:
        downloader = build.YandexSource(PUBLIC_URL, tmp_path / "data", client)
        target = downloader.root / "source" / Path(name).name
        with pytest.raises(build.BuildError, match="Не удалось скачать"):
            downloader.run()
        assert not target.exists()
        assert target.with_name(target.name + ".part").stat().st_size == 1024 * 1024
        assert (downloader.root / "source" / "nsu_text_corpus_20260706.jsonl").exists()
        broken.clear()
        downloader.run()
        assert target.read_bytes() == remote_files[name]
        counts = Counter(request.url.params["path"] for request in calls if request.url.host == "files.test")
        assert counts[name] == 4
        assert counts["/data/source/nsu_text_corpus_20260706.jsonl"] == 1
        assert not list(downloader.root.rglob("*.part"))


def test_checksum_failure_leaves_no_ready_file_and_user_files_are_not_overwritten(tmp_path, remote_files):
    name = "/data/source/nsu_text_corpus_20260706.jsonl"
    corrupt = {name}
    with httpx.Client(transport=httpx.MockTransport(remote_handler(remote_files, [], corrupt=corrupt))) as client:
        downloader = build.YandexSource(PUBLIC_URL, tmp_path / "data", client)
        target = downloader.root / "source" / Path(name).name
        with pytest.raises(build.BuildError, match="SHA-256"):
            downloader.run()
        assert not target.exists()
        corrupt.clear()
        downloader.run()
        target.write_bytes(b"user modifications")
        with pytest.raises(build.BuildError, match="Локальный файл отличается"):
            downloader.run()
        assert target.read_bytes() == b"user modifications"


def test_raw_preparation_files_are_downloaded_when_prepared_corpora_are_missing(tmp_path, remote_files):
    remote_files.pop("/data/source/nsu_text_corpus_20260706.jsonl")
    remote_files.pop("/data/source/teachers_corpus_20261019.jsonl")
    with httpx.Client(transport=httpx.MockTransport(remote_handler(remote_files, []))) as client:
        downloader = build.YandexSource(PUBLIC_URL, tmp_path / "data", client)
        downloader.run()
    assert (downloader.root / "nsu_main_info.md").exists()
    assert (downloader.root / "source" / "teachers.json").exists()
    assert (downloader.root / "source" / "teachers_chunks.json").exists()


def arguments(tmp_path):
    return [
        "--input-dir",
        str(tmp_path / "missing data"),
        "--output-dir",
        str(tmp_path / "output"),
        "--llm-url",
        "http://llm.test:9022",
        "--tokenizer",
        "unused-tokenizer",
        "--source-url",
        PUBLIC_URL,
    ]


def test_builder_downloads_before_source_discovery_and_model_preflight(tmp_path, remote_files, monkeypatch):
    with httpx.Client(transport=httpx.MockTransport(remote_handler(remote_files, []))) as client:
        builder = build.Builder(build.parse_args(arguments(tmp_path)), http_client=client)

        def preflight():
            assert len(builder.sources.news) == 4
            assert all(path.is_file() for path in builder.sources.files())
            raise build.BuildError("models not started for this test")

        monkeypatch.setattr(builder, "preflight", preflight)
        with pytest.raises(build.BuildError, match="models not started"):
            builder.build()
        assert not builder.args.output_dir.exists()


def test_dry_run_can_plan_download_into_missing_directory_without_network(tmp_path, capsys):
    def no_network(request):
        pytest.fail(f"Dry run made an HTTP request: {request.url}")

    with httpx.Client(transport=httpx.MockTransport(no_network)) as client:
        builder = build.Builder(build.parse_args([*arguments(tmp_path), "--dry-run"]), http_client=client)
        builder.build()
    assert "00-download" in capsys.readouterr().out
    assert not builder.args.input_dir.exists()
    assert not builder.work.exists()


def test_missing_required_remote_file_fails_before_downloading_payloads(tmp_path, remote_files):
    remote_files.pop("/data/source/nsu_text_corpus_20250318_validated.jsonl")
    calls = []
    with httpx.Client(transport=httpx.MockTransport(remote_handler(remote_files, calls))) as client:
        downloader = build.YandexSource(PUBLIC_URL, tmp_path / "data", client)
        with pytest.raises(build.BuildError, match="отсутствует"):
            downloader.run()
    assert not any(request.url.host == "files.test" for request in calls)

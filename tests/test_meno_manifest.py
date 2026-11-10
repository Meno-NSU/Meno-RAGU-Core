import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

_CONTRACT_SRC = Path.home() / "Meno-RAG-Core-gateway" / "packages" / "meno-contract" / "src"


def _assert_valid_manifest(body):
    if not _CONTRACT_SRC.is_dir():
        pytest.skip("Meno-RAG-Core-gateway checkout not found; contract validation skipped")
    before = set(sys.modules)
    sys.path.insert(0, str(_CONTRACT_SRC))
    try:
        from meno_contract.models import Manifest

        Manifest.model_validate(body)
    finally:
        sys.path.remove(str(_CONTRACT_SRC))
        for name in set(sys.modules) - before:
            if name == "meno_contract" or name.startswith("meno_contract."):
                del sys.modules[name]


def _build_client(monkeypatch, **env):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api.main import app

    return app


@pytest.fixture(autouse=True)
def _fresh_settings_cache():
    yield
    from meno_rag.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    app = _build_client(monkeypatch)
    with TestClient(app) as c:
        app.state.vllm_registry.list_models = AsyncMock(
            return_value=[
                {"id": "gemma-4-12B-it", "endpoint": "http://v", "created": 1, "object": "model", "owned_by": "vllm"},
                {"id": "other-model", "endpoint": "http://v", "created": 1, "object": "model", "owned_by": "vllm"},
            ]
        )
        app.state.openrouter_registry = None
        yield c


def test_the_manifest_follows_the_contract(client):
    r = client.get("/meno/v1/manifest")
    assert r.status_code == 200
    body = r.json()

    assert body["protocol"] == "meno-backend/1"
    assert body["name"] == "RAGU"
    assert body["version"] == "nsu-ragu-2026-11"
    assert body["kb"] == "nsu-ragu"
    assert body["stages"] == {"prepare": {"ru": "Подготовка", "en": "Preparation"}}

    model_ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert [m["id"] for m in body["llms"]] == model_ids
    names = {m["id"]: m["name"] for m in body["llms"]}
    assert names["gemma-4-12B-it"] == "Gemma 4 12B"
    assert names["other-model"] == "other-model"
    assert body["default_llm"] in model_ids

    _assert_valid_manifest(body)


def test_the_build_is_omitted_when_unset(client):
    assert "build" not in client.get("/meno/v1/manifest").json()


def test_the_build_is_reported_when_set(monkeypatch):
    app = _build_client(monkeypatch, MENO_BACKEND_BUILD="abc123")
    with TestClient(app) as c:
        assert c.get("/meno/v1/manifest").json()["build"] == "abc123"


def test_a_bad_version_setting_fails_at_startup(monkeypatch):
    from meno_rag.config import Settings

    monkeypatch.setenv("MENO_BACKEND_VERSION", "bad version!")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def _manifest_for(client, models):
    client.app.state.vllm_registry.list_models = AsyncMock(
        return_value=[{"endpoint": "http://v", "created": 1, "object": "model", "owned_by": "vllm", **m} for m in models]
    )
    r = client.get("/meno/v1/manifest")
    assert r.status_code == 200
    return r.json()


def test_duplicate_ids_give_one_llm(client):
    body = _manifest_for(client, [{"id": "gemma-4-12B-it"}, {"id": "gemma-4-12B-it", "endpoint": "http://w"}])
    assert [m["id"] for m in body["llms"]] == ["gemma-4-12B-it"]
    _assert_valid_manifest(body)


def test_at_most_32_llms_all_vllm(client):
    body = _manifest_for(client, [{"id": f"v{i}"} for i in range(40)])
    ids = [m["id"] for m in body["llms"]]
    assert ids == [f"v{i}" for i in range(32)]
    _assert_valid_manifest(body)


def test_openrouter_models_are_never_listed_even_with_auth_off(client):
    assert not client.app.state.settings.auth_enabled
    client.app.state.openrouter_registry = AsyncMock()
    client.app.state.openrouter_registry.list_models = AsyncMock(
        return_value=[{"id": f"or/m{i}", "display_name": f"OR {i}"} for i in range(5)]
    )
    body = _manifest_for(client, [{"id": "v0"}])
    assert [m["id"] for m in body["llms"]] == ["v0"]
    _assert_valid_manifest(body)


def test_the_few_shot_stage_is_labelled_when_enabled(monkeypatch):
    app = _build_client(monkeypatch, QA_FEWSHOTS_ENABLED="true")
    with TestClient(app) as c:
        stages = c.get("/meno/v1/manifest").json()["stages"]
    assert stages["fewshot_selection"] == {"ru": "Подбор примеров", "en": "Few-shot selection"}
    assert stages["prepare"] == {"ru": "Подготовка", "en": "Preparation"}


def test_an_id_with_a_space_is_skipped(client):
    body = _manifest_for(client, [{"id": "bad id"}, {"id": "good"}])
    assert [m["id"] for m in body["llms"]] == ["good"]
    assert body["default_llm"] == "good"
    _assert_valid_manifest(body)


def test_a_long_display_name_is_cut_to_128(client, monkeypatch):
    monkeypatch.setattr(client.app.state.settings, "llm_display_names", "long=" + "x" * 200)
    body = _manifest_for(client, [{"id": "long"}])
    assert len(body["llms"][0]["name"]) == 128
    _assert_valid_manifest(body)

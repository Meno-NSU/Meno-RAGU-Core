import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

_CONTRACT_SRC = Path.home() / "Meno-RAG-Core-gateway" / "packages" / "meno-contract" / "src"


def _build_client(monkeypatch, **env):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from meno_rag.config import get_settings

    get_settings.cache_clear()
    from meno_rag.api.main import app

    return app


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
    assert body["stages"] == {}

    model_ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert [m["id"] for m in body["llms"]] == model_ids
    names = {m["id"]: m["name"] for m in body["llms"]}
    assert names["gemma-4-12B-it"] == "Gemma 4 12B"
    assert names["other-model"] == "other-model"
    assert body["default_llm"] in model_ids

    if _CONTRACT_SRC.is_dir():
        sys.path.insert(0, str(_CONTRACT_SRC))
        try:
            from meno_contract.models import Manifest
        finally:
            sys.path.remove(str(_CONTRACT_SRC))
        Manifest.model_validate(body)


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

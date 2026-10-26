import runpy
import sys
from pathlib import Path

from meno_rag.config import Settings


def test_index_runner_uses_code_cwd_and_keeps_keys_out_of_argv(monkeypatch):
    import subprocess

    settings = Settings(
        _env_file=None,
        openai_api_key="llm-secret",
        embedder_api_key="embedding-secret",
        vllm_endpoints="http://llm:8000/v1",
    )
    monkeypatch.setattr("meno_rag.config.Settings", lambda: settings)
    seen = {}

    def call(command, **kwargs):
        seen.update(command=command, **kwargs)
        return 0

    monkeypatch.setattr(subprocess, "call", call)
    monkeypatch.setattr(sys, "argv", ["index_graph.py", "--chunk_header", "title"])
    script = Path(__file__).resolve().parents[1] / "scripts/index_graph.py"
    namespace = runpy.run_path(str(script))
    assert namespace["main"]() == 0
    assert seen["cwd"].name == "code"
    assert seen["env"]["MENO_INDEX_LLM_KEY"] == "llm-secret"
    assert seen["env"]["MENO_INDEX_EMBEDDER_KEY"] == "embedding-secret"
    assert "llm-secret" not in " ".join(seen["command"])
    assert "embedding-secret" not in " ".join(seen["command"])
    assert "http://llm:8000/v1" in seen["command"]
    assert seen["command"][-2:] == ["--chunk_header", "title"]

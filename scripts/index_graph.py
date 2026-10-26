#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

from meno_rag.config import Settings


def main():
    settings = Settings()
    if not settings.vllm_endpoint_list:
        raise SystemExit("VLLM_ENDPOINTS must contain the indexing LLM endpoint.")
    code_dir = Path(__file__).resolve().parents[1] / "code"
    command = [
        sys.executable,
        "prepare_graph_index.py",
        "-i",
        str(settings.corpus_path.resolve()),
        "-r",
        str(settings.chunk_mapping_path.resolve()),
        "-o",
        str(settings.graph_dir.resolve()),
        "-a",
        str(settings.abbreviations_path.resolve()),
        "-e",
        settings.embedder_model,
        "--embedder_url",
        settings.embedder_url,
        "--llm",
        settings.rag_rewrite_rerank_model or settings.default_model or "bond005/meno-lite-0.2",
        "--url",
        settings.vllm_endpoint_list[0].rstrip("/").removesuffix("/v1") + "/v1",
        "--qdrant_location",
        settings.qdrant_url,
        *sys.argv[1:],
    ]
    environment = dict(
        os.environ, MENO_INDEX_LLM_KEY=settings.openai_api_key, MENO_INDEX_EMBEDDER_KEY=settings.embedder_api_key
    )
    return subprocess.call(command, cwd=code_dir, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())

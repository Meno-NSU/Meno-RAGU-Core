from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "vendor" / "tiktoken" / "o200k_base.tiktoken"
ASSET_URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
ASSET_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"


def test_vendored_gpt4o_encoding_matches_published_tiktoken_hash():
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == ASSET_SHA256
    assert hashlib.sha1(ASSET_URL.encode()).hexdigest() == CACHE_KEY


def test_ragu_search_tokenizer_initializes_from_readonly_cache_without_network(tmp_path):
    cache = tmp_path / "tiktoken-cache"
    cache.mkdir()
    vocabulary = cache / CACHE_KEY
    shutil.copyfile(ASSET, vocabulary)
    vocabulary.chmod(0o444)
    cache.chmod(0o555)
    script = r'''
import json
from unittest.mock import patch

with patch("tiktoken.load.read_file", side_effect=AssertionError("Unexpected tokenizer download")), \
     patch("socket.socket.connect", side_effect=AssertionError("Unexpected network connection")):
    from ragu.utils.token_truncation import TokenTruncation

    tokenizer = TokenTruncation(model_id="gpt-4o", tokenizer_type="tiktoken", max_tokens=8)
    text = "Какие факультеты есть в НГУ?"
    tokens = tokenizer.encoder.encode(text)
    assert tokenizer.encoder.name == "o200k_base"
    assert tokenizer.encoder.decode(tokens) == text
    truncated = tokenizer(text * 20)
    assert len(tokenizer.encoder.encode(truncated)) <= 8
    print(json.dumps({"tokens": tokens, "truncated": truncated}, ensure_ascii=False))
'''
    environment = {**os.environ, "TIKTOKEN_CACHE_DIR": str(cache), "TOKENIZERS_PARALLELISM": "false"}
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=tmp_path, env=environment,
            capture_output=True, text=True, check=True, timeout=60,
        )
    finally:
        cache.chmod(0o755)
        vocabulary.chmod(0o644)
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["tokens"] == [138228, 13298, 119090, 33910, 17199, 743, 4349, 175925, 30]
    assert data["truncated"]
    assert list(cache.iterdir()) == [vocabulary]


def test_ragu_tokenizer_rejects_corrupt_offline_vocabulary(tmp_path):
    cache = tmp_path / "tiktoken-cache"
    cache.mkdir()
    (cache / CACHE_KEY).write_bytes(ASSET.read_bytes()[:1024])
    script = r'''
from unittest.mock import patch

with patch("tiktoken.load.read_file", side_effect=RuntimeError("Offline vocabulary is invalid")):
    from ragu.utils.token_truncation import TokenTruncation

    try:
        TokenTruncation(model_id="gpt-4o", tokenizer_type="tiktoken")
    except ValueError as error:
        assert "Offline vocabulary is invalid" in str(error)
    else:
        raise AssertionError("A corrupt vocabulary must never initialize the search tokenizer")
'''
    environment = {**os.environ, "TIKTOKEN_CACHE_DIR": str(cache), "TOKENIZERS_PARALLELISM": "false"}
    subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment,
        capture_output=True, text=True, check=True, timeout=60,
    )

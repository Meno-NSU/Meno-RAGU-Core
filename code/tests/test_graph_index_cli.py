import runpy
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest


@pytest.mark.parametrize('explicit', [False, True])
def test_indexing_keys_come_from_environment_and_cli_has_priority(monkeypatch, explicit):
    monkeypatch.setenv('MENO_INDEX_LLM_KEY', 'environment-llm-key')
    monkeypatch.setenv('MENO_INDEX_EMBEDDER_KEY', 'environment-embedding-key')
    arguments = ['prepare_graph_index.py', '-i', 'corpus.jsonl', '-r', 'mapping.json', '-e', 'bge-m3', '--vector_only']
    if explicit:
        arguments += ['--key', 'cli-llm-key', '--embedder_key', 'cli-embedding-key']
    monkeypatch.setattr(sys, 'argv', arguments)
    original = ArgumentParser.parse_args
    parsed = []

    class Parsed(Exception):
        pass

    def capture(parser, *args, **kwargs):
        parsed.append(original(parser, *args, **kwargs))
        raise Parsed

    monkeypatch.setattr(ArgumentParser, 'parse_args', capture)
    script = Path(__file__).resolve().parents[1] / 'prepare_graph_index.py'
    with pytest.raises(Parsed):
        runpy.run_path(str(script), run_name='__main__')
    assert parsed[0].openai_api_key == ('cli-llm-key' if explicit else 'environment-llm-key')
    assert parsed[0].embedder_api_key == ('cli-embedding-key' if explicit else 'environment-embedding-key')

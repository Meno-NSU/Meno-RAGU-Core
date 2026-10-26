import runpy
import sys
from argparse import ArgumentParser
from pathlib import Path

import pytest
from unittest.mock import AsyncMock

from prepare_graph_index import ContextBudgetLLM, configure_logging, logger


@pytest.mark.parametrize('level, debug_visible', [('INFO', False), ('DEBUG', True)])
def test_graph_logging_honors_requested_level_without_ansi_sequences(capsys, level, debug_visible):
    try:
        configure_logging(level)
        logger.debug('graph-debug-probe')
        logger.info('graph-info-probe')
        logger.complete()
        output = capsys.readouterr().out
        assert ('graph-debug-probe' in output) == debug_visible
        assert 'graph-info-probe' in output
        assert '\x1b[' not in output
    finally:
        with capsys.disabled():
            configure_logging('INFO')


@pytest.mark.parametrize('explicit', [False, True])
def test_indexing_keys_come_from_environment_and_cli_has_priority(monkeypatch, explicit):
    monkeypatch.setenv('MENO_INDEX_LLM_KEY', 'environment-llm-key')
    monkeypatch.setenv('MENO_INDEX_EMBEDDER_KEY', 'environment-embedding-key')
    monkeypatch.setenv('QDRANT_API_KEY', 'environment-qdrant-key')
    arguments = ['prepare_graph_index.py', '-i', 'corpus.jsonl', '-r', 'mapping.json', '-e', 'bge-m3', '--vector_only']
    arguments += ['--llm_tokenizer', '/models/graph-tokenizer', '--embedder_tokenizer', '/models/bge-m3',
                  '--llm_context_tokens', '32768', '--embedder_tokens', '4096', '--log_level', 'DEBUG']
    if explicit:
        arguments += ['--key', 'cli-llm-key', '--embedder_key', 'cli-embedding-key', '--qdrant_key', 'cli-qdrant-key']
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
    assert parsed[0].qdrant_key == ('cli-qdrant-key' if explicit else 'environment-qdrant-key')
    assert parsed[0].llm_tokenizer == '/models/graph-tokenizer'
    assert parsed[0].embedder_tokenizer == '/models/bge-m3'
    assert parsed[0].llm_context_tokens == 32768
    assert parsed[0].embedder_tokens == 4096
    assert parsed[0].log_level == 'DEBUG'


@pytest.mark.parametrize('prompt_tokens, allowed', [(7, True), (8, False)])
def test_graph_llm_checks_context_including_completion_budget(prompt_tokens, allowed):
    import asyncio

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [0] * prompt_tokens

    client = AsyncMock()
    client.chat_completion.return_value = 'response'
    llm = ContextBudgetLLM(client=client, model_name='served-alias', tokenizer=Tokenizer(),
                           context_tokens=10, completion_tokens=3)
    conversation = [{'role': 'user', 'content': 'query'}]
    if allowed:
        assert asyncio.run(llm.chat_completion(conversation)) == 'response'
        assert llm.overflow_count == 0
        client.chat_completion.assert_awaited_once()
    else:
        with pytest.raises(ValueError, match='Increase --llm_context_tokens'):
            asyncio.run(llm.chat_completion(conversation))
        assert llm.overflow_count == 1
        client.chat_completion.assert_not_awaited()

import io
import json

from graph_utils import llm_limits
from graph_utils.llm_limits import effective_context_tokens, served_context_tokens


def test_the_served_context_comes_from_the_models_list(monkeypatch):
    seen = {}

    def urlopen(request, timeout):
        seen['url'], seen['auth'] = request.full_url, request.headers.get('Authorization')
        return io.BytesIO(json.dumps({'data': [{'id': 'other', 'max_model_len': 8192},
                                               {'id': 'gemma-4-12B-it', 'max_model_len': 32000}]}).encode())

    monkeypatch.setattr(llm_limits.urllib.request, 'urlopen', urlopen)
    assert served_context_tokens('http://everest:9022/v1/', 'gemma-4-12B-it', 'EMPTY') == 32000
    assert seen == {'url': 'http://everest:9022/v1/models', 'auth': 'Bearer EMPTY'}


def test_an_unknown_model_or_an_unreachable_server_gives_no_limit(monkeypatch):
    monkeypatch.setattr(llm_limits.urllib.request, 'urlopen',
                        lambda request, timeout: io.BytesIO(b'{"data": [{"id": "other", "max_model_len": 8192}]}'))
    assert served_context_tokens('http://x/v1', 'gemma-4-12B-it', None) is None

    def fail(request, timeout):
        raise OSError('connection refused')

    monkeypatch.setattr(llm_limits.urllib.request, 'urlopen', fail)
    assert served_context_tokens('http://x/v1', 'gemma-4-12B-it', None) is None


def test_the_build_never_plans_more_context_than_the_server_serves():
    assert effective_context_tokens(60000, 32000) == 32000
    assert effective_context_tokens(16000, 32000) == 16000
    assert effective_context_tokens(60000, None) == 60000

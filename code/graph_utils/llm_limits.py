"""The context the served model actually accepts.

--llm_context_tokens is what the build plans prompts for; the summaries pack long descriptions up to it. When it
exceeds the model's max_model_len (graph v2: 60000 planned, gemma served 32000), the first long description fails
with HTTP 400 and the build stops. The build asks the server and plans for the smaller of the two.
"""
import json
import urllib.request


def served_context_tokens(base_url: str, model: str, api_key: str | None, timeout: float = 10.0) -> int | None:
    """max_model_len of the model in the OpenAI-compatible /models list (vLLM reports it), or None."""
    request = urllib.request.Request(base_url.rstrip('/') + '/models')
    if api_key:
        request.add_header('Authorization', f'Bearer {api_key}')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            models = json.load(response).get('data') or []
    except (OSError, ValueError):
        return None
    for item in models:
        if item.get('id') == model and isinstance(item.get('max_model_len'), int):
            return item['max_model_len']
    return None


def effective_context_tokens(configured: int, served: int | None) -> int:
    return min(configured, served) if served else configured

import httpx
import pytest

from meno_rag.llm.client import VLLMClient
from meno_rag.llm.registry import VLLMRegistry


@pytest.mark.asyncio
async def test_private_model_discovery_and_completion_use_endpoint_key():
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers.get("Authorization")))
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": request.url.host}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        keys = {"http://everest:9022": "remote-secret"}
        registry = VLLMRegistry(
            ["http://llm:8000/v1", "http://everest:9022"],
            http_client=http,
            api_key="local-secret",
            endpoint_api_keys=keys,
        )
        assert len(await registry.discover()) == 2
        client = VLLMClient(http_client=http, api_key="local-secret", endpoint_api_keys=keys)
        await client.chat_completion(base_url="http://everest:9022/v1", model="remote", messages=[])
    assert seen == [
        ("llm", "Bearer local-secret"),
        ("everest", "Bearer remote-secret"),
        ("everest", "Bearer remote-secret"),
    ]

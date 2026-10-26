import asyncio
from types import SimpleNamespace

import pytest

from meno_rag.config import Settings
from meno_rag.stand.resources import GraphRuntime


@pytest.mark.asyncio
async def test_graph_runtime_owns_loop_allows_concurrency_and_closes(monkeypatch):
    loops = []
    state = {"active": 0, "maximum": 0, "closed": False}

    async def search(**kwargs):
        loops.append(asyncio.get_running_loop())
        state["active"] += 1
        state["maximum"] = max(state["maximum"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return {"query": kwargs["search_query"]}

    def load(self):
        self.engines = SimpleNamespace(event_loop=asyncio.new_event_loop(), asearch=search)

    async def close(self):
        state["closed"] = True

    monkeypatch.setattr(GraphRuntime, "_load", load)
    monkeypatch.setattr(GraphRuntime, "_close_clients", close)
    runtime = GraphRuntime(Settings(_env_file=None))
    results = await asyncio.gather(runtime.search("a"), runtime.search("b"))
    await runtime.aclose()
    assert results == [{"query": "a"}, {"query": "b"}]
    assert loops[0] is loops[1] and loops[0] is not asyncio.get_running_loop()
    assert state["maximum"] == 2 and state["closed"]
    assert loops[0].is_closed()


@pytest.mark.asyncio
async def test_graph_timeout_cancels_search_on_its_owner_loop(monkeypatch):
    state = {"cancelled": False}

    async def search(**kwargs):
        try:
            await asyncio.sleep(10)
        finally:
            state["cancelled"] = True

    def load(self):
        self.engines = SimpleNamespace(event_loop=asyncio.new_event_loop(), asearch=search)

    async def close(self):
        await asyncio.sleep(0)

    monkeypatch.setattr(GraphRuntime, "_load", load)
    monkeypatch.setattr(GraphRuntime, "_close_clients", close)
    runtime = GraphRuntime(Settings(_env_file=None, graph_timeout_seconds=0.01))
    with pytest.raises(TimeoutError):
        await runtime.search("slow")
    await runtime.aclose()
    assert state["cancelled"]

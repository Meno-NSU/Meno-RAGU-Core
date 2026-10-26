import asyncio

from nltk.stem.snowball import SnowballStemmer

from meno_rag.config import Settings
from meno_rag.stand.ragu_pipeline import RaguRagPipeline
from meno_rag.stand.resources import StandResources


class Graph:
    async def search(self, query):
        await asyncio.sleep(0)
        return {
            "chunks": [(0, 0.5)],
            "entities": [
                {
                    "key": query,
                    "kind": "entity",
                    "name": query,
                    "type": "ORG",
                    "description": f"Факт о {query}",
                    "score": 1.0,
                    "sources": [0],
                }
            ],
            "relations": [
                {
                    "key": query,
                    "kind": "relation",
                    "subject": query,
                    "object": "НГУ",
                    "relation_type": "часть",
                    "description": "Связь",
                    "score": 0.7,
                    "sources": [0],
                }
            ],
        }


class LLM:
    def __init__(self):
        self.prompts = []

    async def chat_completion_text(self, **kwargs):
        return "поисковый запрос"

    async def chat_completion(self, **kwargs):
        self.prompts.append(kwargs["messages"])
        return {
            "choices": [
                {
                    "message": {"content": "2"},
                    "logprobs": {
                        "content": [{"token": "2", "logprob": 0.0, "top_logprobs": [{"token": "2", "logprob": 0.0}]}]
                    },
                }
            ]
        }


def pipeline(**overrides):
    text = "Факультет НГУ."
    resources = StandResources(
        documents=[
            {
                "doc_full_text": text,
                "doc_title": "НГУ",
                "doc_annotation": "",
                "url": ["https://example.org/a", "https://example.org/b"],
                "quality_score": 1.0,
                "chunks": [{"start_char": 0, "end_char": len(text)}],
            }
        ],
        chunk_mapping={"0": {"doc_index": 0, "local_chunk_index": 0}},
        graph=Graph(),
        stemmer=SnowballStemmer("russian"),
        abbreviations={},
    )
    return RaguRagPipeline(
        settings=Settings(_env_file=None, max_rewrite_queries=1, **overrides),
        resources=resources,
        llm_router=LLM(),
        rewrite_semaphore=asyncio.Semaphore(2),
        rerank_semaphore=asyncio.Semaphore(2),
        generation_semaphore=asyncio.Semaphore(2),
        embed_semaphore=asyncio.Semaphore(2),
    )

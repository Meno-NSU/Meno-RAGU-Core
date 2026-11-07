"""Lazy HTTP-only RAGAS metrics with isolated response caches and call traces."""

from __future__ import annotations

import contextvars
import hashlib
import importlib.metadata
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

RAGAS_VERSION = "0.4.3"
ADAPTER_VERSION = 1
METRIC_NAMES = (
    "faithfulness",
    "factual_correctness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
_CALL_RECORDS: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar(
    "ragas_benchmark_call_records", default=None
)


@dataclass(frozen=True, slots=True)
class RagasConfig:
    judge_url: str
    judge_model: str
    judge_revision: str
    embedder_url: str
    embedder_model: str
    cache_dir: Path
    judge_api_key: str = field(default="EMPTY", repr=False)
    embedder_api_key: str = field(default="EMPTY", repr=False)
    max_tokens: int = 4096
    temperature: float = 0.0
    timeout: float = 180.0
    disable_thinking: bool = True
    judge_context_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("judge_url", "embedder_url"):
            value = getattr(self, name)
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{name} must be an absolute HTTP(S) API URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(f"{name} must not contain credentials, query parameters or fragments")
        for name in ("judge_model", "judge_revision", "embedder_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int) or self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if self.judge_context_tokens is not None and (
            type(self.judge_context_tokens) is not int or self.judge_context_tokens < 1
        ):
            raise ValueError("judge_context_tokens must be a positive integer")

    def fingerprint_payload(self) -> dict[str, Any]:
        return fingerprint_payload(self)


def _judge_payload(config: RagasConfig) -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "ragas_version": RAGAS_VERSION,
        "url": config.judge_url.rstrip("/"),
        "model": config.judge_model,
        "revision": config.judge_revision,
        "temperature": config.temperature,
        "top_p": 1.0,
        "max_tokens": config.max_tokens,
        "disable_thinking": config.disable_thinking,
        "instructor_max_retries": 2,
    }


def _embedding_payload(config: RagasConfig) -> dict[str, Any]:
    return {
        "adapter_version": ADAPTER_VERSION,
        "ragas_version": RAGAS_VERSION,
        "url": config.embedder_url.rstrip("/"),
        "model": config.embedder_model,
    }


def fingerprint_payload(config: RagasConfig) -> dict[str, Any]:
    """Return content-affecting settings without secrets or execution settings."""
    return {
        "judge": _judge_payload(config),
        "embeddings": _embedding_payload(config),
        "metrics": {
            "faithfulness": {},
            "factual_correctness": {"mode": "f1", "atomicity": "low", "coverage": "low", "beta": 1.0},
            "answer_relevancy": {"strictness": 3},
            "context_precision": {"implementation": "ContextPrecisionWithReference"},
            "context_recall": {},
        },
    }


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load_components() -> dict[str, Any]:
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    installed_version = importlib.metadata.version("ragas")
    if installed_version != RAGAS_VERSION:
        raise RuntimeError(f"Expected ragas=={RAGAS_VERSION}, found {installed_version}")
    import httpx
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend, cacher
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        FactualCorrectness,
        Faithfulness,
    )

    return {
        "http_client": httpx.AsyncClient,
        "openai_client": AsyncOpenAI,
        "cache": DiskCacheBackend,
        "cacher": cacher,
        "llm_factory": llm_factory,
        "embedding_factory": embedding_factory,
        "metric_classes": (
            Faithfulness,
            FactualCorrectness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
        ),
    }


class RagasMetricError(RuntimeError):
    def __init__(self, message: str, *, metric_name: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.metric_name = metric_name
        self.details = details or {}


class UndefinedMetricError(RagasMetricError):
    """The metric did not produce a finite numeric score."""


class RagasAdapter:
    metric_names = METRIC_NAMES
    version = RAGAS_VERSION

    def __init__(self, config: RagasConfig) -> None:
        self.config = config
        self._metrics: dict[str, Any] | None = None
        self._http_clients: list[Any] = []
        self._sdk_clients: list[Any] = []
        self._caches: list[Any] = []
        self._budget_guard: Any = None

    @property
    def fingerprint_payload(self) -> dict[str, Any]:
        return fingerprint_payload(self.config)

    async def __aenter__(self) -> RagasAdapter:
        if self._metrics is not None:
            raise RuntimeError("RagasAdapter is already open")
        components = _load_components()
        try:
            if self.config.judge_context_tokens is not None:
                from benchmarking.judge_budget import JudgeBudgetGuard

                self._budget_guard = JudgeBudgetGuard(
                    self.config.judge_url, self.config.judge_api_key,
                    self.config.judge_context_tokens, self.config.timeout,
                )
            judge_cache = self._make_cache(components, "judge", _judge_payload(self.config))
            embedding_cache = self._make_cache(components, "embeddings", _embedding_payload(self.config))
            judge_client = self._make_client(components, self.config.judge_url, self.config.judge_api_key, True)
            embedding_client = self._make_client(
                components, self.config.embedder_url, self.config.embedder_api_key, False
            )
            llm_kwargs: dict[str, Any] = {
                "client": judge_client,
                "cache": judge_cache,
                "temperature": self.config.temperature,
                "top_p": 1.0,
                "max_tokens": self.config.max_tokens,
                "max_retries": 2,
            }
            if self.config.disable_thinking:
                llm_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            llm = components["llm_factory"](self.config.judge_model, **llm_kwargs)
            embeddings = components["embedding_factory"](
                "openai",
                model=self.config.embedder_model,
                client=embedding_client,
                interface="modern",
                cache=embedding_cache,
            )
            if hasattr(embeddings, "aembed_texts"):
                embeddings.aembed_texts = components["cacher"](cache_backend=embedding_cache)(embeddings.aembed_texts)
            faithfulness, correctness, relevancy, precision, recall = components["metric_classes"]
            self._metrics = {
                "faithfulness": faithfulness(llm=llm),
                "factual_correctness": correctness(llm=llm, mode="f1", atomicity="low", coverage="low", beta=1.0),
                "answer_relevancy": relevancy(llm=llm, embeddings=embeddings, strictness=3),
                "context_precision": precision(llm=llm),
                "context_recall": recall(llm=llm),
            }
            return self
        except BaseException:
            await self.aclose()
            raise

    def _make_cache(self, components: dict[str, Any], kind: str, payload: dict[str, Any]) -> Any:
        path = Path(self.config.cache_dir) / kind / _digest(payload)
        path.mkdir(parents=True, exist_ok=True)
        cache = components["cache"](cache_dir=str(path))
        self._caches.append(cache)
        return cache

    def _make_client(self, components: dict[str, Any], url: str, api_key: str, judge: bool) -> Any:
        async def request_hook(request: Any) -> None:
            request.extensions["meno_benchmark_started"] = time.monotonic()
            if judge and self._budget_guard is not None and request.url.path.endswith("/chat/completions"):
                from benchmarking.judge_budget import JudgeContextError

                records = _CALL_RECORDS.get()
                try:
                    checked = await self._budget_guard.check(json.loads(request.content))
                except JudgeContextError as exc:
                    if records is not None:
                        records.append({"kind": "judge_budget", "error": self._redact(str(exc)),
                                        **self._redact(exc.details)})
                    raise
                else:
                    if records is not None:
                        records.append({"kind": "judge_budget", **checked})

        async def response_hook(response: Any) -> None:
            records = _CALL_RECORDS.get()
            if records is None:
                return
            await response.aread()
            request = response.request
            try:
                response_payload: Any = response.json()
            except (ValueError, UnicodeError):
                response_payload = {"body": response.text}
            try:
                request_payload = json.loads(request.content)
            except (ValueError, UnicodeError):
                request_payload = None
            if not judge and isinstance(response_payload, dict):
                response_payload = {
                    key: response_payload[key] for key in ("model", "usage", "error") if key in response_payload
                }
            records.append({
                "kind": "judge" if judge else "embeddings",
                "status_code": response.status_code,
                "elapsed_seconds": time.monotonic() - request.extensions.get("meno_benchmark_started", time.monotonic()),
                "request": self._redact(request_payload) if judge else None,
                "response": self._redact(response_payload),
            })

        http_client = components["http_client"](
            timeout=self.config.timeout,
            event_hooks={"request": [request_hook], "response": [response_hook]},
        )
        self._http_clients.append(http_client)
        client = components["openai_client"](
            base_url=url.rstrip("/"), api_key=api_key or "EMPTY", timeout=self.config.timeout,
            max_retries=0, http_client=http_client,
        )
        self._sdk_clients.append(client)
        return client

    async def score(self, metric_name: str, sample: dict[str, Any]) -> dict[str, Any]:
        if self._metrics is None:
            raise RuntimeError("Use RagasAdapter as an async context manager")
        if metric_name not in self.metric_names:
            raise ValueError(f"Unknown RAGAS metric: {metric_name}")
        keys = {
            "faithfulness": ("user_input", "response", "retrieved_contexts"),
            "factual_correctness": ("response", "reference"),
            "answer_relevancy": ("user_input", "response"),
            "context_precision": ("user_input", "reference", "retrieved_contexts"),
            "context_recall": ("user_input", "reference", "retrieved_contexts"),
        }[metric_name]
        for key in keys:
            value = sample.get(key)
            if key == "retrieved_contexts":
                valid = isinstance(value, list) and bool(value) and all(isinstance(x, str) and x.strip() for x in value)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise RagasMetricError(f"Missing or invalid RAGAS input: {key}", metric_name=metric_name)
        records: list[dict[str, Any]] = []
        token = _CALL_RECORDS.set(records)
        started = time.monotonic()
        try:
            result = await self._metrics[metric_name].ascore(**{key: sample[key] for key in keys})
            raw_score = getattr(result, "value", None)
            try:
                score = float(raw_score) if not isinstance(raw_score, bool) else float("nan")
            except (TypeError, ValueError, OverflowError):
                score = float("nan")
            details = self._details(records, started)
            if not math.isfinite(score):
                raise UndefinedMetricError(
                    f"RAGAS {metric_name} returned an undefined or nonfinite score",
                    metric_name=metric_name, details=details,
                )
            return {"score": score, "reason": self._redact(getattr(result, "reason", None)), "details": details}
        except RagasMetricError:
            raise
        except Exception as exc:
            budget_errors = [record for record in records if record.get("kind") == "judge_budget" and record.get("error")]
            message = (
                f"RAGAS {metric_name}: {budget_errors[-1]['error']}" if budget_errors
                else self._redact(f"RAGAS {metric_name} failed: {type(exc).__name__}: {exc}")
            )
            raise RagasMetricError(
                message,
                metric_name=metric_name, details=self._details(records, started),
            ) from exc
        finally:
            _CALL_RECORDS.reset(token)

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            for key in (self.config.judge_api_key, self.config.embedder_api_key):
                if key and key != "EMPTY":
                    value = value.replace(key, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {self._redact(key): self._redact(item) for key, item in value.items()}
        return value

    @staticmethod
    def _details(records: list[dict[str, Any]], started: float) -> dict[str, Any]:
        usage: dict[str, int] = {}
        for record in records:
            response = record.get("response")
            tokens = response.get("usage", {}) if isinstance(response, dict) else {}
            if isinstance(tokens, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    count = tokens.get(key)
                    if isinstance(count, int) and not isinstance(count, bool):
                        usage[key] = usage.get(key, 0) + count
        return {
            "ragas_version": RAGAS_VERSION,
            "elapsed_seconds": time.monotonic() - started,
            "http_calls": [record for record in records if record.get("kind") != "judge_budget"],
            "budget_checks": [record for record in records if record.get("kind") == "judge_budget"],
            "usage": usage,
            "usage_scope": "uncached_http_calls_only",
        }

    async def aclose(self) -> None:
        self._metrics = None
        if self._budget_guard is not None:
            await self._budget_guard.aclose()
            self._budget_guard = None
        while self._sdk_clients:
            await self._sdk_clients.pop().close()
        while self._http_clients:
            await self._http_clients.pop().aclose()
        while self._caches:
            cache = self._caches.pop()
            cache.cache.close()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.aclose()

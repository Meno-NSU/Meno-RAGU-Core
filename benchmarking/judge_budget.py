from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class JudgeContextError(RuntimeError):
    def __init__(self, message: str, *, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class JudgeBudgetGuard:
    margin_tokens = 32
    cache_size = 4096

    def __init__(self, url: str, api_key: str, context_tokens: int, timeout: float) -> None:
        if type(context_tokens) is not int or context_tokens <= 0:
            raise ValueError("Окно судьи должно быть положительным целым числом.")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Тайм-аут токенизатора должен быть конечным положительным числом.")
        parts = urlsplit(url)
        if (
            parts.scheme not in {"http", "https"} or not parts.netloc
            or parts.username or parts.password or parts.query or parts.fragment
        ):
            raise ValueError("Адрес судьи должен быть HTTP(S)-адресом без пароля и параметров.")
        prefix = parts.path.rstrip("/")
        if prefix.endswith("/v1"):
            prefix = prefix[:-3]
        self.tokenize_url = urlunsplit((parts.scheme, parts.netloc, prefix + "/tokenize", "", ""))
        self.context_tokens = context_tokens
        self.client = httpx.AsyncClient(
            timeout=timeout, trust_env=False, headers={"Authorization": f"Bearer {api_key or 'EMPTY'}"},
        )
        self._cache: OrderedDict[str, tuple[int, int | None]] = OrderedDict()

    @staticmethod
    def _error(message: str, code: str, **details: Any) -> JudgeContextError:
        return JudgeContextError(message, details={"code": code, **details})

    async def check(self, payload: dict) -> dict:
        reserved = payload.get("max_completion_tokens")
        if reserved is None:
            reserved = payload.get("max_tokens")
        if type(reserved) is not int or reserved <= 0:
            raise self._error(
                "Не задан положительный резерв токенов ответа судьи.", "judge_invalid_request_budget",
            )
        model, messages = payload.get("model"), payload.get("messages")
        if not isinstance(model, str) or not model.strip() or not isinstance(messages, list) or not messages:
            raise self._error("Отсутствуют модель или сообщения судье.", "judge_invalid_tokenizer_input")
        request = {"model": model, "messages": messages, "add_generation_prompt": True}
        if "chat_template_kwargs" in payload:
            if not isinstance(payload["chat_template_kwargs"], dict):
                raise self._error("Некорректные параметры шаблона судьи.", "judge_invalid_tokenizer_input")
            request["chat_template_kwargs"] = payload["chat_template_kwargs"]
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        cached = key in self._cache
        if cached:
            count, served_limit = self._cache.pop(key)
            self._cache[key] = (count, served_limit)
        else:
            try:
                response = await self.client.post(self.tokenize_url, json=request)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise self._error(
                    "Не удалось проверить окно судьи через /tokenize. Запрос генерации отклонён.",
                    "judge_tokenizer_unavailable", error_type=type(exc).__name__,
                    status_code=exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None,
                ) from exc
            try:
                body = response.json()
                count = body.get("count")
                served_limit = body.get("max_model_len")
                if type(count) is not int or count < 0:
                    raise ValueError("Некорректное число токенов.")
                if served_limit is not None and (type(served_limit) is not int or served_limit <= 0):
                    raise ValueError("Некорректное окно токенизатора.")
            except (ValueError, AttributeError, TypeError) as exc:
                raise self._error(
                    "Токенизатор судьи вернул некорректный ответ. Запрос генерации отклонён.",
                    "judge_tokenizer_invalid_response",
                ) from exc
            self._cache[key] = (count, served_limit)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        limit = min(self.context_tokens, served_limit) if served_limit is not None else self.context_tokens
        info = {
            "prompt_tokens": count, "reserved": reserved, "margin_tokens": self.margin_tokens,
            "context_limit": limit, "tokenizer_context_limit": served_limit, "cached": cached,
        }
        if count + reserved + self.margin_tokens > limit:
            raise self._error(
                f"Запрос судье требует {count} токенов входа, {reserved} токенов ответа "
                f"и запас {self.margin_tokens}, доступное окно — {limit}. Контекст сохранён без обрезки.",
                "judge_context_length_exceeded", **info,
            )
        return info

    async def aclose(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> JudgeBudgetGuard:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

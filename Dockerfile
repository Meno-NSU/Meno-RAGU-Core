FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" HF_HOME=/app/var/huggingface FASTEMBED_CACHE_PATH=/app/var/fastembed
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --create-home meno
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY code ./code
COPY legal ./legal
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv sync --frozen --no-dev && mkdir -p var && chown -R meno:meno var
USER meno
EXPOSE 9006
CMD ["meno-rag-api"]

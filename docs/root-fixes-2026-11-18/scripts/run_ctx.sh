#!/bin/bash
# Paired legacy/sources answers (ctx_answers.py) on stand-v6k2 with the code of feat/context-as-sources.
set -u
F=$HOME/meno-v2/ctx1118
CODE=$F/code-a8d0366
PY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python
S=$HOME/meno-v2/ragu-v2/stand-v6k2
LOGF=$F/logs/answers.log
cd "$CODE" || exit 1
echo "$(date -u +%FT%TZ) run_ctx.sh stand $S args $*" >> "$LOGF"
env QDRANT_URL=http://127.0.0.1:6433 STAND_RESOURCES_DIR="$S" RAGU_GRAPH_DIR="$S/knowledge/ragu_graph" \
    EMBEDDER_URL=http://127.0.0.1:8001/v1 EMBEDDER_MODEL=BAAI/bge-m3 \
    ENGINE_RERANKER_URL=http://everest.nsu.ru:9028/v1 ENGINE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    EMBED_CONCURRENCY=32 VLLM_ENDPOINTS=http://everest.nsu.ru:9022 QA_FEWSHOTS_ENABLED=false \
    CAPTURE_PIPELINE_TRACE=false LOG_LEVEL=WARNING LOG_CONTENT_PREVIEWS=false TZ=Asia/Novosibirsk \
    LOGURU_LEVEL=WARNING TQDM_DISABLE=1 \
    PYTHONPATH=src:code PYTHONDONTWRITEBYTECODE=1 \
    nice "$PY" -u "$F/ctx_answers.py" "$F/jobs.json" --model gemma-4-12B-it --base-url http://everest.nsu.ru:9022/v1 \
    --label v6k2-ctx --progress-file "$F/progress.txt" "$@" >> "$LOGF" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) run_ctx.sh exit $rc" >> "$LOGF"
exit $rc

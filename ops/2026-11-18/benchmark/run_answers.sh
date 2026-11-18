#!/bin/bash
# RAGU in-process answers for the 18.11 benchmark: ragu_answers.py with the environment of the live RAGU stand
# container, as ~/meno-v2/final1116/inproc/run_inproc.sh had it (gemma-4-12B-it on the H100 :9022, engine reranker on
# the H100 :9028, bge-m3 :8001, EMBED_CONCURRENCY=32, LLM rerank on (default), QA few-shots off,
# TZ Asia/Novosibirsk), code Meno-RAGU-Core main d3c7326 (~/meno-v2/final1118/code-d3c7326: 6601cb4 + a graph element enters the context only
# with a retrieved source chunk). The only change against
# 16.11: STAND_RESOURCES_DIR / RAGU_GRAPH_DIR point at the given stand dir. LOG_LEVEL=WARNING (16.11: INFO),
# LOGURU_LEVEL=WARNING (the ragu library's per-search INFO lines) and TQDM_DISABLE=1 only keep the log small; the runner's
# own progress lines are not log records.
# usage: run_answers.sh STAND_DIR JOBS.json LOGFILE [ragu_answers.py options...]
set -u
F=$HOME/meno-v2/final1118
CODE=$F/code-d3c7326
PY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python
S=$1; JOBS=$2; LOGF=$3; shift 3
cd "$CODE" || exit 1
echo "$(date -u +%FT%TZ) run_answers.sh stand $S jobs $JOBS args $*" >> "$LOGF"
env QDRANT_URL=http://127.0.0.1:6433 STAND_RESOURCES_DIR="$S" RAGU_GRAPH_DIR="$S/knowledge/ragu_graph" \
    EMBEDDER_URL=http://127.0.0.1:8001/v1 EMBEDDER_MODEL=BAAI/bge-m3 \
    ENGINE_RERANKER_URL=http://everest.nsu.ru:9028/v1 ENGINE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    EMBED_CONCURRENCY=32 VLLM_ENDPOINTS=http://everest.nsu.ru:9022 QA_FEWSHOTS_ENABLED=false \
    CAPTURE_PIPELINE_TRACE=false LOG_LEVEL=WARNING LOG_CONTENT_PREVIEWS=false TZ=Asia/Novosibirsk \
    LOGURU_LEVEL=WARNING TQDM_DISABLE=1 \
    PYTHONPATH=src:code PYTHONDONTWRITEBYTECODE=1 \
    nice "$PY" -u "$F/ragu_answers.py" "$JOBS" --model gemma-4-12B-it --base-url http://everest.nsu.ru:9022/v1 "$@" \
    >> "$LOGF" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) run_answers.sh exit $rc" >> "$LOGF"
exit $rc

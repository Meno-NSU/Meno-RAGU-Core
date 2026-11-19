#!/bin/bash
# Judge-free replay of graph navigation (nav_sim.py): code feat/nav-expansion b709314 (code-b709314), stand-v6k2 read-only.
D=$HOME/meno-v2/audit1119/nav/replay
PY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python
S=$HOME/meno-v2/ragu-v2/stand-v6k2
OUT=${1:-$D/nav_sim.jsonl}
cd $D/../code-b709314 || exit 1
env QDRANT_URL=http://127.0.0.1:6433 STAND_RESOURCES_DIR=$S RAGU_GRAPH_DIR=$S/knowledge/ragu_graph \
    EMBEDDER_URL=http://127.0.0.1:8001/v1 EMBEDDER_MODEL=BAAI/bge-m3 \
    ENGINE_RERANKER_URL=http://everest.nsu.ru:9028/v1 ENGINE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    EMBED_CONCURRENCY=32 VLLM_ENDPOINTS=http://everest.nsu.ru:9022 QA_FEWSHOTS_ENABLED=false \
    CAPTURE_PIPELINE_TRACE=false LOG_LEVEL=WARNING LOG_CONTENT_PREVIEWS=false TZ=Asia/Novosibirsk \
    LOGURU_LEVEL=WARNING TQDM_DISABLE=1 PYTHONPATH=src:code PYTHONDONTWRITEBYTECODE=1 SIM_CONCURRENCY=${SIM_CONCURRENCY:-3} NAV_LIMIT=${NAV_LIMIT:-} \
    nice $PY -u $D/nav_sim.py $OUT $D/alias_table_v6.json >> $OUT.log 2>&1
echo "$(date -u +%FT%TZ) exit $?" >> $OUT.log

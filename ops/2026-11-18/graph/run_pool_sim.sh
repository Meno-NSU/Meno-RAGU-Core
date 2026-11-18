#!/bin/bash
# Judge-free replay of the balanced graph pool (pool_sim.py) on the gold items of the 558 questions: code
# feat/balanced-graph-pool f9930dd (code-f9930dd), stand-v6k2, settings as the benchmark's run_answers.sh. Run it when
# gemma is back on the H100 :9022 (the rewrite needs it) and the benchmark answers are done.
D=$HOME/meno-v2/ragu-probe/v61
PY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python
S=$HOME/meno-v2/ragu-v2/stand-v6k2
cd $D/code-f9930dd || exit 1
env QDRANT_URL=http://127.0.0.1:6433 STAND_RESOURCES_DIR=$S RAGU_GRAPH_DIR=$S/knowledge/ragu_graph \
    EMBEDDER_URL=http://127.0.0.1:8001/v1 EMBEDDER_MODEL=BAAI/bge-m3 \
    ENGINE_RERANKER_URL=http://everest.nsu.ru:9028/v1 ENGINE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
    EMBED_CONCURRENCY=32 VLLM_ENDPOINTS=http://everest.nsu.ru:9022 QA_FEWSHOTS_ENABLED=false \
    CAPTURE_PIPELINE_TRACE=false LOG_LEVEL=WARNING LOG_CONTENT_PREVIEWS=false TZ=Asia/Novosibirsk \
    LOGURU_LEVEL=WARNING TQDM_DISABLE=1 PYTHONPATH=src:code PYTHONDONTWRITEBYTECODE=1 SIM_CONCURRENCY=2 \
    nice $PY -u $D/pool_sim.py $HOME/meno-v2/bench/items558.jsonl $D/pool_sim.jsonl gemma-4-12B-it \
        http://everest.nsu.ru:9022/v1 > $D/pool_sim.log 2>&1
echo "$(date -u +%FT%TZ) exit $?" >> $D/pool_sim.log
$HOME/meno-v2/venv/bin/python -I $D/pool_sim.py --summary $HOME/meno-v2/bench/items558.jsonl $D/pool_sim.jsonl >> $D/pool_sim.log 2>&1

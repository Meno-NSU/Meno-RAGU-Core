#!/usr/bin/env bash
set -euo pipefail

# Starts the local embedding service (bge-m3) for the Meno knowledge base
# indexing and for the dialogue scripts. The service runs on the local GPU
# (RTX 3060 is enough: bge-m3 is an encoder of about 1.1 GB in bf16). The
# script is idempotent: it does nothing when the service already responds.
#
# VLLM_BIN may be overridden via the environment. The default points to the
# conda environment with a stable vLLM release: the shell PATH may contain a
# development checkout whose tokenizer initialization is incompatible with
# the installed transformers for XLM-R models (bge-m3 included).

EMBEDDER_PORT=8001
EMBEDDER_URL="http://localhost:${EMBEDDER_PORT}/v1/models"
LOCAL_BGE_PATH="/home/bond005/models/bge-m3"
VLLM_BIN="${VLLM_BIN:-/home/bond005/anaconda3/envs/env_for_mtrag/bin/vllm}"
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=8192
LOG_DIR="../logs"
LOG_FILE="${LOG_DIR}/local_embedder.log"

die() { printf '[ОШИБКА] %s\n' "$*" >&2; exit 1; }
log() { printf '[ТРАССА] %s\n' "$*" >&2; }

[[ -d "${LOCAL_BGE_PATH}" ]] || die "локальная модель ${LOCAL_BGE_PATH} не найдена."
[[ -x "${VLLM_BIN}" ]] || die "исполняемый файл vllm (${VLLM_BIN}) не найден; задайте VLLM_BIN."
mkdir -p "${LOG_DIR}"

if curl -sf --max-time 5 "${EMBEDDER_URL}" >/dev/null 2>&1; then
  log "служба векторизации уже отвечает на ${EMBEDDER_URL}."
  exit 0
fi

setsid nohup "${VLLM_BIN}" serve "${LOCAL_BGE_PATH}" \
  --served-model-name BAAI/bge-m3 \
  --task embed \
  --host 127.0.0.1 \
  --port "${EMBEDDER_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  >>"${LOG_FILE}" 2>&1 &
log "служба векторизации запускается (vllm pid $!), журнал: ${LOG_FILE}."

for attempt in $(seq 1 60); do
  sleep 10
  if curl -sf --max-time 5 "${EMBEDDER_URL}" >/dev/null 2>&1; then
    log "служба векторизации готова: ${EMBEDDER_URL}."
    exit 0
  fi
  log "ожидание службы векторизации (попытка ${attempt} из 60)..."
done
die "служба векторизации не поднялась; смотрите журнал ${LOG_FILE}."

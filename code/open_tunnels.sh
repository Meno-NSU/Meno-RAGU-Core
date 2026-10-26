#!/usr/bin/env bash
set -euo pipefail

# Opens the ssh tunnel from the working machine to the Meno generative service
# on the Cascade cluster. The service node is resolved via squeue by the job
# name, so the tunnel survives job re-submissions: just re-run this script.
# The embedding service does not need a tunnel: it runs locally
# (serve_local_embedder.sh).

SSH_KEY="${HOME}/nusc-nsu/nusc-nsu-key"
CLUSTER_LOGIN="bi"
CLUSTER_HOST="11.0.1.18"
MENO_JOB="meno-idx"
LOCAL_MENO_PORT=8023
SERVICE_PORT=8000

die() { printf '[ОШИБКА] %s\n' "$*" >&2; exit 1; }
log() { printf '[ТРАССА] %s\n' "$*" >&2; }

resolve_node() {
    local job_name="$1"
    ssh -i "${SSH_KEY}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
        -l "${CLUSTER_LOGIN}" "${CLUSTER_HOST}" \
        "squeue -h -n ${job_name} -t R -o '%N'"
}

[ -f "${SSH_KEY}" ] || die "ssh-ключ ${SSH_KEY} не найден."

MENO_NODE="$(resolve_node "${MENO_JOB}" | head -n 1)"
[ -n "${MENO_NODE}" ] || die "задание ${MENO_JOB} не найдено в состоянии RUNNING; подайте его на кластере (code/slurm/submit_meno_service.sh)."
log "служба Meno: узел ${MENO_NODE}, локальный порт ${LOCAL_MENO_PORT}."

if command -v fuser >/dev/null 2>&1; then
    fuser -k "${LOCAL_MENO_PORT}/tcp" >/dev/null 2>&1 || true
else
    log "fuser недоступен: если порт занят старым туннелем, завершите его вручную."
fi
sleep 1

setsid nohup ssh -i "${SSH_KEY}" -N \
  -L "${LOCAL_MENO_PORT}:${MENO_NODE}:${SERVICE_PORT}" \
  -l "${CLUSTER_LOGIN}" "${CLUSTER_HOST}" >/dev/null 2>&1 &
log "туннель запущен (ssh pid $!)."

for attempt in $(seq 1 30); do
    sleep 5
    if curl -sf --max-time 5 "http://localhost:${LOCAL_MENO_PORT}/v1/models" >/dev/null 2>&1; then
        log "служба генерации отвечает; туннель готов."
        exit 0
    fi
    log "ожидание службы (попытка ${attempt} из 30)..."
done
die "служба не ответила за отведённое время; проверьте лог задания на кластере."

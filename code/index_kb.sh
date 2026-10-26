#!/usr/bin/env bash
set -euo pipefail

# The full knowledge base indexing driver for the Meno stand. It runs all the
# pipeline steps one by one, from the curated handbook to the RAGU knowledge
# graph. Every step is skipped when its output already exists; export FORCE=1
# to re-run a step despite the existing output. Logs are written into ../logs.
#
# Prerequisites: run from the code directory in the python environment with all
# dependencies. The Qdrant server is started automatically as a docker container
# if it is not reachable; the local embedding service (bge-m3) is started
# automatically via serve_local_embedder.sh; the Meno service on the Cascade
# cluster must be submitted (code/slurm/submit_meno_service.sh) and tunneled
# (open_tunnels.sh) beforehand.

MENO_MODEL="bond005/meno-lite-0.2"
LOCAL_MENO_PATH="/home/bond005/models/meno-lite-0.2"
EMBEDDER_NAME="BAAI/bge-m3"
MENO_URL="http://localhost:8023/v1"
EMBEDDER_URL="http://localhost:8001/v1"
API_KEY="EMPTY"
QDRANT_URL="http://localhost:6333"
QDRANT_CONTAINER="qdrant-meno"
MAXTOKENS=350
MAX_INPUT_LEN=6000
RATE_MAX_SIMULTANEOUS=32

DATA="../data"
SUMMARY="${DATA}/source/nsu_text_corpus_$(date +%Y%m%d).jsonl"
TEACHERS="${DATA}/source/teachers_corpus_20261019.jsonl"
BASE="${DATA}/source/nsu_text_corpus_20250318_validated.jsonl"
NEWS_FILES=(
  "${DATA}/source/filtered_news_20260220_20260306.jsonl"
  "${DATA}/source/filtered_news_20260306_20260403.jsonl"
  "${DATA}/source/filtered_news_20260403_20260709.jsonl"
)

UNITED="${DATA}/united_texts_about_nsu.jsonl"
CHUNKED="${DATA}/chunked_texts_about_nsu.jsonl"
WITH_LENGTHS="${DATA}/chunked_texts_about_nsu_with_lengths.jsonl"
WITH_METADATA="${DATA}/chunked_texts_about_nsu_with_metadata.jsonl"
WITH_SCORES="${DATA}/chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
FINAL="${DATA}/chunked_texts_about_nsu_with_metadata_scores_and_news_dates.jsonl"
REVERSE_INDEX="${DATA}/chunk_mapping_to_texts.json"
GRAPH_DIR="${DATA}/knowledge/ragu_graph"
LLM_CACHE_DIR="${DATA}/knowledge/ragu_graph_llm_cache"
LOG_DIR="../logs"
FORCE="${FORCE:-0}"

NEWS_ARGS=()
for cur_news in "${NEWS_FILES[@]}"; do
  NEWS_ARGS+=(-n "${cur_news}")
done

die() { printf '\n[ОШИБКА] %s\n' "$*" >&2; exit 1; }
log() { printf '\n[ПРОГРЕСС] %s\n' "$*" >&2; }

wait_for_url() {
  local url="$1" label="$2" attempts="${3:-60}"
  for attempt in $(seq 1 "${attempts}"); do
    if curl -sf --max-time 5 "${url}" >/dev/null 2>&1; then
      log "${label} доступен (${url})."
      return 0
    fi
    sleep 10
  done
  die "служба ${label} (${url}) не отвечает."
}

run_step() {
  local name="$1" output="$2"
  shift 2
  if [[ -e "${output}" && "${FORCE}" != "1" ]]; then
    log "шаг «${name}» пропущен: ${output} уже существует."
    return 0
  fi
  log "шаг «${name}»."
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

[[ -f chat.py ]] || die "запускайте скрипт из каталога code (например, bash index_kb.sh)."
mkdir -p "${LOG_DIR}" ../pictures

for needed_file in \
  "${DATA}/nsu_main_info.md" \
  "${DATA}/source/teachers.json" \
  "${DATA}/source/teachers_chunks.json" \
  "${BASE}" \
  "${NEWS_FILES[@]}"; do
  [[ -f "${needed_file}" ]] || die "исходный файл ${needed_file} не найден."
done

python -c 'import ragu, openai, nltk, razdel, transformers, qdrant_client' 2>/dev/null \
  || die "окружение python неполное: нужны graph_ragu, openai, nltk, razdel, transformers."

if ! curl -sf --max-time 5 "${QDRANT_URL}/collections" >/dev/null 2>&1; then
  log "Qdrant не отвечает; запускаю docker-контейнер ${QDRANT_CONTAINER}."
  docker run -d --name "${QDRANT_CONTAINER}" -p 6333:6333 \
    -v qdrant_storage:/qdrant/storage qdrant/qdrant \
    || die "не удалось запустить docker-контейнер Qdrant."
fi
wait_for_url "${QDRANT_URL}/collections" "Qdrant" 30

if ! curl -sf --max-time 5 "${EMBEDDER_URL}/models" >/dev/null 2>&1; then
  log "служба векторизации не отвечает; запускаю локальный сервис serve_local_embedder.sh."
  bash serve_local_embedder.sh || die "не удалось запустить локальную службу векторизации."
fi
wait_for_url "${EMBEDDER_URL}/models" "служба векторизации bge-m3" 90

wait_for_url "${MENO_URL}/models" "служба генерации Meno" 90

if ! locale -a 2>/dev/null | grep -iq 'ru_RU.utf8'; then
  log "предупреждение: локаль ru_RU.UTF-8 не найдена; названия месяцев в шаге дат новостей могут быть некорректны."
fi

run_step "01-nsu-main-info" "${SUMMARY}" \
  python prepare_nsu_main_info_corpus.py \
    -i "${DATA}/nsu_main_info.md" -o "${SUMMARY}"

run_step "02-teachers" "${TEACHERS}" \
  python prepare_teachers_corpus.py \
    -i "${DATA}/source/teachers.json" -c "${DATA}/source/teachers_chunks.json" -o "${TEACHERS}"

run_step "03-combine" "${UNITED}" \
  python combine_data.py \
    -s "${SUMMARY}" -t "${TEACHERS}" -b "${BASE}" "${NEWS_ARGS[@]}" -o "${UNITED}"

run_step "04-chunking" "${CHUNKED}" \
  python apply_chunking.py \
    -i "${UNITED}" -o "${CHUNKED}" -m "${LOCAL_MENO_PATH}" --maxtokens "${MAXTOKENS}"

run_step "05-num-tokens" "${WITH_LENGTHS}" \
  python calculate_num_tokens.py \
    -i "${CHUNKED}" -o "${WITH_LENGTHS}" -m "${LOCAL_MENO_PATH}"

if [[ -e "${WITH_METADATA}" && "${FORCE}" != "1" ]]; then
  log "шаг «06-metadata»: файл ${WITH_METADATA} существует, продолжаю с --resume."
  python generate_title_and_annotation.py \
    -i "${WITH_LENGTHS}" -o "${WITH_METADATA}" \
    -m "${MENO_MODEL}" --url "${MENO_URL}" --key "${API_KEY}" \
    --max_input_len "${MAX_INPUT_LEN}" --resume 2>&1 | tee -a "${LOG_DIR}/06-metadata.log"
else
  run_step "06-metadata" "${WITH_METADATA}" \
    python generate_title_and_annotation.py \
      -i "${WITH_LENGTHS}" -o "${WITH_METADATA}" \
      -m "${MENO_MODEL}" --url "${MENO_URL}" --key "${API_KEY}" \
      --max_input_len "${MAX_INPUT_LEN}"
fi

run_step "07-scoring" "${WITH_SCORES}" \
  python -u evaluate_title_and_annotation.py \
    -i "${WITH_METADATA}" -o "${WITH_SCORES}" -m "${EMBEDDER_NAME}" \
    --url "${EMBEDDER_URL}" --key "${API_KEY}" --minibatch 32

run_step "08-news-dates" "${FINAL}" \
  python update_title_for_news.py \
    -b "${WITH_SCORES}" "${NEWS_ARGS[@]}" -o "${FINAL}"

run_step "09-histogram-with-zeros" "../pictures/histogram_of_text_quality.png" \
  python -u histogram_of_document_scores.py \
    -i "${FINAL}" -o "../pictures/histogram_of_text_quality.png" --zeros

run_step "09-histogram-without-zeros" "../pictures/histogram_of_text_quality_without_zeros.png" \
  python -u histogram_of_document_scores.py \
    -i "${FINAL}" -o "../pictures/histogram_of_text_quality_without_zeros.png"

run_step "10-reverse-index" "${REVERSE_INDEX}" \
  python prepare_chunk_mapping.py \
    -i "${FINAL}" -o "${REVERSE_INDEX}"

mkdir -p "${GRAPH_DIR}" "${LLM_CACHE_DIR}"
run_step "11-graph" "${GRAPH_DIR}/menon_chunk_mapping.json" \
  python -u prepare_graph_index.py \
    -i "${FINAL}" -r "${REVERSE_INDEX}" -o "${GRAPH_DIR}" \
    -e "${EMBEDDER_NAME}" --embedder_url "${EMBEDDER_URL}" --embedder_key "${API_KEY}" \
    --llm "${MENO_MODEL}" --url "${MENO_URL}" --key "${API_KEY}" \
    --qdrant_location "${QDRANT_URL}" \
    --chunk_header title --chunk_header_sources base,news \
    -a ../data/knowledge/abbreviations.json \
    --icl_examples 0 \
    --max_completion_tokens 6144 \
    --cache "${LLM_CACHE_DIR}" \
    --rate_max_simultaneous "${RATE_MAX_SIMULTANEOUS}"

log "индексирование завершено: граф в ${GRAPH_DIR}, корпус ${FINAL}, отображение ${REVERSE_INDEX}."

#!/bin/bash
# 2026-11-18: Meno-RAGU-Core main d3c7326 = 6601cb4 + a graph element enters the context only with a retrieved source chunk (GRAPH_ELEMENT_EVIDENCE, default on); for stand-v6k2 (KB v3.9b indexed chunks). Before switching, and before a rollback, stand_parity.py checks that the stand's chunk mapping and its Qdrant collection are one set of ids.
# 2026-11-17: Meno-RAGU-Core main 6601cb4 = source authority: context order by relevance x quality (not log-odds), the sources of the answer ordered by support x quality, quality on the KB scale from stand-v5k/document_quality.json (already in place; older code ignores it), GRAPH_CHANNEL switch (default on). Same graph and resources (stand-v5k), same version label.
# shows «Переформулировка запроса…» in progress instead of nothing for the analysis call. Same graph and resources (stand-v5k), same version label.
# RAGU_VLLM_ENDPOINTS=<url> (optional) points the stand's LLM elsewhere, e.g. the Everest H100 gemma http://everest.nsu.ru:9022;
# RAGU_ENGINE_RERANKER_URL=<url/v1> (optional) the engine cross-encoder, e.g. the Everest H100 one http://everest.nsu.ru:9028/v1.
# Switch the RAGU stand to graph v3, named "RAGU v3" for the Menon gateway: Meno-RAGU-Core main 29c3646 (turn routing
# as in Menon 1.1, section sources; official sources before posts with source prior weight 1, staff-page anchors, graph
# elements and one page no longer crowd document chunks out, UNFIT relations shown without their type) and the graph v3
# resources (~m.kulakov/meno-v2/ragu-v2/stand-v3 or stand-v3k) with their Qdrant :6433.
# Run as root:  sudo bash /home/m.kulakov/meno-v2/ragu-probe/deploy_ragu_v3.sh /home/m.kulakov/meno-v2/ragu-v2/stand-v3k
# The graph and its Qdrant come from compose.ragu-v3.yaml next to this script, so a plain "scripts/backend.sh up core"
# (or the rollback) brings the stand back to the nsu-v1 graph, never to half of v3.
set -euo pipefail

TARGET=d3c73264bc85bd72914ae87b5e511f8f18b45c04
D=/home/rsolikhov/Meno-Deploy
ENVF=$D/.env.backend
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STAND=${1:-/home/m.kulakov/meno-v2/ragu-v2/stand-v5k}  # or .../stand-v3k: v3 plus the KB v3.8 documents in the chunk search
# the version names the resources, so v3 and v3k answers are separate leaderboard rows
VERSION=nsu-ragu-$(basename "$STAND" | sed "s/^stand-//")-2026-11-18
# the name follows the resources too ("RAGU v3k", "RAGU v4k"); the Menon gateway shows the label of ~/meno-v2/stands.json
NAME="RAGU $(basename "$STAND" | sed "s/^stand-//")"
TS=$(date -u +%Y%m%d-%H%M%S)

[[ $EUID -eq 0 ]] || { echo "run with sudo"; exit 1; }
[[ -f $ENVF ]] || { echo "no $ENVF"; exit 1; }
[[ -f $STAND/knowledge/ragu_graph/qdrant_collections.json ]] || { echo "no graph v3 resources in $STAND; nothing changed"; exit 1; }
python3 - "$STAND/knowledge/ragu_graph/qdrant_collections.json" <<'PY'
import json, sys, urllib.request
for name in json.load(open(sys.argv[1])).values():
    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:6433/collections/{name}", timeout=30))["result"]
    assert info["points_count"] > 0, f"{name} is empty"
print("graph v3 collections on :6433 are in place")
PY
# the chunk mapping the stand reads and the points of its chunk collection must be one set: a stand served against
# another stand's collection loses chunks without an error (18.11: stand-v6k's map against the v6k2 collection)
python3 -I "$HERE/stand_parity.py" "$STAND" || { echo "the stand's chunk mapping and its Qdrant collection differ; nothing changed"; exit 1; }

IMAGE=$(grep -E '^CORE_IMAGE=' "$ENVF" | tail -1 | cut -d= -f2- || true)
IMAGE=${IMAGE:-meno-ragu-core:local}
SAVED=${IMAGE%%:*}:before-v3-$TS
SRC=$(grep -E '^CORE_SOURCE_DIR=' "$ENVF" | tail -1 | cut -d= -f2- || true)
SRC=${SRC:-../meno_ragu_stand}
[[ $SRC = /* ]] || SRC=$(cd "$D" && cd "$SRC" && pwd)
git_src() { git -c safe.directory="$SRC" -C "$SRC" "$@"; }
compose_override() { docker compose --project-name meno-ragu-backend --project-directory "$D" --env-file "$ENVF" \
    -f "$D/compose.backend.yaml" -f "$HERE/compose.ragu-v3.yaml" "$@"; }

CORE=$(cd "$D" && ./scripts/backend.sh ps -q core)
[[ -n $CORE ]] || { echo "the compose core of meno-ragu-backend is not running; nothing changed"; exit 1; }
if [[ -n $(git_src status --porcelain --untracked-files=no) ]]; then
    echo "the source checkout has local changes; nothing changed:"; git_src status --short; exit 1
fi
PREV_COMMIT=$(git_src rev-parse HEAD)
PREV_MOUNT=$(docker inspect "$CORE" --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Source}}{{end}}{{end}}')
echo "the stand now serves $PREV_MOUNT"

echo "== backup"
docker tag "$IMAGE" "$SAVED"
install -m 600 "$ENVF" "$ENVF.before-v3-$TS"
# a redeploy over v3/v3k rewrites the override below; the rollback brings back the one the stand ran with
PREV_OVERRIDE=""
if [[ -f "$HERE/compose.ragu-v3.yaml" ]] && grep -qF "source: $PREV_MOUNT" "$HERE/compose.ragu-v3.yaml"; then
    PREV_OVERRIDE=$HERE/compose.ragu-v3.before-$TS.yaml
    cp "$HERE/compose.ragu-v3.yaml" "$PREV_OVERRIDE"
fi
cat > "$HERE/rollback_ragu_v3.sh" <<ROLLBACK
#!/bin/bash
# Back to the graph, image, settings and source of before the graph v3 deploy of $TS. Run as root.
set -euo pipefail
if [[ -f $PREV_MOUNT/knowledge/ragu_graph/qdrant_collections.json ]] && ! python3 -I $HERE/stand_parity.py $PREV_MOUNT; then
    echo "$PREV_MOUNT no longer matches its chunk collection; nothing changed. A consistent rollback:"
    echo "  sudo bash $HERE/deploy_ragu_v7.sh /home/m.kulakov/meno-v2/ragu-v2/stand-v5k"
    exit 1
fi
docker tag $SAVED $IMAGE
install -m 600 $ENVF.before-v3-$TS $ENVF
git -c safe.directory=$SRC -C $SRC reset -q --keep $PREV_COMMIT
cd $D
if [[ "$PREV_MOUNT" == /home/m.kulakov/meno-v2/ragu-v2/stand && -f "$HERE/compose.ragu-v2.yaml" ]]; then
    # the stand ran graph v2 (deploy_ragu_v2.sh): back to it with its compose override
    docker compose --project-name meno-ragu-backend --project-directory $D --env-file $ENVF \\
        -f $D/compose.backend.yaml -f $HERE/compose.ragu-v2.yaml up -d --no-deps --wait --wait-timeout 1800 core
elif [[ -n "$PREV_OVERRIDE" ]]; then
    # the stand ran graph v3 resources ($PREV_MOUNT): back to them with the override of then
    cp $PREV_OVERRIDE $HERE/compose.ragu-v3.yaml
    docker compose --project-name meno-ragu-backend --project-directory $D --env-file $ENVF \\
        -f $D/compose.backend.yaml -f $HERE/compose.ragu-v3.yaml up -d --no-deps --wait --wait-timeout 1800 core
else
    ./scripts/backend.sh up -d --no-deps --wait --wait-timeout 1800 core
fi
curl -fsS http://127.0.0.1:9007/meno/v1/manifest; echo
ROLLBACK
chmod 755 "$HERE/rollback_ragu_v3.sh"
echo "rollback: sudo bash $HERE/rollback_ragu_v3.sh"

echo "== source ${TARGET:0:7}"
git_src fetch -q origin main
git_src checkout -q main
git_src merge -q --ff-only origin/main
[[ $(git_src rev-parse HEAD) == "$TARGET" ]] || { echo "origin/main is $(git_src rev-parse --short HEAD), expected ${TARGET:0:7}"; exit 1; }

echo "== settings"
set_env() { if grep -qE "^$1=" "$ENVF"; then sed -i "s|^$1=.*|$1=$2|" "$ENVF"; else echo "$1=$2" >> "$ENVF"; fi; }
set_env MENO_BACKEND_NAME "$NAME"
set_env MENO_BACKEND_VERSION "$VERSION"
set_env MENO_BACKEND_BUILD "${TARGET:0:7}"
# optional: the stand's LLM endpoint(s), e.g. sudo RAGU_VLLM_ENDPOINTS=http://everest.nsu.ru:9022 bash deploy_ragu_v7.sh (the H100 gemma:
# 15 s mean per answer at 1 request, 26 s at 4, against 38 s / 118 s on the A6000, same answers); unset = unchanged
[[ -n "${RAGU_VLLM_ENDPOINTS:-}" ]] && set_env VLLM_ENDPOINTS "$RAGU_VLLM_ENDPOINTS"
# optional: the engine cross-encoder (search + source attribution), e.g. RAGU_ENGINE_RERANKER_URL=http://everest.nsu.ru:9028/v1
# (bge-reranker-v2-m3 on the H100, max-num-seqs 256: the same scores as :8002, 0.18 s instead of 0.45 s per 120 pairs)
[[ -n "${RAGU_ENGINE_RERANKER_URL:-}" ]] && set_env ENGINE_RERANKER_URL "$RAGU_ENGINE_RERANKER_URL"
# optional: how many searches (embedder + graph + cross-encoder) of all requests run at once (stand default 8)
[[ -n "${RAGU_EMBED_CONCURRENCY:-}" ]] && set_env EMBED_CONCURRENCY "$RAGU_EMBED_CONCURRENCY"
cat > "$HERE/compose.ragu-v3.yaml" <<YAML
# Graph v3 for the RAGU stand (deploy_ragu_v3.sh of $TS): its resources and its Qdrant.
services:
  core:
    environment:
      QDRANT_URL: http://127.0.0.1:6433
    volumes:
      - type: bind
        source: $STAND
        target: /app/data
        read_only: true
        bind:
          create_host_path: false
YAML

echo "== build and restart core"
compose_override up -d --no-deps --build --wait --wait-timeout 1800 core
for i in $(seq 160); do
    build=$(curl -fsS -m 5 http://127.0.0.1:9007/meno/v1/manifest 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("build",""))' 2>/dev/null || true)
    [[ $build == "${TARGET:0:7}" ]] && break
    sleep 15
done
[[ $build == "${TARGET:0:7}" ]] || { echo "the stand still reports build '$build'; see ./scripts/backend.sh logs --tail 100 core"; exit 1; }
curl -fsS http://127.0.0.1:9007/meno/v1/manifest; echo
name=$(curl -fsS -m 5 http://127.0.0.1:9007/meno/v1/manifest | python3 -c 'import json,sys; print(json.load(sys.stdin).get("name",""))')
[[ $name == "$NAME" ]] || { echo "the stand is named '$name', expected '$NAME'"; exit 1; }
docker inspect "$(compose_override ps -q core)" --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}resources: {{.Source}}{{end}}{{end}}'

echo "== check"
curl -fsS -N --max-time 300 http://127.0.0.1:9007/v1/chat/completions -H 'Content-Type: application/json' \
    -H "X-Meno-Request-Id: deploy-smoke-$TS" \
    -d '{"model": "meno-lite", "knowledge_base_id": "nsu-ragu", "stream": true, "messages": [{"role": "user", "content": "Какие предметы ведёт Рустам Мулляджанов?"}]}' \
    > "/tmp/ragu-smoke-$TS.sse"
python3 - "/tmp/ragu-smoke-$TS.sse" <<'PY'
import json, sys
text = open(sys.argv[1], encoding="utf-8").read()
events = [b for b in text.split("\n\n") if b.startswith("event: sources")]
answer_at = text.find('"content"')
assert len(events) == 1, f"expected one sources event, got {len(events)}"
assert answer_at != -1 and text.index("event: sources") > answer_at, "sources must come after the answer"
sources = json.loads(events[0].split("\n", 1)[1][6:])["sources"]
print(f"smoke ok: {len(sources)} sources after the answer")
for s in sources:
    print("  ", s["document_title"][:80], "->", s["source_url"])
PY
echo "graph v3 is live; rollback: sudo bash $HERE/rollback_ragu_v3.sh"

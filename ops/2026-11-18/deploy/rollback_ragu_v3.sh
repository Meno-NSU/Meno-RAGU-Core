#!/bin/bash
# Back to the graph, image, settings and source of before the graph v3 deploy of 20261118-091502. Run as root.
set -euo pipefail
if [[ -f /home/m.kulakov/meno-v2/ragu-v2/stand-v6k/knowledge/ragu_graph/qdrant_collections.json ]] && ! python3 -I /home/m.kulakov/meno-v2/ragu-probe/stand_parity.py /home/m.kulakov/meno-v2/ragu-v2/stand-v6k; then
    echo "/home/m.kulakov/meno-v2/ragu-v2/stand-v6k no longer matches its chunk collection; nothing changed. A consistent rollback:"
    echo "  sudo bash /home/m.kulakov/meno-v2/ragu-probe/deploy_ragu_v7.sh /home/m.kulakov/meno-v2/ragu-v2/stand-v5k"
    exit 1
fi
docker tag meno-ragu-core:before-v3-20261118-091502 meno-ragu-core:local
install -m 600 /home/rsolikhov/Meno-Deploy/.env.backend.before-v3-20261118-091502 /home/rsolikhov/Meno-Deploy/.env.backend
git -c safe.directory=/home/rsolikhov/meno_ragu_stand -C /home/rsolikhov/meno_ragu_stand reset -q --keep 6601cb490bf2f1795c285f8905bcc542161fa565
cd /home/rsolikhov/Meno-Deploy
if [[ "/home/m.kulakov/meno-v2/ragu-v2/stand-v6k" == /home/m.kulakov/meno-v2/ragu-v2/stand && -f "/home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v2.yaml" ]]; then
    # the stand ran graph v2 (deploy_ragu_v2.sh): back to it with its compose override
    docker compose --project-name meno-ragu-backend --project-directory /home/rsolikhov/Meno-Deploy --env-file /home/rsolikhov/Meno-Deploy/.env.backend \
        -f /home/rsolikhov/Meno-Deploy/compose.backend.yaml -f /home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v2.yaml up -d --no-deps --wait --wait-timeout 1800 core
elif [[ -n "/home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v3.before-20261118-091502.yaml" ]]; then
    # the stand ran graph v3 resources (/home/m.kulakov/meno-v2/ragu-v2/stand-v6k): back to them with the override of then
    cp /home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v3.before-20261118-091502.yaml /home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v3.yaml
    docker compose --project-name meno-ragu-backend --project-directory /home/rsolikhov/Meno-Deploy --env-file /home/rsolikhov/Meno-Deploy/.env.backend \
        -f /home/rsolikhov/Meno-Deploy/compose.backend.yaml -f /home/m.kulakov/meno-v2/ragu-probe/compose.ragu-v3.yaml up -d --no-deps --wait --wait-timeout 1800 core
else
    ./scripts/backend.sh up -d --no-deps --wait --wait-timeout 1800 core
fi
curl -fsS http://127.0.0.1:9007/meno/v1/manifest; echo

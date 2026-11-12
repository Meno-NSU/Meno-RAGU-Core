#!/bin/bash
# Waits for the entity labelling of label_v5 (entity_subtypes.py) to end, then retypes every relation of graph v4.
cd "$HOME/meno-v2/ragu-probe"
PY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python
while pgrep -u "$(id -u)" -f "entity_subtypes.py ontology/nsu.yaml label_v5" > /dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) entities done: $(wc -l < label_v5/entity_labels.jsonl) labels" >> label_v5/relations.log
LABEL_CONCURRENCY=48 nice "$PY" -u relation_retype.py ontology/nsu.yaml label_v5/relations.jsonl label_v5/entity_labels.jsonl \
    label_v5/relation_labels.jsonl >> label_v5/relations.log 2>&1 < /dev/null
echo "$(date -u +%FT%TZ) relations done: $(wc -l < label_v5/relation_labels.jsonl) labels" >> label_v5/relations.log

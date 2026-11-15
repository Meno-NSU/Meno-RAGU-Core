#!/bin/bash
# NSU-ontology labelling of graph v4 by gemma (Everest :9022), on prod, in ~/meno-v2/ragu-probe.
# usage: PY=<python with ragu+openai> label_run.sh pilot|full [concurrency]
#   pilot: the pilot sample (pilot_v5/relations.jsonl, entities.jsonl) labelled again with the current prompts into
#          pilot_v5b/ (whole names: the pilot's names are made whole the way label_inputs.py does it)
#   full:  every node and relation of graph v4 (label_inputs.py) into label_v5/
# Each step is resumable (the scripts skip what their output has); the log is label_run.<mode>.log.
set -euo pipefail
cd "$HOME/meno-v2/ragu-probe"
mode=$1
export LABEL_CONCURRENCY=${2:-24}
: "${PY:?PY must point to a python with ragu and openai}"
CK=$HOME/meno-v2/ragu-v2/graph-checkpoints-v4
case "$mode" in
  pilot) out=pilot_v5b; rel_in=$out/relations.jsonl;     ent_in=$out/entities.jsonl ;;
  full)  out=label_v5;  rel_in=$out/relations.jsonl;      ent_in=$out/entities.jsonl ;;
  *) echo "mode: pilot|full" >&2; exit 2 ;;
esac
mkdir -p "$out"
if [ ! -s "$out/relations.jsonl" ]; then
  if [ "$mode" = full ]; then nice "$PY" -I label_inputs.py "$CK" "$out" gv5
  else nice "$PY" -I label_inputs.py "$CK" "$out" gv5 --repair pilot_v5; fi
fi
nice "$PY" -u entity_subtypes.py ontology/nsu.yaml "$ent_in" "$out/entity_labels.jsonl"
nice "$PY" -u relation_retype.py ontology/nsu.yaml "$rel_in" "$out/entity_labels.jsonl" "$out/relation_labels.jsonl"
echo "done $(date -u +%FT%TZ)"

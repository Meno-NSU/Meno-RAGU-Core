#!/bin/bash
# Start the balanced-pool replay once the final1118 chain saw gemma back on the H100 (state/gw_restored.done) and the
# chain is done or failed; give up at 23:30 UTC 18.11.
D=$HOME/meno-v2/ragu-probe/v61; F=$HOME/meno-v2/final1118
END=$(date -u -d "2026-11-18 23:30:00" +%s)
echo "$(date -u +%FT%TZ) waiting for gemma back (final1118 state/gw_restored.done)" >> $D/after_bench.log
until [ -e $F/state/gw_restored.done ] && { [ -e $F/DONE ] || [ -e $F/FAILED ]; }; do
    [ "$(date +%s)" -ge $END ] && { echo "$(date -u +%FT%TZ) gave up" >> $D/after_bench.log; exit 0; }
    sleep 120
done
echo "$(date -u +%FT%TZ) starting run_pool_sim.sh" >> $D/after_bench.log
bash $D/run_pool_sim.sh
echo "$(date -u +%FT%TZ) run_pool_sim.sh done" >> $D/after_bench.log

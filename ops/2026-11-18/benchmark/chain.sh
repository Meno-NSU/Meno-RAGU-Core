#!/bin/bash
# Benchmark 18.11: RAGU stand-v6k2 against Menon 1.1 (rerun after the external audit of stand-v6k2: KB side rebuilt from
# KB v3.9b indexed chunks with repaired texts and annotations; code main d3c7326; the stand-v6k2 run is in archive-v6k/), exactly as the final benchmark of 16.11 ("Meno Eval v1": qwen
# nugget judge + RAGAS); prod side. Menon 1.1's 16.11 answers, qwen judgments and RAGAS are reused as they are.
#  1 wait   graph v6 chain step "done" (~/meno-v2/ragu-probe/v6/state/step); give up at 14:00 UTC 18.11 (FAILED + why)
#  2 verify stand-v6k2: document_quality.json (built with scripts/stand_document_quality.py from KB v3.9b when the stand
#           step wrote none), Qdrant collections green and non-empty, every menon_chunk_mapping id a point
#           (verify_stand.py); then in the answer process: graph nodes/edges against v5's, smoke run of 4 questions
#  3 answer RAGU v6k in-process (ragu_answers.py via run_answers.sh: code 6601cb4, the live container's settings), the
#           16.11 items in priority order core -> set B -> synthetic_kb -> no reference, adaptive concurrency against
#           the live-stand probe and the H100 metrics, a second pass for failed items
#  4 ship   blind judge tasks (make_tasks_final.py: point_v6c* = core + set B, point_v6s* = synthetic_kb), answers and
#           items to Everest final1118/inbox, READY; the Everest chain swaps gemma -> qwen on :9022 and judges
#  5 follow Everest state.json: gateway refresh + Menon check once qwen is up, interim.md once core + set B are judged,
#           gateway checks (refresh, one RAGU + one Menon request) once gemma is back
#  6 report analyze_eval_v1.py (same strata, weights, bootstrap seed 1115), RAGAS side by side, README.md here and on
#           Everest final1118/, DONE (or FAILED with the reason)
# Resumable: finished steps leave state/<step>.done. The Everest token comes on stdin and stays in this shell
# (not exported; handed to the helpers on their stdin). Status: status.txt (one line); log: chain.log.
# Start: launch.sh (token on its stdin).
set -u
F=$HOME/meno-v2/final1118
read -r EVT
exec < /dev/null
cd "$F" || exit 1
echo $$ > $F/chain.pid
B=$HOME/meno-v2/ragu-v2; S=$B/stand-v6k2; G=$B/bundle/knowledge/ragu_graph_v6
V6=$HOME/meno-v2/ragu-probe/v6; ST=$V6/state
CODE=$F/code-d3c7326; SPY=/home/rsolikhov/meno_ragu_stand/.venv/bin/python; VPY=$HOME/meno-v2/venv/bin/python
KBDOCS=$HOME/meno-v2/kb-v39b/kb/chunks/v1/docs.jsonl.gz
J=$F/judge
V5_NODES=346754; V5_EDGES=927166      # graph v5 as the stand loaded it on 16.11 (inproc/ragu_v5k_core.log)
V6_DEADLINE=$(date -u -d '2026-11-18 14:00:00' +%s)
mkdir -p $F/state $F/verify $F/answers $F/logs $F/ev

log() { echo "$(date -u +%FT%TZ) $*" >> $F/chain.log; }
status() { echo "$(date -u +%FT%TZ) $*" > $F/status.txt.tmp && mv $F/status.txt.tmp $F/status.txt; log "STATUS $*"; }
is_done() { [ -e $F/state/$1.done ]; }
done_step() { date -u +%FT%TZ > $F/state/$1.done; log "step $1 done"; }
free_gb() { df -B1G --output=avail $F | tail -1 | tr -dc 0-9; }
evget() { printf '%s\n' "$EVT" | timeout 900 python3 -I $F/evget.py "$1" "$2" > /dev/null 2>> $F/logs/ev.err; }
evput() { local d=$1; shift; printf '%s\n' "$EVT" | timeout 3600 python3 -I $F/evput_chunked.py "$d" "$@" >> $F/logs/ev.log 2>&1; }
fail() {
    status "FAILED: $*"
    { echo "$(date -u +%FT%TZ) FAILED: $*"; echo; echo "--- chain.log (tail)"; tail -30 $F/chain.log; } > $F/FAILED
    if ! is_done shipped; then echo "$*" > $F/ev/ABORT; evput final1118/inbox $F/ev/ABORT; fi
    evput final1118 $F/FAILED
    exit 1
}
log "chain.sh start (pid $$)"
[ -e $F/DONE ] && { log "DONE exists; nothing to do"; exit 0; }
rm -f $F/FAILED
for f in jobs.json smoke.json judge/cfg_v6c.json judge/cfg_v6s.json judge/cfg_analysis.json judge/keys/point_keys.json; do
    [ -s $F/$f ] || fail "setup file $f missing"
done

# ---------------------------------------------------------------- 1. graph v6
if ! is_done v6; then
    while true; do
        stp=$(cat $ST/step 2>/dev/null)
        if [ "$stp" = done ] && [ -e $ST/stand.done ] && [ -d $S/knowledge/ragu_graph ]; then break; fi
        bl=$(head -c 300 $ST/blocked 2>/dev/null); fl=$(head -c 300 $ST/failed 2>/dev/null)
        cp=$(cat $ST/chain.pid 2>/dev/null); al=$(kill -0 "$cp" 2>/dev/null && echo alive || echo NOT RUNNING)
        status "waiting for graph v6: step ${stp:-?}, v6 chain pid $cp $al${bl:+; blocked: $bl}${fl:+; failed: $fl}"
        if [ "$(date +%s)" -ge $V6_DEADLINE ]; then
            { echo "graph v6 chain did not reach step done by 2026-11-18 14:00 UTC"
              echo "step: $stp; v6 chain pid $cp: $al"; echo "blocked: ${bl:-none}"; echo "failed: ${fl:-none}"
              echo; echo "--- ~/meno-v2/ragu-v2/v6_status.txt"; cat $B/v6_status.txt
              echo; echo "--- ~/meno-v2/ragu-probe/v6/chain.log (tail)"; tail -25 $V6/chain.log; } > $F/verify/v6_wait_failed.txt
            fail "graph v6 not done by 14:00 UTC 18.11: step $stp, v6 chain $al${bl:+, blocked: $bl}${fl:+, failed: $fl} (verify/v6_wait_failed.txt)"
        fi
        sleep 120
    done
    log "graph v6 chain: $(tail -1 $V6/chain.log)"
    done_step v6
fi

# ---------------------------------------------------------------- 2. verify stand-v6k2
if ! is_done verify; then
    status "verifying stand-v6k2 (document quality, collections, chunk mapping)"
    [ "$(free_gb)" -ge 16 ] || fail "free disk $(free_gb) GB < 16 GB before the verification"
    if [ ! -e $S/document_quality.json ]; then
        log "document_quality.json: building from $KBDOCS"
        (cd $CODE && PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 timeout 3600 nice $SPY scripts/stand_document_quality.py $S $KBDOCS) \
            > $F/verify/document_quality.log 2>&1 || fail "document_quality.json build failed: $(tail -2 $F/verify/document_quality.log | tr '\n' ' ')"
        chmod 664 $S/document_quality.json; chmod o+r $S/document_quality.json
        echo "built by ~/meno-v2/final1118/chain.sh $(date -u +%FT%TZ) from $KBDOCS" >> $F/verify/document_quality.log
    else
        echo "present before the chain (written by the stand step): $(python3 -c "import json; print(len(json.load(open('$S/document_quality.json'))))") documents change quality" \
            > $F/verify/document_quality.log
    fi
    echo "stand-v5k for comparison: $(python3 -c "import json; print(len(json.load(open('$B/stand-v5k/document_quality.json'))))") documents change quality" >> $F/verify/document_quality.log
    log "document quality: $(head -1 $F/verify/document_quality.log)"
    EXCL=$(sed -n 's/^[[:space:]]*exclude_chunk_ids[[:space:]]*=[[:space:]]*//p' $ST/stand_kb_ok 2>/dev/null | tail -1)
    GRAPH_TAG=ragu_graph_v6 timeout 3600 python3 -I $F/verify_stand.py $S $G $F/verify/verify.json "$EXCL" > $F/verify/verify.log 2>&1 \
        || fail "stand-v6k2 check failed: $(grep -E '^FAIL|Traceback|Error' $F/verify/verify.log | head -5 | cut -c1-200 | tr '\n' ' ')"
    log "verify: $(tail -1 $F/verify/verify.log)"
    done_step verify
fi

# ---------------------------------------------------------------- 3. RAGU v6k answers
if ! is_done answers; then
    [ -s $F/state/answers_deadline ] || date -u -d '+7 hours' +%FT%TZ > $F/state/answers_deadline
    HARD=$(( $(date -u -d "$(cat $F/state/answers_deadline)" +%s) + 3600 ))
    run=0
    while true; do
        run=$((run + 1))
        SM=(--smoke $F/smoke.json); grep -qs 'VERDICT: OK' $F/verify/smoke.md && SM=()
        status "answers: loading stand-v6k2 (run $run${SM:+; graph check and smoke run first})"
        setsid bash $F/run_answers.sh $S $F/jobs.json $F/logs/answers.log "${SM[@]}" \
            --graph-check $V5_NODES,$V5_EDGES,0.85,1.6 --resources-json $F/verify/resources.json \
            --adaptive --start 8 --min 2 --max 24 --fast 40 --slow 50 --probe-out $F/probe.jsonl \
            --progress-file $F/answers_progress.txt --deadline "$(cat $F/state/answers_deadline)" &
        RP=$!
        echo $RP > $F/state/answers.pid
        log "answer run $run pid $RP (pgid $(ps -o pgid= -p $RP | tr -d ' ')), deadline $(cat $F/state/answers_deadline)"
        while kill -0 $RP 2>/dev/null; do
            [ -s $F/answers_progress.txt ] && status "$(cut -c22- $F/answers_progress.txt)"
            if [ "$(date +%s)" -ge $HARD ]; then
                log "answer run past its hard limit: stopping group $RP"
                kill -TERM -- -$RP; sleep 60; kill -KILL -- -$RP 2>/dev/null
            fi
            sleep 60
        done
        wait $RP; rc=$?
        log "answer run $run exit $rc"
        # 0 all answered, 1 some failed or the deadline, 3 smoke, 4 graph check: final; else (a crash) resume, 3 runs at most
        case $rc in 0|1|3|4) break;; esac
        [ $run -ge 3 ] && break
        [ "$(date +%s)" -ge $(date -u -d "$(cat $F/state/answers_deadline)" +%s) ] && break
        log "answer run crashed (rc $rc): $(grep -E 'Error|Traceback' $F/logs/answers.log | tail -2 | cut -c1-200 | tr '\n' ' '); resuming in 60 s"
        sleep 60
    done
    case $rc in
        3) fail "smoke run failed on stand-v6k2: $(grep VERDICT $F/verify/smoke.md 2>/dev/null) (verify/smoke.md)";;
        4) fail "graph check failed: $(python3 -c "import json; print(json.load(open('$F/verify/resources.json')).get('check'))" 2>/dev/null)";;
    esac
    grep -qs 'VERDICT: OK' $F/verify/smoke.md || fail "no smoke verdict (answer run exit $rc): $(grep -E 'Error|Traceback' $F/logs/answers.log | tail -2 | cut -c1-200 | tr '\n' ' ')"
    python3 $F/merge_records.py $F > $F/logs/merge.log 2>&1 || fail "answers incomplete (exit $rc): $(tail -2 $F/logs/merge.log | tr '\n' ' ')"
    log "answers: $(head -1 $F/logs/merge.log)"
    done_step answers
fi

# ---------------------------------------------------------------- 4. judge tasks, ship to Everest
if ! is_done shipped; then
    status "building the blind judge tasks and shipping to Everest"
    (cd $J && python3 make_tasks_final.py cfg_v6c.json --force && python3 make_tasks_final.py cfg_v6s.json --force) > $F/logs/make_tasks.log 2>&1 \
        || fail "make_tasks failed: $(tail -3 $F/logs/make_tasks.log | tr '\n' ' ')"
    log "tasks: $(grep -h 'task files' $F/logs/make_tasks.log | tr '\n' ' ')"
    (cd $F && tar -cJf $F/ship.tar.xz judge/judge_llm.py judge/prompts $(cd $F && ls judge/tasks/point_v6*.json) items/items_v3new.jsonl \
        items/setB_items.jsonl -C $F/answers ragu6k_v3.records.jsonl ragu6k_setB.records.jsonl) || fail "ship.tar.xz failed"
    sha256sum $F/ship.tar.xz | cut -d' ' -f1 > $F/ev/READY
    evput final1118/inbox $F/ship.tar.xz || fail "upload of ship.tar.xz failed: $(tail -2 $F/logs/ev.log | tr '\n' ' ')"
    evput final1118/inbox $F/ev/READY || fail "upload of READY failed"
    log "shipped: ship.tar.xz $(stat -c %s $F/ship.tar.xz) bytes"
    done_step shipped
fi

# ---------------------------------------------------------------- 5. follow the Everest chain
EV_DEADLINE=$(( $(date -u -d "$(cat $F/state/shipped.done)" +%s) + 13 * 3600 ))
marker() { python3 -c "import json,sys; d=json.load(open('$F/ev/state.json')); m=d['markers'].get(sys.argv[1]); print(m if m else '')" "$1" 2>/dev/null; }
errs=0
while ! is_done everest; do
    if evget final1118/state.json $F/ev/state.json.new && python3 -c "import json; json.load(open('$F/ev/state.json.new'))" 2>/dev/null; then
        mv $F/ev/state.json.new $F/ev/state.json; errs=0
    else
        errs=$((errs + 1)); log "Everest state not readable ($errs)"
    fi
    if [ -s $F/ev/state.json ]; then
        ph=$(python3 -c "import json; print(json.load(open('$F/ev/state.json'))['phase'][21:])" 2>/dev/null)
        status "Everest: $ph"
        h=$(marker HOLD); [ -n "$h" ] && ! is_done hold_seen && { log "Everest reports a hold on the H100: $h"; done_step hold_seen; }
        if [ -n "$(marker QWEN_UP)" ] && ! is_done swap_gw; then
            python3 -I $F/gw_check.py swap $F/verify/gw_swap.json > $F/logs/gw_swap.log 2>&1
            log "gateway during the swap: $(tail -1 $F/logs/gw_swap.log)"; done_step swap_gw
        fi
        if [ -n "$(marker JUDGE_CORE_DONE)" ] && ! is_done interim; then
            if evget final1118/out_core.tar.gz $F/ev/out_core.tar.gz && tar -xzf $F/ev/out_core.tar.gz -C $F/ev; then
                cp $F/ev/out/point_v6c*.qwen.jsonl $J/out/
                (cd $J && timeout 1800 $VPY analyze_eval_v1.py cfg_analysis.json eval_v1 $F/interim_tables --judges qwen --pair ragu6k,menon) \
                    > $F/logs/analysis_interim.log 2>&1
                timeout 600 $VPY $F/report.py interim $F > $F/logs/report_interim.log 2>&1
                log "interim: $(tail -1 $F/logs/report_interim.log)"
                evput final1118 $F/interim.md
                done_step interim
            fi
        fi
        if [ -n "$(marker GEMMA_RESTORED)" ] && ! is_done gw_restored; then
            if python3 -I $F/gw_check.py restored $F/verify/gw_restored.json > $F/logs/gw_restored.log 2>&1; then
                log "gateway after the restore: $(tail -1 $F/logs/gw_restored.log)"
            else
                log "ALERT gateway after the restore NOT ok: $(tail -1 $F/logs/gw_restored.log)"
            fi
            done_step gw_restored
        fi
        [ -n "$(marker GEMMA_FAILED)" ] && ! is_done gemma_alert && { log "ALERT Everest: gemma NOT restored: $(marker GEMMA_FAILED)"; done_step gemma_alert; }
        if [ -n "$(marker ALL_DONE)" ]; then done_step everest; break; fi
        a=$(marker ABORTED)
        if [ -n "$a" ]; then
            if [ -z "$(marker SWAP_STARTED)" ] || [ -n "$(marker GEMMA_RESTORED)$(marker GEMMA_FAILED)" ]; then
                [ -n "$(marker GEMMA_RESTORED)" ] && ! is_done gw_restored && continue
                fail "Everest chain aborted: $a; gemma: $( [ -n "$(marker SWAP_STARTED)" ] && echo "$(marker GEMMA_RESTORED)$(marker GEMMA_FAILED)" || echo 'never stopped' )"
            fi
        fi
    fi
    [ "$(date +%s)" -ge $EV_DEADLINE ] && fail "Everest chain not done 13 h after shipping (phase: ${ph:-?})"
    sleep 90
done

# ---------------------------------------------------------------- 6. analysis and report
if ! is_done report; then
    status "analysis and report"
    evget final1118/results.tar.gz $F/ev/results.tar.gz || fail "results.tar.gz download failed"
    rm -rf $F/ev/results && mkdir -p $F/ev/results && tar -xzf $F/ev/results.tar.gz -C $F/ev/results || fail "results unpack failed"
    cp $F/ev/results/judge/out/point_v6*.qwen.jsonl $J/out/
    cp $F/ev/results/judge/out/qwen.runs.jsonl $J/out/qwen.runs.v6k.jsonl 2>/dev/null
    rm -rf $F/ragas && cp -r $F/ev/results/ragas $F/ragas 2>/dev/null
    (cd $J && timeout 1800 $VPY analyze_eval_v1.py cfg_analysis.json eval_v1 out_eval_v1 --judges qwen --pair ragu6k,menon) \
        > $F/logs/analysis.log 2>&1 || fail "analysis failed: $(tail -3 $F/logs/analysis.log | tr '\n' ' ')"
    timeout 1800 $VPY $F/ragas_side.py $F/ragas_side.json $F/ragas_side.md \
        v3_ref=$F/ragas/v3_ragu6k_ref,$HOME/meno-v2/final1116/everest_results/ragas/v3_menon_ref \
        B=$F/ragas/B_ragu6k,$HOME/meno-v2/final1116/everest_results/ragas/B_menon \
        v3_noref=$F/ragas/v3_ragu6k_noref,$HOME/meno-v2/final1116/everest_results/ragas/v3_menon_noref > $F/logs/ragas_side.log 2>&1 \
        || log "ragas_side failed: $(tail -2 $F/logs/ragas_side.log | tr '\n' ' ')"
    timeout 600 $VPY $F/report.py final $F > $F/logs/report.log 2>&1 || fail "report failed: $(tail -3 $F/logs/report.log | tr '\n' ' ')"
    (cd $F && tar -czf $F/analysis_bundle.tar.gz judge/out_eval_v1 judge/keys judge/cfg_*.json judge/make_tasks_final.py \
        judge/analyze_eval_v1.py judge/nightscore.py judge/judge_llm.py judge/prompts judge/eval_v1 verify answers/answers_summary.json \
        ragas_side.json ragas_side.md interim.md README.md chain.sh launch.sh run_answers.sh ragu_answers.py prep_items.py \
        merge_records.py verify_stand.py gw_check.py report.py ragas_side.py jobs.json smoke.json ids logs/answers.log \
        chain.log probe.jsonl 2>/dev/null)
    evput final1118 $F/README.md || log "README upload failed"
    evput final1118/analysis $F/analysis_bundle.tar.gz $F/judge/out_eval_v1/eval_v1_tables.md $F/judge/out_eval_v1/eval_v1.json \
        $F/judge/out_eval_v1/eval_v1_rows.csv $F/ragas_side.md $F/ragas_side.json $F/interim.md $F/judge/keys/point_keys.json \
        $F/answers/answers_summary.json $F/verify/smoke.md $F/verify/verify.json $F/verify/resources.json \
        $F/verify/document_quality.log || log "analysis upload failed"
    done_step report
fi
echo "$(date -u +%FT%TZ) $(head -1 $F/logs/report.log 2>/dev/null)" > $F/DONE
evput final1118 $F/DONE
status "DONE: README.md (here and on Everest /workspace/data/final1118/)"
log "chain.sh done"

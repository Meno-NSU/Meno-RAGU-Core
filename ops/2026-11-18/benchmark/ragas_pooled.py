"""RAGAS of the 18.11 benchmark pooled over the strata: RAGU v6k2 (this run) vs Menon 1.1 (16.11, reused), both
answer_correctness and faithfulness over all referenced questions (core + synthetic_kb + set B). Pooled = every
question counts once (strata weighted by their size); macro = the three strata weigh the same. Paired differences
over the questions both systems have a score for; 95% CI by bootstrap over questions (10,000, seed 1115; macro:
resampled within each stratum)."""
import csv, glob, json, math, os, random
F = os.path.expanduser('~/meno-v2/final1118')
M16 = os.path.expanduser('~/meno-v2/final1116/everest_results/ragas')
RUNS = {'ragu': [f'{F}/ragas/v3_ragu6k_ref', f'{F}/ragas/B_ragu6k'], 'menon': [f'{M16}/v3_menon_ref', f'{M16}/B_menon']}
STRATA = {s: set(json.load(open(f'{F}/ids/{f}.json'))) for s, f in (('core', 'core'), ('synth', 'synth'), ('B', 'setB'))}
METRICS = ('answer_correctness', 'faithfulness')

def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None

scores = {s: {m: {} for m in METRICS} for s in RUNS}
for system, dirs in RUNS.items():
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, 'batch_*.csv'))):
            for r in csv.DictReader(open(p, encoding='utf-8')):
                for m in METRICS:
                    x = num(r.get(m))
                    if x is not None:
                        scores[system][m][r['item_id']] = x
stratum = {i: s for s, ids in STRATA.items() for i in ids}

def mean(xs): return sum(xs) / len(xs) if xs else float('nan')

def boot(groups, macro, n=10000, seed=1115):
    rng = random.Random(seed)
    est = []
    for _ in range(n):
        if macro:
            est.append(mean([mean([g[rng.randrange(len(g))] for _ in range(len(g))]) for g in groups]))
        else:
            allv = [x for g in groups for x in g]
            est.append(mean([allv[rng.randrange(len(allv))] for _ in range(len(allv))]))
    est.sort()
    return est[int(.025 * n)], est[int(.975 * n)]

out = {}
print(f"{'метрика':20} {'вариант':26} {'RAGU v6k2':>10} {'Менон 1.1':>10} {'разность':>9}  95% ДИ          n пар")
for m in METRICS:
    a, b = scores['ragu'][m], scores['menon'][m]
    common = [i for i in a if i in b and i in stratum]
    per = {s: [a[i] - b[i] for i in common if stratum[i] == s] for s in STRATA}
    rows = {}
    for s in STRATA:
        ia = [a[i] for i in a if stratum.get(i) == s]; ib = [b[i] for i in b if stratum.get(i) == s]
        rows[s] = (mean(ia), mean(ib), mean(per[s]), boot([per[s]], False), len(per[s]))
    pa = [a[i] for i in a if i in stratum]; pb = [b[i] for i in b if i in stratum]
    alld = [d for s in STRATA for d in per[s]]
    rows['pooled'] = (mean(pa), mean(pb), mean(alld), boot([per[s] for s in STRATA], False), len(alld))
    rows['macro'] = (mean([rows[s][0] for s in STRATA]), mean([rows[s][1] for s in STRATA]),
                     mean([rows[s][2] for s in STRATA]), boot([per[s] for s in STRATA], True), len(alld))
    out[m] = rows
    for k, label in (('core', 'основной слой'), ('synth', 'синтетика по базе'), ('B', 'набор B'),
                     ('pooled', 'все вопросы (по числу)'), ('macro', 'среднее трёх слоёв')):
        ra, rb, d, (lo, hi), n = rows[k]
        print(f"{m:20} {label:26} {ra:10.3f} {rb:10.3f} {d:+9.3f}  [{lo:+.3f}; {hi:+.3f}]  {n}")
json.dump({m: {k: list(v[:3]) + [list(v[3]), v[4]] for k, v in r.items()} for m, r in out.items()},
          open(f'{F}/ragas_pooled.json', 'w'), ensure_ascii=False, indent=1)

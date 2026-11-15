"""Every number of the RAGAS benchmark (954 questions) for all pipelines, for the comparison page: metric means with
bootstrap CI, distributions, paired differences against Menon 1.1 · gemma, answers (length, refusals), contexts
(blocks, size), sources (count), tokens and timings where the run recorded them, and breakdowns by
question category and topic.

Runs: Menon 1.0 / 1.1 × meno-lite / gemma / OCC-RAG (out/m1x-*), RAGU NSUv1 × meno-lite / gemma (R. Solikhov's
harness, nsuv1/nsuv1_slim.jsonl), RAGU v3 / v3k × gemma (out/ragu-v3*-gemma; context precision from out/*-cp, scored on
one block per prompt document / graph element). usage: python all_stats.py > report/pipelines/data.json
"""
import json, re, sys
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = {"answer_correctness": "answer_correctness", "factual_correctness": "factual_correctness(mode=f1)",
           "faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy", "context_recall": "context_recall",
           "context_precision": "llm_context_precision_with_reference", "context_entity_recall": "context_entity_recall"}
RUNS = [  # id, pipeline, model, harness
    ("m10-lite", "Менон 1.0", "meno-lite", "ours"), ("m10-gemma", "Менон 1.0", "gemma", "ours"),
    ("m10-occ", "Менон 1.0", "OCC-RAG", "ours"),
    ("m11-lite", "Менон 1.1", "meno-lite", "ours"), ("m11-gemma", "Менон 1.1", "gemma", "ours"),
    ("m11-occ", "Менон 1.1", "OCC-RAG", "ours"),
    ("nsuv1-lite", "RAGU NSUv1", "meno-lite", "solikhov"), ("nsuv1-gemma", "RAGU NSUv1", "gemma", "solikhov"),
    ("ragu-v3-gemma", "RAGU v3", "gemma", "ours"), ("ragu-v3k-gemma", "RAGU v3k", "gemma", "ours"),
]
REFERENCE = "m11-gemma"
# a refusal-like opening: the answer starts by saying the documents lack the information (as export_dash.py)
LACK = re.compile(r"нет (?:информации|данных|сведений|описания)|не содерж|недостаточно (?:информации|данных)|не указан|"
                  r"не упомин|не удалось (?:найти|определить)|отсутству", re.I)


def norm(url):
    p = urlsplit((url or "").strip().lower())
    return f"{p.netloc.removeprefix('www.')}{p.path.rstrip('/')}"


def scores_of(run):
    frames = [pd.read_csv(f) for f in sorted((HERE / "out" / run).glob("batch_*.csv"))]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("item_id").set_index("item_id")
    out = pd.DataFrame({k: df[v] if v in df else np.nan for k, v in METRICS.items()}, index=df.index)
    cp_dirs = [d for d in (HERE / "out" / f"{run}-cp", HERE / "out" / f"{run}-cp-fill") if d.exists()]
    if cp_dirs:  # only the re-scored blocks count; the fill pass scores the items the first pass left unscored
        frames = [pd.read_csv(f, usecols=["item_id", METRICS["context_precision"]]) for d in cp_dirs
                  for f in sorted(d.glob("batch_*.csv"))]
        c = pd.concat(frames).dropna().drop_duplicates("item_id").set_index("item_id")[METRICS["context_precision"]]
        out["context_precision"] = c.reindex(out.index)
    out["response"] = df["response"].fillna("")
    return out


def records_of(run):
    path = HERE / "records" / (f"{run}-cp.jsonl" if run.startswith("ragu") else f"{run}.jsonl")
    rows = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        ctx = [c.get("text") or "" for c in r.get("contexts") or []]
        stages = r.get("stages_ms") or {}
        rows[r["item_id"]] = {
            "contexts": len(ctx), "context_chars": sum(map(len, ctx)),
            "graph_elements": sum(t.startswith("**Элемент графа") for t in ctx),
            "sources": [norm(s.get("source_url")) for s in r.get("sources") or []],
            "latency_s": r.get("total_s"), "ttft_s": r.get("ttft_s"),
            "queries": len(r["search_queries"]) if isinstance(r.get("search_queries"), list) else None,
            **{f"stage_{k}": v for k, v in stages.items()},
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def nsuv1(model):
    rows = {}
    for line in (HERE / "nsuv1" / "nsuv1_slim.jsonl").read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["model"].startswith(model):
            continue
        item = f"legacy-{int(r['index']):03d}"
        s = r["scores"]
        rows[item] = {
            **{k: s.get(k, np.nan) for k in METRICS},  # the five metrics of his harness; the other two stay NaN
            "response": r["response"], "contexts": r["contexts"], "context_chars": r["context_chars"],
            "graph_elements": np.nan, "sources": [norm(x["url"]) for x in r["sources"]],
            "latency_s": (r["total_ms"] or 0) / 1000 or r["elapsed_s"], "ttft_s": None, "queries": r["queries"],
            "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            **{f"stage_{k}": v for k, v in (r["stages_ms"] or {}).items()},
        }
    return pd.DataFrame.from_dict(rows, orient="index")


items = {}
for line in (Path.home() / "Meno-QA-Benchmark/data/v3/items.jsonl").read_text(encoding="utf-8").split("\n"):
    if line.strip():
        it = json.loads(line)
        rid = (it.get("provenance") or {}).get("ragas_run_id")
        if rid:
            items[rid] = it
IDS = sorted(items)

data = {}
for run, pipeline, model, harness in RUNS:
    if run.startswith("nsuv1"):
        df = nsuv1("meno-lite" if run.endswith("lite") else "gemma")
    else:
        df = scores_of(run).join(records_of(run), how="left")
    data[run] = df.reindex(IDS)

rng = np.random.default_rng(0)


def boot(values):
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return None
    b = rng.choice(v, (2000, len(v))).mean(axis=1)
    return [round(float(v.mean()), 4), round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def q(series, p):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(s.quantile(p)), 2) if len(s) else None


out = {"items": len(IDS), "reference": REFERENCE, "runs": [], "paired": {}, "by_category": {}, "by_topic": {},
       "histograms": {}, "stages": {}}
cats = sorted({items[i]["category"] for i in IDS})
topics = [t for t, _ in pd.Series([items[i]["topic"] for i in IDS]).value_counts().items()][:8]
for run, pipeline, model, harness in RUNS:
    df = data[run]
    resp = df["response"].fillna("").astype(str)
    answered = resp.str.strip() != ""
    row = {"id": run, "pipeline": pipeline, "model": model, "harness": harness, "answered": int(answered.sum()),
           "metrics": {}, "answers": {}, "contexts": {}, "sources": {}, "cost": {}}
    for k in METRICS:
        v = pd.to_numeric(df[k], errors="coerce").to_numpy(dtype=float)
        ci = boot(v)
        if ci is None:
            continue
        ok = v[~np.isnan(v)]
        row["metrics"][k] = {"mean": ci[0], "low": ci[1], "high": ci[2], "n": int(len(ok)),
                             "median": round(float(np.median(ok)), 4), "zero": round(float((ok == 0).mean()), 4),
                             "high_share": round(float((ok >= 0.7).mean()), 4)}
        out["histograms"].setdefault(k, {})[run] = np.histogram(ok, bins=10, range=(0, 1))[0].tolist()
    words = resp.str.split().str.len()
    row["answers"] = {
        "chars_mean": round(float(resp[answered].str.len().mean())), "chars_median": int(resp[answered].str.len().median()),
        "chars_p10": int(resp[answered].str.len().quantile(.1)), "chars_p90": int(resp[answered].str.len().quantile(.9)),
        "words_median": int(words[answered].median()), "words_mean": round(float(words[answered].mean())),
        "lack_opening": round(float(resp.str[:220].str.contains(LACK).mean()), 4),
        "lack_anywhere": round(float(resp.str.contains(LACK).mean()), 4),
        "relevancy_zero": round(float((pd.to_numeric(df["answer_relevancy"], errors="coerce") == 0).mean()), 4),
        "markdown_lists": round(float(resp.str.contains(r"^\s*(?:[*-]|\d+\.)\s", regex=True, flags=re.M).mean()), 4),
    }
    row["contexts"] = {"blocks_mean": round(float(df["contexts"].mean()), 2), "blocks_median": q(df["contexts"], .5),
                       "chars_mean": round(float(df["context_chars"].mean())), "chars_median": q(df["context_chars"], .5),
                       "chars_p90": q(df["context_chars"], .9),
                       "graph_elements_mean": None if df["graph_elements"].isna().all() else round(float(df["graph_elements"].mean()), 2),
                       "empty": round(float((df["contexts"] == 0).mean()), 4)}
    nsrc = df["sources"].apply(lambda s: len(s) if isinstance(s, list) else np.nan)
    row["sources"] = {"mean": round(float(nsrc.mean()), 2), "median": q(nsrc, .5), "none": round(float((nsrc == 0).mean()), 4)}
    cost = {}
    for col, name in (("latency_s", "latency_s"), ("ttft_s", "ttft_s"), ("prompt_tokens", "prompt_tokens"),
                      ("completion_tokens", "completion_tokens"), ("queries", "queries")):
        if col in df and pd.to_numeric(df[col], errors="coerce").notna().any():
            cost[name] = {"median": q(df[col], .5), "p90": q(df[col], .9), "mean": round(float(pd.to_numeric(df[col], errors="coerce").mean()), 2)}
    row["cost"] = cost
    out["stages"][run] = {c.removeprefix("stage_"): q(df[c], .5) for c in df.columns if c.startswith("stage_")}
    out["runs"].append(row)
    for key, field, groups in (("by_category", "category", cats), ("by_topic", "topic", topics)):
        out[key][run] = {m: {g: boot(pd.to_numeric(df[m], errors="coerce")[[items[i][field] == g for i in IDS]].to_numpy(dtype=float))
                             for g in groups} for m in ("answer_correctness", "factual_correctness")}
out["category_sizes"] = {c: sum(items[i]["category"] == c for i in IDS) for c in cats}
out["topic_sizes"] = {t: sum(items[i]["topic"] == t for i in IDS) for t in topics}

ref = data[REFERENCE]
for run, *_ in RUNS:
    if run == REFERENCE:
        continue
    res = {}
    for k in METRICS:
        a = pd.to_numeric(ref[k], errors="coerce")
        b = pd.to_numeric(data[run][k], errors="coerce")
        x = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        if len(x) < 30:
            continue
        d = (x["b"] - x["a"]).to_numpy()
        bs = rng.choice(d, (2000, len(d))).mean(axis=1)
        res[k] = {"diff": round(float(d.mean()), 4), "low": round(float(np.percentile(bs, 2.5)), 4),
                  "high": round(float(np.percentile(bs, 97.5)), 4), "n": int(len(d)),
                  "better": int((d > 0.05).sum()), "same": int((abs(d) <= 0.05).sum()), "worse": int((d < -0.05).sum())}
    out["paired"][run] = res
json.dump(out, sys.stdout, ensure_ascii=False, indent=1)

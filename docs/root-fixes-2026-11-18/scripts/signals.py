"""Which signal tells the unsupported sentences of an answer? (#77, no new judge run) For the core v6k2 answers of the
18.11 benchmark, per sentence: s_gemma (support_pilot.py: gemma's own 0/1/2 check), s_ce (the best cross-encoder
score of the sentence against the context, split into fragments of up to ~700 characters: bge-reranker-v2-m3, the
stand's attribution model) and s_num (numbers, e-mails and links of the sentence that occur nowhere in the context,
the question or the history). The qwen judge's unsupported and contradicting claims are mapped to the sentence they
share most words with; sentence and answer level AUCs.
usage: signals.py RECORDS.jsonl LABELS.jsonl PILOT.jsonl OUT.jsonl RERANKER_URL MODEL   |   --summary OUT.jsonl
"""
import asyncio, json, re, sys

WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")
MAIL_URL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+|https?://\S+|www\.\S+", re.I)


def stems(text):
    return {w.lower()[:5] for w in WORD.findall(text) if len(w) > 2}


def specifics(text):
    flat = re.sub(r"(?<=\d)[\s  ](?=\d{3}\b)", "", text)  # "5 000" is one number
    return {m.rstrip(".,") for m in NUM.findall(flat)} | {m.rstrip(".,;)").lower() for m in MAIL_URL.findall(text)}


def fragments(pieces, size=700):
    out = []
    for piece in pieces:
        buf = ""
        for para in re.split(r"\n\s*\n", piece):
            para = para.strip()
            if not para:
                continue
            if buf and len(buf) + len(para) > size:
                out.append(buf); buf = ""
            buf = (buf + "\n" + para) if buf else para
            while len(buf) > 2 * size:
                out.append(buf[:size]); buf = buf[size:]
        if buf:
            out.append(buf)
    return out


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


async def main(rec_path, lab_path, pilot_path, out_path, url, model):
    import httpx
    from meno_rag.stand.attribution import RerankerScorer, split_sentences
    lab = {l["item_id"]: l for l in map(json.loads, open(lab_path, encoding="utf-8"))}
    pilot = {r["item_id"]: {s["sentence"]: s["score"] for s in r["sentences"]}
             for r in map(json.loads, open(pilot_path, encoding="utf-8"))}
    items = {}
    for l in open("/home/m.kulakov/meno-v2/final1118/items/items_v3new.jsonl", encoding="utf-8"):
        if l.strip():
            i = json.loads(l); items[i["id"]] = i
    gate = asyncio.Semaphore(4); out = open(out_path, "w", encoding="utf-8")
    async with httpx.AsyncClient(timeout=60) as http:
        scorer = RerankerScorer(http, url, model, "")

        async def one(r):
            it = items.get(r["item_id"], {}); hist = it.get("history") or []
            if isinstance(hist, str):
                hist = []
            pieces = [c.get("text") or "" for c in r.get("contexts") or []]
            known = specifics("\n".join(pieces) + "\n" + it.get("question", "") + "\n"
                              + "\n".join(h.get("content") or "" for h in hist if isinstance(h, dict)))
            frags = fragments(pieces)
            sents = split_sentences(r["response"].strip())
            claims = [("unsupported", c) for c in lab[r["item_id"]]["unsupported"]] + \
                     [("contradicts", c) for c in lab[r["item_id"]]["contradicts"]]
            bad = {}
            for kind, c in claims:
                cs = stems(str(c)); best, bi = 0.0, None
                for k, s in enumerate(sents):
                    ss = stems(s); inter = len(cs & ss)
                    j = inter / max(len(cs | ss), 1)
                    if inter >= 2 and j > best:
                        best, bi = j, k
                if bi is not None:
                    bad.setdefault(bi, []).append(kind)
            rows = []
            async with gate:
                for k, s in enumerate(sents):
                    ce = max(await scorer(s, frags)) if frags else 0.0
                    rows.append({"sentence": s, "s_ce": round(ce, 4), "s_gemma": pilot.get(r["item_id"], {}).get(s),
                                 "s_num": sorted(specifics(s) - known), "bad": bad.get(k, [])})
            out.write(json.dumps({"item_id": r["item_id"], "n_claims_mapped": len(bad),
                                  "n_claims": len(claims), "sentences": rows}, ensure_ascii=False) + "\n")
        recs = [r for r in map(json.loads, open(rec_path, encoding="utf-8")) if r["item_id"] in lab]
        await asyncio.gather(*(one(r) for r in recs))


def summary(out_path):
    rows = list(map(json.loads, open(out_path, encoding="utf-8")))
    sent = [s for r in rows for s in r["sentences"]]
    claims = sum(r["n_claims"] for r in rows); mapped = sum(r["n_claims_mapped"] for r in rows)
    bad = [s for s in sent if s["bad"]]; good = [s for s in sent if not s["bad"]]
    print(f"{len(rows)} answers, {len(sent)} sentences; judge claims {claims}, mapped to {mapped} sentences; "
          f"bad sentences {len(bad)}")
    for name, f in (("gemma 1-score", lambda s: 1 - (s["s_gemma"] if s["s_gemma"] is not None else 1)),
                    ("cross-encoder 1-max", lambda s: 1 - s["s_ce"]),
                    ("ungrounded specifics", lambda s: float(bool(s["s_num"])))):
        print(f"sentence AUC {name}: {auc([f(s) for s in bad], [f(s) for s in good]):.3f}")
    for th in (0.1, 0.2, 0.3, 0.5):
        tp = sum(s["s_ce"] < th for s in bad); fp = sum(s["s_ce"] < th for s in good)
        print(f"   cross-encoder < {th}: {tp}/{len(bad)} bad ({tp / max(len(bad), 1):.1%}), "
              f"{fp}/{len(good)} other ({fp / max(len(good), 1):.1%}); precision {tp / max(tp + fp, 1):.1%}")
    tp = sum(bool(s["s_num"]) for s in bad); fp = sum(bool(s["s_num"]) for s in good)
    print(f"   ungrounded specifics: {tp}/{len(bad)} bad ({tp / max(len(bad), 1):.1%}), {fp}/{len(good)} other "
          f"({fp / max(len(good), 1):.1%}); precision {tp / max(tp + fp, 1):.1%}")
    ex = [s for s in good if s["s_num"]][:6]
    for s in ex:
        print("   other with specifics:", s["s_num"][:4], s["sentence"][:120])


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2])
    else:
        asyncio.run(main(*sys.argv[1:7]))

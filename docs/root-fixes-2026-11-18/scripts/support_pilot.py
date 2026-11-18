"""Can gemma tell which sentences of an answer its sources do not support? A probe for grounded generation (#77),
measured against the labels the qwen judge already gave (no new judge run): every sentence of a RAGU v6k2 answer of the
18.11 benchmark is scored against the context the answer was written from (the final evidence of its record) on the
0/1/2 scale of the LLM rerank (guided choice, probabilities from logprobs). The answer, then the sentence, come after
the sources, so the sentences of one answer share the prompt prefix (vLLM prefix cache).
usage: support_pilot.py RECORDS.jsonl OUT.jsonl MODEL BASE_URL [CONCURRENCY] [LIMIT]
       support_pilot.py --summary OUT.jsonl LABELS.jsonl      (LABELS: item_id, unsupported, contradicts, response_type)
"""
import asyncio, json, math, os, sys, time

SYSTEM = """Вы проверяете ответ ассистента Новосибирского государственного университета (НГУ) по источникам, из которых он написан. Вам дают источники, вопрос пользователя, весь ответ и одно предложение из ответа. Оцените, подтверждают ли источники фактическое содержание этого предложения:
2 — всё, что предложение утверждает о НГУ, людях, датах, числах, адресах, правилах и событиях, прямо есть в источниках;
1 — источники подтверждают только часть предложения;
0 — источники не содержат этих сведений или им противоречат.
Если в предложении нет фактов (вежливая фраза, совет обратиться в приёмную комиссию, общее знание, сообщение о том, что сведений нет), ставьте 2.
Ответьте одной цифрой: 0, 1 или 2."""


def user_prompt(sources, question, answer, sentence):
    return (f"ИСТОЧНИКИ\n\n{sources}\n\n==========\nВОПРОС ПОЛЬЗОВАТЕЛЯ\n{question}\n\n==========\nОТВЕТ АССИСТЕНТА\n"
            f"{answer}\n\n==========\nПРОВЕРЯЕМОЕ ПРЕДЛОЖЕНИЕ\n{sentence}\n\nОценка (0, 1 или 2):")


def probs(choice):
    top = choice["logprobs"]["content"][0]["top_logprobs"]
    raw = {t["token"].strip(): t["logprob"] for t in top}
    p = {k: math.exp(raw.get(k, -100.0)) for k in ("0", "1", "2")}
    z = sum(p.values()) or 1.0
    return {k: v / z for k, v in p.items()}


async def main(rec_path, out_path, model, base_url, conc=4, limit=None):
    import httpx
    from meno_rag.stand.attribution import split_sentences
    done = {json.loads(l)["item_id"] for l in open(out_path, encoding="utf-8")} if os.path.exists(out_path) else set()
    recs = [r for r in map(json.loads, open(rec_path, encoding="utf-8"))
            if r.get("ok") and r["item_id"] not in done][: int(limit) if limit else None]
    items = {}
    for path in ("/home/m.kulakov/meno-v2/final1118/items/items_v3new.jsonl",
                 "/home/m.kulakov/meno-v2/final1118/items/setB_items.jsonl"):
        for l in open(path, encoding="utf-8"):
            if l.strip():
                i = json.loads(l); items[i["id"]] = i["question"]
    gate = asyncio.Semaphore(int(conc)); out = open(out_path, "a", encoding="utf-8"); t0 = time.time()
    async with httpx.AsyncClient(timeout=120) as http:
        async def one(r):
            sources = "\n\n".join(c.get("text") or "" for c in r.get("contexts") or [])
            answer = r["response"].strip()
            rows = []
            async with gate:
                for s in split_sentences(answer):
                    body = {"model": model, "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_prompt(sources, items.get(r["item_id"], ""), answer, s)}],
                        "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 5,
                        "guided_choice": ["0", "1", "2"], "chat_template_kwargs": {"enable_thinking": False}}
                    resp = (await http.post(f"{base_url}/chat/completions", json=body)).json()
                    p = probs(resp["choices"][0])
                    rows.append({"sentence": s, "p": {k: round(v, 4) for k, v in p.items()},
                                 "score": round(p["1"] * 0.5 + p["2"], 4),
                                 "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens")})
            out.write(json.dumps({"item_id": r["item_id"], "n_context": len(r.get("contexts") or []),
                                  "sentences": rows}, ensure_ascii=False) + "\n"); out.flush()
        await asyncio.gather(*(one(r) for r in recs))
    print(f"{len(recs)} answers in {time.time() - t0:.0f} s", flush=True)


def summary(out_path, labels_path):
    lab = {l["item_id"]: l for l in map(json.loads, open(labels_path, encoding="utf-8"))}
    rows = [r for r in map(json.loads, open(out_path, encoding="utf-8")) if r["item_id"] in lab]
    def auc(pos, neg):
        if not pos or not neg: return float("nan")
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    for name, bad in (("unsupported", lambda l: bool(l["unsupported"])),
                      ("contradicts", lambda l: bool(l["contradicts"])),
                      ("either", lambda l: bool(l["unsupported"] or l["contradicts"]))):
        # an answer's risk: 1 - the lowest sentence score (no sentences: no risk)
        risk = {r["item_id"]: 1 - min([s["score"] for s in r["sentences"]] or [1.0]) for r in rows}
        pos = [risk[i] for i in risk if bad(lab[i])]; neg = [risk[i] for i in risk if not bad(lab[i])]
        print(f"{name}: {len(pos)} bad / {len(neg)} clean answers, AUC of 1-min(score) {auc(pos, neg):.3f}")
        for th in (0.5, 0.25, 0.1):
            fp = sum(x > 1 - th for x in neg) / max(len(neg), 1); tp = sum(x > 1 - th for x in pos) / max(len(pos), 1)
            print(f"   a sentence below {th}: flags {tp:.1%} of bad, {fp:.1%} of clean answers")
    n = sum(len(r["sentences"]) for r in rows)
    print(f"{len(rows)} answers, {n} sentences, prompt tokens mean "
          f"{sum(s['prompt_tokens'] or 0 for r in rows for s in r['sentences']) / max(n, 1):.0f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(sys.argv[2], sys.argv[3])
    else:
        asyncio.run(main(*sys.argv[1:]))

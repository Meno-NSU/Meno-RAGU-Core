"""Read-only: disliked answers since a date (all models), with question, last turns, comment, sources -> dislikes_new.jsonl"""
import asyncio, os, json, sys, datetime, asyncpg
since = datetime.datetime.fromisoformat(sys.argv[1] if len(sys.argv) > 1 else "2026-11-12T00:00:00+00:00")
env = dict(l.split("=", 1) for l in open(os.path.expanduser("~/Meno-Stand/Meno-Stand-RAG-Core/.env")).read().split("\n") if "=" in l and not l.startswith("#"))
dsn = env["DATABASE_URL"].strip().replace("postgresql+asyncpg://", "postgresql://")
async def main():
    c = await asyncpg.connect(dsn); await c.execute("SET default_transaction_read_only = on")
    rows = await c.fetch("""SELECT f.value, f.comment, f.created_at AS fat, m.model, m.content, m.sources, m.conversation_id, m.created_at AS mat, m.request_id
        FROM message_feedback f JOIN messages m ON m.request_id = f.run_id AND m.role = 'assistant'
        WHERE f.value='down' AND f.created_at >= $1 ORDER BY f.created_at""", since)
    out = []
    for r in rows:
        hist = await c.fetch("SELECT role, content FROM messages WHERE conversation_id=$1 AND created_at < $2 ORDER BY created_at", r["conversation_id"], r["mat"])
        users = [h for h in hist if h["role"] == "user"]
        src = r["sources"] if isinstance(r["sources"], list) else json.loads(r["sources"] or "[]")
        out.append({"fb_at": r["fat"].isoformat(), "model": r["model"], "request_id": r["request_id"], "question": users[-1]["content"] if users else "",
                    "history": [{"role": h["role"], "content": h["content"]} for h in hist[:-1]][-6:],
                    "answer": r["content"], "comment": r["comment"], "sources": src})
    with open(os.path.expanduser("~/meno-v2/dislikes1118/dislikes_all.jsonl"), "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
    print(len(out), "dislikes since", since)
    for i, o in enumerate(out):
        q = " ".join(o["question"].split())[:90]; cm = " ".join((o["comment"] or "").split())[:110]
        print(f"{i:2d} {o['fb_at'][5:16]} {o['model'][:28]:28s} h{len(o['history'])} src{len(o['sources'])} | {q} || {cm}")
asyncio.run(main())

"""Internal labels of the QA context that an answer repeats to the user (both context styles).
usage: leaks.py RECORDS.jsonl...   (answer under 'answer' or 'response')"""
import json, re, sys
from collections import Counter

PATTERNS = {
    # numbered references to context sections: meaningless to a user
    "numbered": re.compile(r"\bDOCUMENT\b|(?:документ|фрагмент|раздел)[а-яё]*\s*(?:№\s*)?\d+", re.I),
    # the graph's internals or the section headings
    "graph": re.compile(r"граф[а-яё]*\s+знани|элемент[а-яё]*\s+граф|\(тип:|—\[|ЭЛЕМЕНТЫ ГРАФА", re.I),
    "sources_style": re.compile(r"[Сс]правк[а-яё]*\s+по\s+материал|СПРАВК|ИСТОЧНИК"),  # case-sensitive: «источник фотонов» is text
    # weaker: talks about the provided documents / context instead of the subject
    "meta": re.compile(r"предоставленн[а-яё]*\s+(?:документ|контекст|фрагмент|материал|источник)|в\s+контексте\b",
                       re.I),
}

def answer_of(r):
    for k in ("answer", "response", "generated_answer"):
        if isinstance(r.get(k), str):
            return r[k]
    return None

if __name__ == "__main__":
    for path in sys.argv[1:]:
        n = 0; hit = Counter(); examples = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line); a = answer_of(r)
            if not a:
                continue
            n += 1
            strong = False
            for name, p in PATTERNS.items():
                m = p.search(a)
                if m:
                    hit[name] += 1
                    strong |= name != "meta"
                    examples.setdefault(name, a[max(0, m.start() - 60): m.end() + 40].replace("\n", " "))
            hit["any_strong"] += strong
        print(f"{path.rsplit('/', 1)[-1]}: {n} answers | " + ", ".join(f"{k} {v} ({100*v/max(n,1):.1f}%)" for k, v in sorted(hit.items())))
        for k, v in examples.items():
            print(f"   e.g. {k}: …{v}…")

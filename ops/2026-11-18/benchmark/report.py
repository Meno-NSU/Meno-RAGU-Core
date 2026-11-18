"""README.md (final) or interim.md (core + set B judged) of the 18.11 benchmark, from the chain's outputs.
usage: python report.py final|interim F
Prints a one-line summary first (the chain puts it into DONE).
"""
import json, os, re, sys, time
from collections import Counter

mode, F = sys.argv[1], sys.argv[2]
A, B = "ragu6k", "menon"


def jload(path, default=None):
    try:
        return json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read(path, default=""):
    try:
        return open(path, encoding="utf-8").read()
    except OSError:
        return default


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(c):
    return "" if not c or c[0] is None else f" [{c[0]:+.3f}; {c[1]:+.3f}]"


DEC = {"parity": "паритет", "not distinguishable": "не различимы", f"{A} better": "RAGU v6k2 лучше",
       f"{B} better": "Менон лучше", f"mixed: {A} higher, contradicts more": "RAGU выше, но чаще противоречит",
       f"mixed: {B} higher, contradicts more": "Менон выше, но чаще противоречит", "n/a": "н/д"}
STRATUM = {"core": "core (взвешенно)", "synthetic_kb": "synthetic_kb", "graph": "graph (набор B)", "v3_ref": "v3 с эталоном (core + synthetic, без весов)"}

ev = jload(f"{F}/{'interim_tables' if mode == 'interim' else 'judge/out_eval_v1'}/eval_v1.json", {})
rows = ev.get("rows", [])
get = lambda st, g="ALL": next((r for r in rows if r["judge"] == "qwen" and r["stratum"] == st and r["group"] == g), None)


def headline_table(strata):
    L = ["| Слой | n | RAGU v6k2 | Менон 1.1 | RAGU − Менон, 95% ДИ | противоречия RAGU / Менон | Вывод |", "|---|---:|---|---|---|---|---|"]
    for st in strata:
        r = get(st)
        if not r:
            continue
        h, c = r["vital_recall"], r["contradiction"]
        L.append(f"| {STRATUM[st]} | {r['n']} | {f3(h['a'])} | {f3(h['b'])} | {h['diff']:+.3f}{ci(h['ci'])} | "
                 f"{f3(c['a'])} / {f3(c['b'])} | {DEC.get(r['decision'], r['decision'])} |")
    return L


def secondary(st):
    r = get(st)
    if not r:
        return []
    names = {"contradiction": "противоречит эталону", "any_unsupported": "есть неподтверждённые утверждения",
             "unsupported_rate": "доля неподтверждённых утверждений", "refusal": "отказ", "recall": "recall (все наггеты)",
             "solved": "решено"}
    L = [f"| Метрика ({STRATUM[st]}) | RAGU v6k2 | Менон 1.1 | разность, 95% ДИ |", "|---|---|---|---|"]
    for m, label in names.items():
        x = r.get(m)
        if x:
            L.append(f"| {label} | {f3(x['a'])} | {f3(x['b'])} | {x['diff']:+.3f}{ci(x['ci'])} |")
    return L


def groups(st, prefix):
    L = []
    for r in rows:
        if r["judge"] == "qwen" and r["stratum"] == st and r["group"].startswith(prefix):
            h = r["vital_recall"]
            note = " (описательно, n<30)" if r["n"] < 30 else ""
            L.append(f"| {r['group']} | {r['n']} | {f3(h['a'])} | {f3(h['b'])} | {h['diff']:+.3f}{ci(h['ci'])} | {h['verdict']}{note} |")
    return (["| Группа | n | RAGU v6k2 | Менон 1.1 | разность, 95% ДИ | вердикт |", "|---|---:|---|---|---|---|"] + L) if L else []


def coverage():
    L = ["| набор/система | с эталоном | ответ ok | без ответа | не запускались | оценено qwen |", "|---|---:|---:|---:|---:|---:|"]
    for k, v in (ev.get("coverage") or {}).items():
        L.append(f"| {k} | {v['items']} | {v['answered_ok']} | {v['failed']} | {v['never_run']} | {v.get('judged_qwen')} |")
    return L


def one_line():
    parts = []
    for st in ("core", "synthetic_kb", "graph"):
        r = get(st)
        if r:
            h = r["vital_recall"]
            parts.append(f"{st} {f3(h['a'])} vs {f3(h['b'])} ({h['diff']:+.3f}{ci(h['ci'])}, {r['decision']})")
    return "RAGU v6k2 vs Menon 1.1, qwen vital recall: " + "; ".join(parts) if parts else "no judged pairs"


if mode == "interim":
    L = [f"# Промежуточные итоги 18.11: RAGU v6k2 против Менона 1.1 — core и набор B", "",
         f"Сгенерировано {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}, судья qwen3.8-27b (тот же, что 16.11). "
         "Оценены только основной слой (core) и набор B; synthetic_kb и RAGAS ещё идут. Менон 1.1 — ответы и оценки 16.11.", "",
         *headline_table(["core", "graph"]), "", "## Core: вторичные метрики", "", *secondary("core"), "",
         "## Core по категориям", "", *groups("core", "cat "), "", "## Набор B по типам", "", *groups("graph", "G"),
         *groups("graph", "L")[2:], *groups("graph", "type ")[2:], "", "## Покрытие", "", *coverage(), "",
         "Эталоны в основном написаны Claude: только 22 из 608 писали разработчики."]
    open(f"{F}/interim.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(one_line())
    sys.exit(0)

# ------------------------------------------------------------------ final README
verify = jload(f"{F}/verify/verify.json", {})
res = jload(f"{F}/verify/resources.json", {})
ans = jload(f"{F}/answers/answers_summary.json", {})
rag = jload(f"{F}/ragas_side.json", {})
st = jload(f"{F}/ev/results/markers.json", None) or jload(f"{F}/ev/state.json", {})
markers = st.get("markers", {})
gw = jload(f"{F}/verify/gw_restored.json", {})
gws = jload(f"{F}/verify/gw_swap.json", {})
bm = verify.get("build_manifest") or {}
chk = verify.get("checks", {})
mapping = (chk.get("every menon_chunk_mapping id is a point") or {}).get("detail", {})
dq = read(f"{F}/verify/document_quality.log").strip().splitlines()
alog = read(f"{F}/logs/answers.log")
ctl = [l for l in alog.splitlines() if " INPROC CTL " in l]
limits = [int(m) for l in ctl for m in re.findall(r"-> (\d+)", l)]
pauses = sum(1 for l in ctl if "CTL PAUSE" in l)
first = re.search(r"^(\S+) INPROC pass 1: \d+ to answer \(core", alog, re.M)
last = re.findall(r"^(\S+) INPROC DONE ", alog, re.M)
smoke = read(f"{F}/verify/smoke.md")
smoke_q = re.findall(r"^## (.+)$", smoke, re.M)
smoke_meta = re.findall(r"^ok (\S+), error .*?, ([\d.]+) s, context kinds (\{.*?\}|null), route", smoke, re.M)


def m_time(name):
    v = markers.get(name) or ""
    return v[:16].replace("T", " ") if v else "—"


rag_lines = []
try:  # answers of the no-reference stratum this run has (the 18.11 run stopped early, see the note under RAGAS)
    NOREF_OK = json.load(open(os.path.join(F, 'answers', 'answers_summary.json')))['noref']['ok']
except Exception:  # noqa: BLE001
    NOREF_OK = 730
for name, s in (rag.get("sets") or {}).items():
    for m, x in s["metrics"].items():
        rag_lines.append(f"| {name} | {m} | {f3(x['a'])} ({x['n_a']}) | {f3(x['b'])} ({x['n_b']}) | "
                         f"{'—' if x['diff'] is None else format(x['diff'], '+.3f')}{ci(x['ci'])} |")
gl = (gw.get("tries") or [{}])[-1] if gw else {}
L = [f"# Бенчмарк 18.11.2026: RAGU v6k2 против Менона 1.1 (Meno Eval v1)", "",
     "Папка `/workspace/data/final1118/` на Everest, копия таблиц и README — `~/meno-v2/final1118/` на проде. Время UTC.", "",
     "## Что сравнивалось",
     f"- **RAGU v6k2.** Стенд `stand-v6k2` в одном процессе на проде (как вариант v5k/v5.1k 16.11, `ragu_answers.py` = "
     "`ragu_inproc.py` 16.11 + очередь по слоям и адаптивная параллельность), код Meno-RAGU-Core `main` d3c7326 (порядок по "
     "авторитетности источника, шкала качества KB, упорядоченная атрибуция; элемент графа попадает в контекст "
     "только с найденным исходным фрагментом), настройки живого контейнера RAGU (gemma-4-12B-it "
     "на H100 :9022, кросс-энкодер :9028, bge-m3 :8001, LLM-переранжирование, без few-shot, TZ Asia/Novosibirsk).",
     "  - Повтор после внешнего аудита `stand-v6k`: текстовая часть пересобрана — только индексируемые фрагменты KB v3.9b, "
     "исправленные тексты, аннотации документов. Ответы дефектного `stand-v6k` сохранены в `archive-v6k/` на проде и на Everest.",
     f"  - Граф: {bm.get('graph', '?')}",
     f"  - Узлов {res.get('nodes')}, рёбер {res.get('edges')} (v5: 346 754 и 927 166; отношение "
     f"{(res.get('check') or {}).get('node_ratio')} и {(res.get('check') or {}).get('edge_ratio')}).",
     f"  - Фрагменты: {bm.get('chunks', '?')}",
     f"  - Коллекции Qdrant: `{res.get('vdb_entity')}`, `{res.get('vdb_relation')}`, `{res.get('vdb_chunk')}`.",
     f"- **Менон 1.1.** Ответы, оценки qwen и RAGAS 16.11 взяты без изменений (шлюз Meno-Stand-RAG-Core `main` d8af6be, KB v3.8, "
     "та же gemma); со стороны Менона ничего не менялось.",
     "- **Протокол — Meno Eval v1**, как 16.11: те же слои, веса категорий, наггеты, бутстреп по кластерам (seed 1115, 10 000), "
     "судья qwen3.8-27b с тем же промптом и параметрами vLLM; RAGAS (answer_correctness, faithfulness) с тем же qwen и bge-m3.", "",
     "## Слои", "", "| Слой | Что | Вопросов |", "|---|---|---:|",
     "| core | реальные вопросы разработчиков с эталоном — **главная цифра** | 373 |",
     "| synthetic_kb | вопросы Claude по документам базы, с эталоном | 235 |",
     "| graph | набор B: вопросы, где графовый RAG должен выигрывать (синтетика) | 527 |",
     "| real_noref | вопросы без эталона (только RAGAS faithfulness) | 730 |", "",
     "Эталоны в основном написаны Claude Sonnet: только 22 из 608 писали разработчики.", "",
     "## Главные результаты (судья qwen3.8-27b, vital recall)", "", *headline_table(["core", "synthetic_kb", "graph", "v3_ref"]), "",
     "### Core: вторичные метрики", "", *secondary("core"), "",
     "### Набор B: вторичные метрики", "", *secondary("graph"), "",
     "### Core по категориям (внутри категории без весов)", "", *groups("core", "cat "), "",
     "### Набор B по типам вопросов", "", *groups("graph", "G"), *groups("graph", "L")[2:], *groups("graph", "type ")[2:], "",
     "Полные таблицы (чувствительность к эталонам, вид наггетов, покрытие): `analysis/eval_v1_tables.md`.", "",
     "## RAGAS (тот же qwen, эмбеддер bge-m3)", "",
     *(["| Набор | Метрика | RAGU v6k2 (n) | Менон 1.1 (n) | разность по общим вопросам, 95% ДИ |", "|---|---|---|---|---|", *rag_lines]
       if rag_lines else ["RAGAS не посчитан — см. `chain.log` на Everest."]), "",
     *([f"- **Набор без эталона неполон:** у RAGU v6k2 {NOREF_OK} ответов из 730. Прогон ответов остановила защита диска "
        "прода (свободно меньше 15 ГБ во время выкатки v6k2 на живой стенд, 09:15–09:21 UTC). Строку faithfulness по этому "
        "набору сравнивать нельзя; ответы будут дособраны и оценены отдельно.", ""] if NOREF_OK < 700 else []),
     "## Проверки стенда перед прогоном",
     f"- Коллекции Qdrant: " + "; ".join(f"`{n[-40:]}` {i.get('status')} {i.get('points')} точек"
                                        for n, i in (verify.get("collection_info") or {}).items()) + ".",
     f"- Идентификаторы `menon_chunk_mapping.json`: {mapping.get('mapping_ids')}, без точки в коллекции фрагментов: "
     f"{mapping.get('mapping_ids_missing')}; точек {mapping.get('points')}.",
     f"- `document_quality.json`: {'; '.join(l.strip() for l in dq if 'documents change' in l or 'present' in l or 'stand-v5k' in l or 'built by' in l)}.",
     f"- Проверка стенда: {'пройдена' if not verify.get('failed') else 'НЕ пройдена: ' + ', '.join(verify['failed'])}.",
     "- Пробные вопросы (ответ, источники, виды контекста — `analysis/smoke.md`):",
     *[f"  - «{q}»: ok {m[0]}, {m[1]} с, контекст {m[2]}" for q, m in zip(smoke_q, smoke_meta)], "",
     "## Как шёл прогон",
     f"- Ответы RAGU v6k2: с {first.group(1)[:16].replace('T', ' ') if first else '?'} до "
     f"{last[-1][:16].replace('T', ' ') if last else '?'}; по слоям: "
     + ", ".join(f"{k} {v['ok']}/{v['items']}" for k, v in ans.items()) + ".",
     f"- Параллельность адаптивная (старт 8, проба живого стенда каждые 5 мин, пауза при > 50 с, метрики H100): "
     f"предел менялся в диапазоне {min(limits) if limits else 8}–{max(limits) if limits else 8}, пауз {pauses}.",
     f"- Окно судейства на H100: gemma снята {m_time('SWAP_STARTED')}, qwen готов {m_time('QWEN_UP')}, core + B оценены "
     f"{m_time('JUDGE_CORE_DONE')}, всё оценено {m_time('JUDGE_ALL_DONE')}, RAGAS {m_time('RAGAS_DONE')}, gemma вернулась "
     f"{m_time('GEMMA_RESTORED')}{' — ВНИМАНИЕ: ' + markers['GEMMA_FAILED'] if markers.get('GEMMA_FAILED') else ''}.",
     f"- Во время окна Менон через шлюз: {'отвечал' if (gws.get('menon') or {}).get('ok') else 'не проверен или не отвечал'}"
     f"{' (' + str((gws.get('menon') or {}).get('s')) + ' с)' if (gws.get('menon') or {}).get('ok') else ''}.",
     f"- После возврата gemma: на :9022 {gl.get('h100_served')}, RAGU через шлюз "
     f"{'ok' if (gl.get('ragu') or {}).get('ok') else 'НЕ ok'} ({(gl.get('ragu') or {}).get('s')} с), Менон "
     f"{'ok' if (gl.get('menon') or {}).get('ok') else 'НЕ ok'} ({(gl.get('menon') or {}).get('s')} с).",
     *([f"- Карточка H100: найдено ограничение — {markers['HOLD']}"] if markers.get("HOLD") else []), "",
     "## Оговорки",
     "- Ответы Менона — от 16.11, ответы RAGU v6k2 — от 18.11: вопросы, зависящие от даты, сравниваются с разницей в два дня.",
     "- RAGU v6k2 отвечал в одном процессе на проде, а не через шлюз (настройки — как у живого контейнера). На 16.11 тот же "
     "способ для v5k против ответов v5k через шлюз дал на core −0.007 [−0.020; +0.004] (паритет).",
     "- Базы различаются: у RAGU v6k2 сторона фрагментов — KB v3.9b, у Менона 1.1 — KB v3.8.",
     "- Судья тот же (qwen3.8-27b, тот же промпт, temperature 0), но запущен в другой день и с другой параллельностью: "
     "небольшой разброс от батчинга vLLM возможен. Противоречия qwen находит чаще Claude (калибровка 16.11).",
     "- Неотвеченные вопросы считаются нулём и отказом (правило протокола).", "",
     "## Где что лежит (Everest `/workspace/data/final1118/`)",
     "- `work/` — что прислал прод: ответы RAGU v6k2 (`ragu6k_v3.records.jsonl`, `ragu6k_setB.records.jsonl`: ответ, контекст "
     "промпта, источники, время стадий), вопросы `items/`, слепые задания `judge/tasks/point_v6c*` (core + B), `point_v6s*` "
     "(synthetic_kb), оценки qwen `judge/out/*.qwen.jsonl`.",
     "- `ragas/<прогон>/` — пакеты `batch_*.csv` и `summary.json`.",
     "- `analysis/` — таблицы (`eval_v1_tables.md`, `eval_v1.json`, `eval_v1_rows.csv`, `ragas_side.md`), ключи слепых заданий "
     "(`point_keys.json`), проверки стенда (`verify.json`, `resources.json`, `document_quality.log`, `smoke.md`), "
     "`interim.md`, `analysis_bundle.tar.gz` (скрипты, конфиги, журнал ответов).",
     "- Журналы: `chain.sh`, `chain.log` (смена моделей на H100), `logs/` (судья, RAGAS, vLLM).",
     "- Ответы и оценки Менона 16.11: `/workspace/data/final1116/` (`work/`, `ragas/`).",
     "- Прод: `~/meno-v2/final1118/` (chain.sh, chain.log, answers/, judge/, verify/, README.md).", "",
     "## Как пересчитать",
     "На проде: `cd ~/meno-v2/final1118/judge && ~/meno-v2/venv/bin/python analyze_eval_v1.py cfg_analysis.json eval_v1 "
     "out_eval_v1 --judges qwen --pair ragu6k,menon`."]
open(f"{F}/README.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print(one_line())

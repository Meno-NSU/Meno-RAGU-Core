"""Markdown report and SVG charts of the pipelines comparison (report/pipelines/data.json, all_stats.py): README.md
with every table, and charts that GitHub shows in the browser. usage: python make_pipelines_md.py OUT_DIR"""
import json, sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((HERE / "report" / "pipelines" / "data.json").read_text(encoding="utf-8"))
RUNS = D["runs"]
METRICS = ["answer_correctness", "factual_correctness", "faithfulness", "answer_relevancy", "context_recall",
           "context_precision", "context_entity_recall"]
NAMES = {"answer_correctness": "Answer correctness", "factual_correctness": "Factual correctness",
         "faithfulness": "Faithfulness", "answer_relevancy": "Answer relevancy", "context_recall": "Context recall",
         "context_precision": "Context precision", "context_entity_recall": "Context entity recall"}
SHORT = {"answer_correctness": "AC", "factual_correctness": "FC", "faithfulness": "Faith", "answer_relevancy": "AR",
         "context_recall": "CR", "context_precision": "CP", "context_entity_recall": "CER"}
PIPES = ["Менон 1.0", "Менон 1.1", "RAGU NSUv1", "RAGU v3", "RAGU v3k"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]  # Okabe-Ito: distinguishable with colour blindness
color = lambda pipeline: COLORS[PIPES.index(pipeline)]
label = lambda r: f'{r["pipeline"]} · {r["model"]}'
by_id = {r["id"]: r for r in RUNS}
f3 = lambda v: "—" if v is None else f"{v:.3f}"
pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"
num = lambda v, d=0: "—" if v in (None, "") else (f"{v:,.{d}f}".replace(",", " "))


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def text(x, y, s, size=12, anchor="start", weight="normal", fill="#333"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
            f'fill="{fill}" font-family="Helvetica, Arial, sans-serif">{escape(s)}</text>')


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>{"".join(body)}</svg>')


def marker(model, x, y, fill):
    if model == "meno-lite":
        return f'<rect x="{x - 5}" y="{y - 5}" width="10" height="10" fill="{fill}" stroke="#fff" stroke-width="1.5"/>'
    if model == "OCC-RAG":
        return f'<path d="M{x} {y - 6} L{x + 6} {y + 5} L{x - 6} {y + 5} Z" fill="{fill}" stroke="#fff" stroke-width="1.5"/>'
    return f'<circle cx="{x}" cy="{y}" r="5.5" fill="{fill}" stroke="#fff" stroke-width="1.5"/>'


def legend(x, y):
    body = []
    for i, p in enumerate(PIPES):
        body += [f'<circle cx="{x + i * 130 + 6}" cy="{y - 4}" r="5.5" fill="{COLORS[i]}"/>', text(x + i * 130 + 16, y, p)]
    body.append(text(x + 5 * 130 + 10, y, "● gemma   ■ meno-lite   ▲ OCC-RAG", fill="#555"))
    return body


def dots_chart():
    """Small multiples: one panel per metric, the runs as rows, mean and 95% interval on a 0-1 scale."""
    left, panel_w, gap, row_h, top = 190, 230, 24, 22, 64
    cols = 4
    rows_n = len(RUNS)
    panel_h = rows_n * row_h + 40
    width = left + cols * (panel_w + gap)
    height = top + 2 * panel_h + 30
    body = [text(16, 26, "Метрики RAGAS: среднее и 95% интервал (954 вопроса, судья qwen3.8-27b)", 16, weight="bold")]
    body += legend(16, 50)
    for k, m in enumerate(METRICS):
        cx = left + (k % cols) * (panel_w + gap)
        cy = top + (k // cols) * panel_h
        body.append(text(cx, cy + 16, NAMES[m], 13, weight="bold"))
        x = lambda v: cx + v * (panel_w - 40)
        for t in (0, 0.25, 0.5, 0.75, 1):
            body.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{cy + 24}" y2="{cy + 24 + rows_n * row_h}" stroke="#e5e5e5"/>')
            body.append(text(x(t), cy + 36 + rows_n * row_h, f"{t:g}", 10, "middle", fill="#777"))
        for i, r in enumerate(RUNS):
            y = cy + 24 + i * row_h + row_h / 2
            if k % cols == 0:
                body.append(text(cx - 10, y + 4, label(r), 11, "end"))
            v = r["metrics"].get(m)
            if not v:
                body.append(text(x(0) + 4, y + 4, "нет", 10, fill="#999"))
                continue
            c = color(r["pipeline"])
            body.append(f'<line x1="{x(v["low"]):.1f}" x2="{x(v["high"]):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>')
            body.append(marker(r["model"], round(x(v["mean"]), 1), y, c))
            body.append(text(x(v["high"]) + 6, y + 4, f'{v["mean"]:.2f}', 10, fill="#444"))
        if k % cols == 0 and k // cols == 1:
            pass
    # the row labels of the second row of panels
    return svg(width, height, body)


def forest_chart():
    metrics = ["answer_correctness", "factual_correctness", "faithfulness", "answer_relevancy", "context_recall"]
    runs = [r for r in RUNS if r["id"] != D["reference"]]
    left, panel_w, gap, row_h, top = 190, 200, 22, 22, 88
    width = left + len(metrics) * (panel_w + gap)
    height = top + len(runs) * row_h + 50
    body = [text(16, 26, "Разница с Меноном 1.1 · gemma на тех же вопросах (среднее и 95% интервал)", 16, weight="bold"),
            text(16, 48, "зелёный: лучше с уверенностью, красный: хуже с уверенностью, серый: разница в пределах шума", 12, fill="#555")]
    for k, m in enumerate(metrics):
        cx = left + k * (panel_w + gap)
        vals = [D["paired"][r["id"]].get(m) for r in runs]
        ext = max(max(abs(v["low"]), abs(v["high"])) for v in vals if v) * 1.1
        x = lambda v: cx + (v + ext) / (2 * ext) * panel_w
        body.append(text(cx + panel_w / 2, top - 8, NAMES[m], 12, "middle", "bold"))
        body.append(f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{top}" y2="{top + len(runs) * row_h}" stroke="#999"/>')
        for t in (-ext / 1.1, ext / 1.1):
            body.append(text(x(t), top + len(runs) * row_h + 16, f"{t:+.2f}", 10, "middle", fill="#777"))
        for i, (r, v) in enumerate(zip(runs, vals)):
            y = top + i * row_h + row_h / 2
            if k == 0:
                body.append(text(cx - 10, y + 4, label(r), 11, "end"))
            if not v:
                body.append(text(x(0) + 4, y + 4, "нет", 10, fill="#999"))
                continue
            c = "#1a9850" if v["low"] > 0 else "#d73027" if v["high"] < 0 else "#999999"
            body.append(f'<line x1="{x(v["low"]):.1f}" x2="{x(v["high"]):.1f}" y1="{y}" y2="{y}" stroke="{c}" stroke-width="2"/>')
            body.append(f'<circle cx="{x(v["diff"]):.1f}" cy="{y}" r="4.5" fill="{c}"/>')
            right = v["diff"] < 0  # the label goes on the side away from the zero line
            body.append(text(x(v["high"]) + 5 if right else x(v["low"]) - 5, y + 4, f'{v["diff"]:+.2f}', 10,
                             'start' if right else 'end', fill='#444'))
    return svg(width, height, body)


def answers_chart():
    left, w, row_h, top = 190, 520, 22, 64
    mx = max(r["answers"]["chars_p90"] for r in RUNS)
    height = top + len(RUNS) * row_h + 40
    body = [text(16, 26, "Длина ответа в символах: медиана (точка) и 10–90-й процентили (полоса)", 16, weight="bold")]
    body += legend(16, 50)[:10]
    x = lambda v: left + v / mx * w
    for t in range(0, int(mx) + 1, 500):
        body.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + len(RUNS) * row_h}" stroke="#e5e5e5"/>')
        body.append(text(x(t), top + len(RUNS) * row_h + 16, str(t), 10, "middle", fill="#777"))
    for i, r in enumerate(RUNS):
        a, y, c = r["answers"], top + i * row_h + row_h / 2, color(r["pipeline"])
        body.append(text(left - 10, y + 4, label(r), 11, "end"))
        body.append(f'<rect x="{x(a["chars_p10"]):.1f}" y="{y - 4}" width="{x(a["chars_p90"]) - x(a["chars_p10"]):.1f}" height="8" rx="4" fill="{c}" opacity="0.35"/>')
        body.append(marker(r["model"], round(x(a["chars_median"]), 1), y, c))
        body.append(text(x(a["chars_p90"]) + 6, y + 4, str(a["chars_median"]), 10, fill="#444"))
    return svg(left + w + 80, height, body)


def stages_chart():
    stages = [("query_rewrite", "переписывание запроса"), ("retrieval", "поиск"), ("rerank", "переранжирование"),
              ("generation", "генерация ответа")]
    runs = [r for r in RUNS if D["stages"].get(r["id"])]
    left, w, row_h, top = 190, 560, 24, 64
    parts = []
    for r in runs:
        st = D["stages"][r["id"]]
        known = [st.get(k) or 0 for k, _ in stages]
        other = sum(v or 0 for k, v in st.items() if k not in dict(stages))
        parts.append([v / 1000 for v in known + [other]])
    mx = max(sum(p) for p in parts)
    x = lambda v: left + v / mx * w
    height = top + len(runs) * row_h + 40
    body = [text(16, 26, "Медианное время этапов на вопрос, секунды (во время прогона бенчмарка)", 16, weight="bold")]
    lx = 16
    for i, (_, n) in enumerate(stages + [("other", "прочее")]):
        body += [f'<rect x="{lx}" y="{40}" width="12" height="12" fill="{(COLORS + ["#bbbbbb"])[i if i < 4 else 5]}"/>',
                 text(lx + 18, 50, n)]
        lx += 40 + len(n) * 7
    for t in range(0, int(mx) + 1, 5):
        body.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{top + len(runs) * row_h}" stroke="#e5e5e5"/>')
        body.append(text(x(t), top + len(runs) * row_h + 16, f"{t} с", 10, "middle", fill="#777"))
    for i, (r, p) in enumerate(zip(runs, parts)):
        y, acc = top + i * row_h, 0
        body.append(text(left - 10, y + row_h / 2 + 4, label(r), 11, "end"))
        for k, v in enumerate(p):
            if v > 0:
                body.append(f'<rect x="{x(acc):.1f}" y="{y + 5}" width="{max(0.5, x(acc + v) - x(acc) - 1):.1f}" height="{row_h - 10}" fill="{(COLORS + ["#bbbbbb"])[k if k < 4 else 5]}"/>')
            acc += v
        body.append(text(x(acc) + 6, y + row_h / 2 + 4, f"{acc:.1f} с", 10, fill="#444"))
    return svg(left + w + 70, height, body)


for name, make in (("metrics.svg", dots_chart), ("diff_vs_menon11.svg", forest_chart), ("answer_length.svg", answers_chart),
                   ("stages.svg", stages_chart)):
    (OUT / name).write_text(make(), encoding="utf-8")

# --- README.md -------------------------------------------------------------------------------------------------
md = []
cp = {r["id"]: r["metrics"].get("context_precision", {}) for r in RUNS}
md.append("# Пайплайны НГУ на RAGAS\n")
md.append("954 проверенных вопроса бенчмарка (525 сложных, 389 фактических, 40 зависящих от даты); судья qwen3.8-27b "
          "(fp8, без рассуждений, температура 0), эмбеддинги BAAI/bge-m3, харнесс Meno-QA-Benchmark на RAGAS 0.3.1. "
          "Среднее по вопросам и 95% бутстрэп-интервал. RAGU NSUv1 — первая версия RAGU-стенда; его оценки из прогона "
          "Р. Солихова: тот же судья и те же метрики по названию, но RAGAS 0.4.3 и свой формат контекста, без answer "
          "correctness и context entity recall, поэтому сравнение NSUv1 с остальными приблизительное.\n")
md.append(f"Context precision у RAGU v3 и v3k предварительная: пересчёт на правильных блоках контекста ещё идёт "
          f"(сейчас {cp['ragu-v3-gemma'].get('n', 0)} и {cp['ragu-v3k-gemma'].get('n', 0)} вопросов из 954); первый "
          f"экспорт резал их промпт по абзацам и занижал эту метрику.\n")
md.append("![Метрики](metrics.svg)\n")
md.append("## Все метрики\n")
rows = []
for r in RUNS:
    cells = [label(r)]
    for m in METRICS:
        v = r["metrics"].get(m)
        cells.append("—" if not v else f'{v["mean"]:.3f} [{v["low"]:.3f}; {v["high"]:.3f}]' +
                     (f' (n={v["n"]})' if v["n"] < 900 else ""))
    rows.append(cells)
md.append(table(["Пайплайн"] + [NAMES[m] for m in METRICS], rows) + "\n")
md.append("Доли: оценок 0 и оценок ≥ 0,7 по каждой метрике.\n")
rows = []
for r in RUNS:
    rows.append([label(r)] + [("—" if not r["metrics"].get(m) else f'{pct(r["metrics"][m]["zero"])} / {pct(r["metrics"][m]["high_share"])}')
                              for m in METRICS])
md.append(table(["Пайплайн"] + [f"{SHORT[m]}: 0 / ≥0,7" for m in METRICS], rows) + "\n")
md.append("## Разница с Меноном 1.1 · gemma\n")
md.append("На общих вопросах: разница средних [95% интервал]; ниже — на скольких вопросах лучше / так же / хуже "
          "(порог 0,05).\n")
md.append("![Разница с Меноном 1.1](diff_vs_menon11.svg)\n")
rows = []
for r in RUNS:
    if r["id"] == D["reference"]:
        continue
    p = D["paired"][r["id"]]
    rows.append([label(r)] + ["—" if m not in p else f'{p[m]["diff"]:+.3f} [{p[m]["low"]:+.3f}; {p[m]["high"]:+.3f}]'
                              for m in METRICS])
md.append(table(["Пайплайн"] + [NAMES[m] for m in METRICS], rows) + "\n")
rows = []
for r in RUNS:
    if r["id"] == D["reference"]:
        continue
    p = D["paired"][r["id"]]
    rows.append([label(r)] + ["—" if m not in p else f'{p[m]["better"]} / {p[m]["same"]} / {p[m]["worse"]}' for m in METRICS])
md.append(table(["Пайплайн"] + [f"{SHORT[m]}: лучше/так же/хуже" for m in METRICS], rows) + "\n")
md.append("## Ответы\n")
md.append("«Нехватка в начале» — первые 220 символов говорят, что в документах нет информации; «упоминает нехватку» — "
          "где угодно в ответе; «relevancy = 0» — судья счёл ответ уклончивым.\n")
md.append("![Длина ответов](answer_length.svg)\n")
rows = [[label(r), r["answers"]["chars_median"], f'{r["answers"]["chars_p10"]}–{r["answers"]["chars_p90"]}',
         r["answers"]["words_median"], pct(r["answers"]["lack_opening"]), pct(r["answers"]["lack_anywhere"]),
         pct(r["answers"]["relevancy_zero"]), pct(r["answers"]["markdown_lists"]), r["answered"]] for r in RUNS]
md.append(table(["Пайплайн", "Символов, медиана", "10–90%", "Слов, медиана", "Нехватка в начале", "Упоминает нехватку",
                 "Relevancy = 0", "Со списками", "Ответов"], rows) + "\n")
md.append("## Контекст и промпт\n")
md.append("Блоки контекста — документы промпта и, у RAGU, элементы графа. Точные токены промпта записаны только в прогоне NSUv1.\n")
rows = []
for r in RUNS:
    c, s, cost = r["contexts"], r["sources"], r["cost"]
    rows.append([label(r), f'{c["blocks_mean"]:.1f}', "—" if c["graph_elements_mean"] is None else f'{c["graph_elements_mean"]:.1f}',
                 num(c["chars_median"]), num(c["chars_p90"]), f'{s["mean"]:.1f}',
                 num(cost.get("queries", {}).get("mean"), 1), num(cost.get("prompt_tokens", {}).get("median")),
                 num(cost.get("completion_tokens", {}).get("median"))])
md.append(table(["Пайплайн", "Блоков", "из них элементов графа", "Символов контекста, медиана", "90%",
                 "Источников в ответе", "Поисковых запросов", "Токенов промпта", "Токенов ответа"], rows) + "\n")
md.append("## Скорость\n")
md.append("Медианы во время прогона бенчмарка (зависят от нагрузки на сервер в тот момент). RAGU v3/v3k отвечали "
          "офлайн-скриптом без замеров.\n")
md.append("![Время этапов](stages.svg)\n")
rows = []
for r in RUNS:
    cost, st = r["cost"], D["stages"].get(r["id"]) or {}
    if not cost.get("latency_s"):
        continue
    rows.append([label(r), num(cost["latency_s"]["median"], 1), num(cost["latency_s"]["p90"], 1),
                 num(cost.get("ttft_s", {}).get("median"), 1)] +
                [num((st.get(k) or 0) / 1000, 1) if st.get(k) is not None else "—" for k in ("query_rewrite", "retrieval", "rerank", "generation")])
md.append(table(["Пайплайн", "Ответ, медиана, с", "Ответ, p90, с", "Первый токен, с", "Переписывание", "Поиск",
                 "Переранжирование", "Генерация"], rows) + "\n")
GN = {"complex": "сложные", "factual": "фактические", "time_sensitive": "зависят от даты", "cii_labs": "лаборатории и ЦИИ",
      "other": "прочее", "masters": "магистратура", "admission": "поступление", "student_life": "студенческая жизнь",
      "phd": "аспирантура", "scholarships": "стипендии", "international": "иностранцам"}
for key, title, sizes in (("by_category", "По типам вопросов", D["category_sizes"]), ("by_topic", "По темам вопросов", D["topic_sizes"])):
    md.append(f"## {title}\n")
    for m in ("answer_correctness", "factual_correctness"):
        groups = list(sizes)
        rows = []
        for r in RUNS:
            vals = D[key][r["id"]][m]
            rows.append([label(r)] + [f3(vals[g][0]) if vals.get(g) else "—" for g in groups])
        md.append(f"{NAMES[m]}:\n")
        md.append(table(["Пайплайн"] + [f"{GN.get(g, g)} ({sizes[g]})" for g in groups], rows) + "\n")
md.append("## Файлы\n")
md.append("`data.json` — все числа этой страницы; CSV: `metrics.csv` (метрики), `paired.csv` (разница с Меноном 1.1), "
          "`groups.csv` (по типам и темам), `hist.csv` (распределения оценок), `runs.csv` (ответы, контекст, время), "
          "`stages.csv` (этапы). Посчитано `all_stats.py` из файлов прогонов.\n")
(OUT / "README.md").write_text("\n".join(md), encoding="utf-8")
print("written", OUT)

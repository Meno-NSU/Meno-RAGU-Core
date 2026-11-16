"""Forest plot of RAGU − Menon (vital-nugget recall) with 95% CIs per question type, sets A and B, from pointwise.csv."""
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["metric"] == "vital_recall" and r["ci_low"] not in ("", "None")
        and (r["system_a"], r["system_b"]) == ("menon", "ragu")]
LABELS = {"ALL": "все вопросы", "G (graph types)": "G: все графовые", "L (single-doc controls)": "L: один документ",
          "G1": "G1 мост через сущность", "G2": "G2 полный список", "G3": "G3 сравнение", "G4": "G4 обобщение",
          "G5": "G5 связи людей и подразделений", "G6": "G6 хронология", "G7": "G7 пересечение условий",
          "L1": "L1 факт", "L2": "L2 число / таблица", "complex": "составные", "factual": "фактические",
          "multi_turn": "диалог", "unanswerable": "нет ответа в базе", "general": "общие", "time_sensitive": "зависят от даты"}
ORDER = {"A": ["ALL", "complex", "factual", "multi_turn", "unanswerable", "general"],
         "B": ["ALL", "G (graph types)", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "L (single-doc controls)", "L1", "L2"]}
by = {(r["set"], r["group"]): r for r in rows}
W, LEFT, RIGHT, ROW, TOP = 760, 300, 40, 26, 40
lo_x, hi_x = -0.4, 0.4
x = lambda v: LEFT + (float(v) - lo_x) / (hi_x - lo_x) * (W - LEFT - RIGHT)
lines, y = [], TOP
for s, title in (("A", "Набор A: реальные вопросы (200)"), ("B", "Набор B: вопросы под граф (527)")):
    lines.append(f'<text x="16" y="{y}" class="h">{title}</text>'); y += 18
    for g in ORDER[s]:
        r = by.get((s, g))
        if not r:
            continue
        d, l, h = float(r["b_minus_a"]), float(r["ci_low"]), float(r["ci_high"])
        sig = l > 0 or h < 0
        lines.append(f'<text x="16" y="{y + 4}" class="l">{LABELS.get(g, g)} (n={r["n"]})</text>')
        lines.append(f'<line x1="{x(l):.1f}" x2="{x(h):.1f}" y1="{y}" y2="{y}" class="ci"/>')
        lines.append(f'<circle cx="{x(d):.1f}" cy="{y}" r="5" class="{"pt sig" if sig else "pt"}"><title>{LABELS.get(g, g)}: {d:+.2f} [{l:+.2f}; {h:+.2f}]</title></circle>')
        lines.append(f'<text x="{W - 8}" y="{y + 4}" class="v" text-anchor="end">{d:+.2f}</text>')
        y += ROW
    y += 14
H = y + 30
axis = [f'<line x1="{x(0):.1f}" x2="{x(0):.1f}" y1="{TOP - 10}" y2="{y - 10}" class="zero"/>']
for t in (-0.4, -0.2, 0, 0.2, 0.4):
    axis.append(f'<text x="{x(t):.1f}" y="{y + 6}" class="t" text-anchor="middle">{t:+.1f}</text>')
axis.append(f'<text x="{(LEFT + W - RIGHT) / 2:.1f}" y="{y + 24}" class="t" text-anchor="middle">RAGU − Менон, доля vital-наггетов; 95% интервал (кластерный бутстреп)</text>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, sans-serif">
<style>
.h{{font-size:14px;font-weight:600;fill:#1f2328}} .l{{font-size:12px;fill:#1f2328}} .v{{font-size:12px;fill:#59636e}} .t{{font-size:11px;fill:#59636e}}
.ci{{stroke:#8c959f;stroke-width:2}} .pt{{fill:#ffffff;stroke:#2f6fb3;stroke-width:2}} .sig{{fill:#2f6fb3}} .zero{{stroke:#59636e;stroke-dasharray:3 3}}
@media (prefers-color-scheme: dark){{ .h,.l{{fill:#e6edf3}} .v,.t{{fill:#9198a1}} .pt{{fill:#0d1117;stroke:#6ca4e0}} .sig{{fill:#6ca4e0}} .zero{{stroke:#9198a1}} }}
</style>
{chr(10).join(axis + lines)}
</svg>'''
open(sys.argv[2], "w").write(svg)

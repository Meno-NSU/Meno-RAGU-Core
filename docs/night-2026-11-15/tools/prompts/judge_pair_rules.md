# Judging two answers side by side (pairwise)

A task file `/home/devuser/meno-night-1115/judge/tasks/<task>.json` holds items: id (opaque), question, history (earlier turns, may be empty), reference (a verified correct answer), answer_1, answer_2 — two answers of different assistants for Novosibirsk State University. You do not know which assistant wrote which; never guess, and do not favour a position.

For each item decide, using the reference as ground truth:
- correctness: which answer has fewer or less serious wrong statements versus the reference ("1", "2" or "tie");
- completeness: which answer covers more of what the reference says a correct answer needs ("1", "2" or "tie");
- overall: which answer is better for the user — correctness first, then completeness, then clarity and usefulness. A longer answer is NOT better by itself; correct extra details are fine, padding is not. If the answers are about equally good, say "tie".
- reason: ≤25 words in Russian.

Output: write `/home/devuser/meno-night-1115/judge/out/<task>.<judge>.jsonl`, one JSON per item in input order: {"id", "correctness", "completeness", "overall", "reason"}. Write it with a short Python script (json.dumps(..., ensure_ascii=False)); check every id is present. If the output file already exists and is complete, skip the task.

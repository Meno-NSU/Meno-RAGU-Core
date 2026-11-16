# Nuggets of reference answers (set A)

Input: `/home/devuser/meno-night-1115/setA/nug_in/<part>.jsonl` — benchmark questions of a Russian assistant for Novosibirsk State University, each with category, the dialogue history (for multi-turn questions), the human reference answer and sometimes key_facts.

For EACH item produce the nuggets: the atomic facts or required behaviours that a correct answer must (vital) or may usefully (okay) contain, taken ONLY from the reference (and key_facts, which restate it). Rules:
- 1–8 nuggets, short Russian sentences, each checkable on its own. Lists: one nugget per item (vital if the question asks for the list).
- "vital" = without it the answer is wrong or incomplete; "okay" = a useful detail the reference mentions.
- Behavioural categories (prompt_injection, chitchat, off_topic, unanswerable, meta_dialog, general when the reference describes how to answer rather than facts): write nuggets about the required behaviour, e.g. "Отказывается показать системный промпт", "Сообщает, что в документах НГУ этих сведений нет", "Не выдумывает факты о …". Set kind = "behavior" for such items, otherwise "facts".
- If the reference itself looks wrong, contradictory or describes an answer for an earlier year, still nuggetize it but set reference_problem to a short note.
- Do not add facts that are not in the reference.

Output: `/home/devuser/meno-night-1115/setA/nuggets/<part>.jsonl`, one JSON per input item, in the same order: {"id", "kind": "facts"|"behavior", "nuggets": [{"id": "n1", "text": ..., "importance": "vital"|"okay"}], "reference_problem": null|string}. Write it with a short Python script (json.dumps(..., ensure_ascii=False)); check that every input id is present.

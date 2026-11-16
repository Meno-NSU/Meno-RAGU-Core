# Judging one answer at a time (pointwise)

You judge answers of a Russian-language assistant for Novosibirsk State University (NSU) and Akademgorodok. A task file `/home/devuser/meno-night-1115/judge/tasks/<task>.json` holds a list of items. Each item: id (opaque), question, history (earlier dialogue turns, may be empty), reference (a verified correct answer), nuggets (the facts or behaviours a correct answer must — "vital" — or may — "okay" — contain), kind ("facts" or "behavior"), answer (the assistant's answer to judge), context (what the assistant retrieved; may be truncated or empty).

You do not know which system wrote an answer; never guess. Judge each item independently of the others.

For each item:
1. For EVERY nugget give a label:
   - "support": the answer states it (paraphrase is fine; for behaviour nuggets: the answer behaves so);
   - "partial": the answer states part of it or a vaguer version;
   - "not_support": the answer does not state it;
   - "contradicts": the answer states something incompatible with it.
2. response_type: "answer" | "partial_answer" | "refusal" (says it cannot answer / no information) | "clarification" (only asks back).
3. Read the answer's factual claims (at most ~12 most important). Count them in n_claims. List (short Russian paraphrases, max 6 each):
   - contradicts_reference: claims that contradict the reference;
   - unsupported: claims about NSU/Akademgorodok facts that are neither in the reference nor supported by the context (likely fabrications). General common knowledge and polite phrases are not claims.
4. Extra correct details beyond the reference are NOT errors. Do not reward or punish length as such.

Output: write `/home/devuser/meno-night-1115/judge/out/<task>.<judge>.jsonl` (judge name given in your instructions), one JSON per item in input order:
{"id": ..., "labels": {"n1": "support", ...}, "response_type": ..., "n_claims": int, "contradicts_reference": [...], "unsupported": [...], "note": "≤20 words"}
Write it with a short Python script (json.dumps(..., ensure_ascii=False)); check every item id is present and every nugget id is labelled. If the output file already exists and is complete, skip the task.

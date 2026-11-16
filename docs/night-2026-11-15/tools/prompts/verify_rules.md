# Verifying benchmark questions (set B)

For a bundle `<bundle>`: read the source documents `/home/devuser/meno-night-1115/bundles/<bundle>.md` and the candidate questions `/home/devuser/meno-night-1115/setB/raw/<bundle>.jsonl` (written from those documents; the writing rules are in /home/devuser/meno-night-1115/prompts/write_rules.md). Be strict. For EACH question check against the documents:
1. Answerable from the cited documents alone; the reference is factually correct and complete per the documents. Fix it if not (recompute nuggets / list_items).
2. Unambiguous for a user asking an assistant about all of NSU (not about this bundle); add a missing qualifier if needed.
3. Type: a G type must really need ≥2 documents, otherwise relabel L1/L2 or drop; G2/G7 lists must be complete per the bundle and scoped so that is plausible.
4. Not stale, not trivial, natural phrasing.
Write the surviving questions (possibly fixed) to `/home/devuser/meno-night-1115/setB/verified/<bundle>.jsonl` with the same fields plus "verdict" ("keep"|"fixed") and "verifier_notes". Write dropped ones (fields id, reason) to `/home/devuser/meno-night-1115/setB/verified/<bundle>.dropped.jsonl`. Use a short Python script (json.dumps(..., ensure_ascii=False)) so the JSON is valid. Create directories if needed.

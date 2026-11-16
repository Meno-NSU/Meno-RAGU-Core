# Writing benchmark questions from a bundle (set B)

You write benchmark questions for a Russian university QA assistant (Novosibirsk State University / Akademgorodok). A bundle file `/home/devuser/meno-night-1115/bundles/<bundle>.md` holds 6–24 public web documents of one site section, each under "## Dn" with its URL and Title. Read it in full.

## Question types (see section 3 of /home/devuser/meno-night-1115/research/methodology.md for examples)
- G1 bridge (2–3 hops through an entity across documents)
- G2 aggregation/listing across several documents (reference = the complete list from the bundle; scope the question so the bundle really holds the whole list)
- G3 comparison of objects described in different documents
- G4 global/thematic: "какие основные направления/возможности/…", summarising several documents
- G5 relation between people / units / projects across documents
- G6 temporal chain (order of events) across documents — only with explicit dates in the documents
- G7 set with combined conditions (intersection of constraints) across documents
- L1 single-hop exact fact from one document (control)
- L2 a number / threshold / table value from one document (control)

## Rules
- Use ONLY the bundle's documents; every fact of the reference must be stated in the cited documents. No outside knowledge.
- Phrase questions like a real applicant, student, parent or staff member would ask in Russian: short, natural, no document ids, no verbatim headings. Name the faculty/institute so the question is unambiguous for an assistant covering all of NSU.
- G types must truly need at least two documents ("doc_ids", explain in "why_multi_doc"); otherwise label L1/L2.
- Avoid facts that go stale (upcoming events, current vacancies, news). If a number belongs to a year, put the year into the question.
- Lists: at most 12 items; list every item in "list_items".
- Reference: correct, complete, concise Russian. Nuggets: 2–10 atomic facts of the reference; "vital" = any correct answer must contain it, "okay" = useful detail.
- Aim for 8–12 questions per bundle, about 2/3 G types and 1/3 L1/L2; skip types the bundle cannot support. Fewer good questions beat many weak ones.

## Output
Write `/home/devuser/meno-night-1115/setB/raw/<bundle>.jsonl` (UTF-8, one JSON object per line, Cyrillic as is) with fields: id ("<bundle>-qNN"), bundle, type, question, reference, nuggets [{id:"n1", text, importance:"vital"|"okay"}], list_items (array or null), gold_urls (URLs of the cited documents, copied exactly), doc_ids (e.g. ["D3","D7"]), why_multi_doc (empty for L types). Write the file with a short Python script (json.dumps(..., ensure_ascii=False)) so the JSON is valid.

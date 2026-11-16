# Case analysis: why one system answered better (RAGU graph RAG vs Menon 1.1 naive hybrid RAG)

You get a JSONL file of cases. Each case: the question (with dialogue history if any), the trusted reference and its nuggets
(vital/okay facts), both answers, both retrieved contexts (cut at ~6000 chars; RAGU's context mixes graph elements —
entity/relation descriptions — and text chunks), Menon's routing info (`menon_v2`: route, standalone question, subqueries),
the blind judge's labels for each system (`judge_*`: nugget labels, contradiction, unsupported claims, response type) and the
pairwise verdicts, plus `gold_coverage` (share of the reference's gold pages present in each system's context).

Systems: RAGU = knowledge graph over the NSU/Akademgorodok corpus; local search (entities → neighbours → relations → chunks)
plus naive chunk search, LLM rerank, gemma-4-12B answer. Menon 1.1 = query rewrite/routing, hybrid FRIDA + BM25 chunk search,
bge reranker, answerability/coverage check, the same gemma-4-12B answer model.

For EACH case decide, from the evidence in the file only:
1. `verdict_ok`: is the judge's verdict (which system is better) right when you read the full answers against the reference?
   If the reference itself looks wrong or outdated, say so (`reference_problem`).
2. `mechanism`: the main reason for the difference, one of:
   `graph_element` (the winning answer rests on a graph entity/relation description), `retrieval_chunk` (winner's context had a
   needed text chunk the loser's lacked), `multi_doc_aggregation` (needed facts from several documents; winner gathered them),
   `dialogue_resolution` (history/coreference handled better), `refusal` (loser refused or said "not found" although the answer
   was reachable — say whether its context had it), `generation` (both contexts had the facts; one answer used them better),
   `hallucination` (loser stated facts not in its context), `reference_problem`, `other`.
3. `evidence_winner` / `evidence_loser`: short quotes (≤300 chars each) from the contexts/answers that prove the mechanism.
4. `graph_used`: true if a graph element (not a plain chunk) in RAGU's context carried a needed fact.
5. `root_cause` (one line): what in the losing pipeline caused it (e.g. retrieval missed page X, coverage gate refused, rerank
   dropped chunk, entity merge wrong, answer prompt ignored context). Name the root cause, not a hand-written rule or word list.

Read every answer in full (do not truncate them). Be strict: "RAGU wins" is not a finding unless the evidence shows why.

Write `/home/devuser/meno-night-1115/cases/out/<input basename>.jsonl`: one JSON object per case, in input order:
{"qid", "category", "winner" ("ragu"|"menon"|"tie"), "verdict_ok", "mechanism", "graph_used", "evidence_winner",
 "evidence_loser", "root_cause", "note"}.
Use a script in your own private temp dir (mktemp -d) to write the file; check the line count equals the input count.
Reply with one short line.

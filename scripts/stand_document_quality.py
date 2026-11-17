"""document_quality.json of a stand: every document's quality on the Menon KB's scale (meno_rag.stand.quality).

usage: python scripts/stand_document_quality.py STAND_DIR KB_DOCS_JSONL_GZ [MIN_DOCUMENTS]

STAND_DIR holds chunked_texts_about_nsu_with_metadata_and_scores.jsonl; KB_DOCS_JSONL_GZ is the KB release's
kb/chunks/v1/docs.jsonl.gz (urls, quality_score). Writes STAND_DIR/document_quality.json with the documents whose
quality changes (document index -> quality) and prints how the quality of each content source moved.
"""

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from meno_rag.stand.context import normalize_urls
from meno_rag.stand.quality import QualityScale


def kb_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as fp:
        for line in fp:
            document = json.loads(line)
            for url in document.get("urls") or []:
                url = url.get("url") if isinstance(url, dict) else url
                if url and document.get("quality_score") is not None:
                    yield url, float(document["quality_score"])


def main(stand_dir, kb_docs, min_documents=30):
    scale = QualityScale.from_kb(kb_rows(kb_docs), min_documents=min_documents)
    changed, moves = {}, defaultdict(list)
    corpus = Path(stand_dir) / "chunked_texts_about_nsu_with_metadata_and_scores.jsonl"
    with corpus.open(encoding="utf-8") as fp:
        for index, line in enumerate(fp):
            document = json.loads(line)
            own = float(document.get("quality_score") or 1.0)
            urls = normalize_urls(document.get("url"))
            quality = scale.quality_of(urls[0], own) if urls else own
            if quality != own:
                changed[str(index)] = quality
            moves[document.get("content_source") or "?"].append((own, quality))
    (Path(stand_dir) / "document_quality.json").write_text(json.dumps(changed), encoding="utf-8")
    print(f"{len(changed)} documents change quality; per content source: documents, changed, median before -> after")
    for source, rows in sorted(moves.items()):
        moved = sum(1 for own, new in rows if own != new)
        print(f"  {source:10} {len(rows):7d} {moved:7d}  {statistics.median(o for o, _ in rows):.2f} -> "
              f"{statistics.median(n for _, n in rows):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))

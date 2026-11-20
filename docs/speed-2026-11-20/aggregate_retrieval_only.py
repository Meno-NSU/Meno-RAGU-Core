"""Offline two-arm view of the existing paired/ID-balanced factorial analysis."""
import argparse
import json
from pathlib import Path

from aggregate_factorial import summarize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.input.read_text())
    result = summarize(raw)
    result["effects"] = [row for row in result["effects"] if row["factor"] == "early"]
    result["stability_against_e0p0"] = [row for row in result["stability_against_e0p0"]
                                          if row["candidate_early"] == 1 and row["candidate_priority"] == 0]
    for row in result["stability_against_e0p0"]:
        row["limit"] = "First changed observable stage only; complete raw material remains server-local for factual review"
    result["design"] = {"early_rerank": False, "priority": 0, "rerank_guidance": "structured",
                        "prompt_timezone": "Asia/Novosibirsk", "source_streaming": False}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

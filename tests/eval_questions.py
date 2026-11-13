"""Run the public evaluation questions and check the expected source files are cited.

Usage:  uv run python -m tests.eval_questions [--verbose]
"""

from __future__ import annotations

import sys
import time

from pharma_agent.agent import Agent, AgentResources
from pharma_agent.config import get_settings
from pharma_agent.corpus import load_corpus
from pharma_agent.index import HybridIndex

EVAL_SET = [
    ("What are the main contraindications for warfarin?", {"dailymed-warfarin.md"}),
    (
        "Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?",
        {"dailymed-warfarin.md", "dailymed-apixaban.md"},
    ),
    (
        "A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?",
        {"dailymed-warfarin.md", "dailymed-sertraline.md"},
    ),
    (
        "A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?",
        {"dailymed-metformin.md"},
    ),
]


def main() -> int:
    verbose = "--verbose" in sys.argv
    s = get_settings()
    docs, chunks = load_corpus(s.corpus_dir)
    agent = Agent(AgentResources(docs, HybridIndex(chunks, s.cache_dir, s.embedding_model)))
    passed = 0
    for question, expected in EVAL_SET:
        t = time.time()
        res = agent.ask(question)
        cited = {src["file"] for src in res["sources"]}
        ok = expected <= cited
        passed += ok
        review = res.get("review") or {}
        print(f"\n{'PASS' if ok else 'FAIL'} ({time.time() - t:.1f}s) {question}")
        print(f"  expected={sorted(expected)} cited={sorted(cited)}")
        print(f"  tools={[c['tool'] for c in res['tool_calls']]} critic={review.get('verdict')} conf={review.get('confidence')}")
        if verbose:
            print(res["answer"])
    print(f"\n{passed}/{len(EVAL_SET)} questions cite all expected sources")
    return 0 if passed == len(EVAL_SET) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the 4 public evaluation questions and check the expected source files are cited.

Usage: uv run python -m tests.eval_questions [--verbose]
"""

import io
import sys

from pharma_agent.agent import ask

if isinstance(sys.stdout, io.TextIOWrapper):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_SET = [
    ("What are the main contraindications for warfarin?", {"dailymed-warfarin.md"}),
    ("Compare the contraindications of warfarin and apixaban. What conditions are shared and which are unique to each drug?",
     {"dailymed-warfarin.md", "dailymed-apixaban.md"}),
    ("A patient currently on warfarin is prescribed sertraline. What is the risk of bleeding and what monitoring is recommended?",
     {"dailymed-warfarin.md", "dailymed-sertraline.md"}),
    ("A patient with an eGFR of 25 mL/min is being considered for metformin. Is metformin contraindicated at this level of renal function, and what is the threshold?",
     {"dailymed-metformin.md"}),
]

passed = 0
for question, expected in EVAL_SET:
    result = ask(question)
    cited = {s["file"] for s in result["sources"]}
    passed += expected <= cited
    print(f"\n{'PASS' if expected <= cited else 'FAIL'} ({result['latency_s']}s) {question}\n  expected={sorted(expected)} cited={sorted(cited)}")
    for d in result["details"]:
        print(f"  - {d['question']} -> routes {[(r['route'], r['confidence']) for r in d['routes']]} kept={d['review'].get('kept')} issues={len(d['review'].get('issues', []))}")
    if "--verbose" in sys.argv:
        print(result["answer"])
print(f"\n{passed}/{len(EVAL_SET)} questions cite all expected sources")
sys.exit(passed != len(EVAL_SET))

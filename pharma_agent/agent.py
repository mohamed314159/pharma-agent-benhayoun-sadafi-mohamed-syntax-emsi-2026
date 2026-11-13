"""Hand-rolled agent loop: Plan -> Research (tool use) -> Draft -> Critic -> Answer.

No orchestration framework on purpose: every step is a plain function, the
control flow fits on one screen, and each step emits an event so the UI can
show the agent's reasoning live (Server-Sent Events).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator

from . import llm
from .config import get_settings
from .corpus import Document
from .index import HybridIndex
from .prompts import CRITIC_PROMPT, PLANNER_PROMPT, RESEARCH_PROMPT, WRITER_PROMPT
from .tools import EvidenceStore, Toolbox

log = logging.getLogger(__name__)

DISCLAIMER = (
    "⚠️ This answer is generated from a limited set of FDA labels and consumer drug monographs "
    "for informational purposes only. It is not medical advice — consult a qualified healthcare "
    "professional before making any clinical decision."
)
CITE_RE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")

Event = dict[str, Any]


@dataclass
class AgentResources:
    docs: list[Document]
    index: HybridIndex

    @property
    def drugs(self) -> list[str]:
        return sorted({d.drug for d in self.docs if d.source != "glossary"})


def _event(kind: str, **data: Any) -> Event:
    return {"type": kind, "ts": time.time(), **data}


class Agent:
    def __init__(self, resources: AgentResources):
        self.res = resources
        self.settings = get_settings()

    # ------------------------------------------------------------------ public
    def ask(self, question: str) -> dict[str, Any]:
        """Run the full pipeline and return the final payload."""
        final: dict[str, Any] = {}
        for ev in self.stream(question):
            if ev["type"] == "final":
                final = ev["result"]
        return final

    def stream(self, question: str) -> Iterator[Event]:
        started = time.time()
        store = EvidenceStore()
        tools = Toolbox(self.res.docs, self.res.index, store)

        # 1. PLAN ---------------------------------------------------------------
        plan = self._plan(question)
        yield _event("plan", plan=plan)

        if not plan.get("in_scope", True):
            answer = plan.get("out_of_scope_reply") or (
                "I can only answer pharmacology questions about the drugs in my corpus: "
                + ", ".join(self.res.drugs) + "."
            )
            yield _event("final", result=self._result(question, answer, plan, store, tools, None, started))
            return

        # 2. RESEARCH (agent-driven tool use) -------------------------------------
        yield from self._research(question, plan, tools)

        # 3. DRAFT ----------------------------------------------------------------
        draft = self._write(question, plan, store)
        yield _event("draft", chars=len(draft))

        # 4. CRITIC (+ optional revision rounds) ----------------------------------
        review: dict[str, Any] = {}
        for round_ in range(self.settings.max_revisions + 1):
            review = self._critique(question, draft, store)
            yield _event("review", round=round_ + 1, review=review)
            if review.get("verdict") == "pass" or round_ == self.settings.max_revisions:
                break
            for q in review.get("follow_up_queries", [])[:3]:
                tools.run("search_corpus", json.dumps({"query": q, "k": 4}))
                yield _event("tool", tool="search_corpus", args={"query": q}, reason="critic follow-up")
            draft = self._write(question, plan, store, feedback=review)
            yield _event("draft", chars=len(draft), revision=round_ + 1)

        # 5. ANSWER -----------------------------------------------------------------
        yield _event("final", result=self._result(question, draft, plan, store, tools, review, started))

    # ------------------------------------------------------------------- steps
    def _plan(self, question: str) -> dict[str, Any]:
        plan = llm.chat_json(
            PLANNER_PROMPT.format(drugs=", ".join(self.res.drugs)),
            question,
        )
        plan.setdefault("sub_questions", [question])
        plan["drugs"] = [d.lower() for d in plan.get("drugs", []) if d.lower() in self.res.drugs]
        return plan

    def _research(self, question: str, plan: dict[str, Any], tools: Toolbox) -> Iterator[Event]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": RESEARCH_PROMPT.format(drugs=", ".join(self.res.drugs))},
            {
                "role": "user",
                "content": f"Question: {question}\n\nPlan:\n{json.dumps(plan, indent=2, ensure_ascii=False)}",
            },
        ]
        schemas = tools.schemas()
        for round_ in range(self.settings.max_tool_rounds):
            msg = llm.chat(messages, tools=schemas)
            if not msg.tool_calls:
                if not tools.store.items and round_ == 0:
                    # The agent must ground its answer: nudge it once to use a tool.
                    messages.append({"role": "user", "content": "Use the tools to gather evidence first."})
                    continue
                yield _event("research_done", notes=msg.content or "")
                return
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": c.id, "type": "function",
                     "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in msg.tool_calls
                ],
            })
            for call in msg.tool_calls:
                args = call.function.arguments
                result = tools.run(call.function.name, args)
                yield _event("tool", tool=call.function.name, args=_safe_json(args), result_preview=result[:240])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        yield _event("research_done", notes="(tool budget exhausted)")

    def _write(self, question: str, plan: dict[str, Any], store: EvidenceStore, feedback: dict | None = None) -> str:
        evidence = "\n\n".join(ev.render() for ev in store.items.values()) or "(no evidence found)"
        user = (
            f"Question: {question}\n\nSub-questions:\n"
            + "\n".join(f"- {q}" for q in plan.get("sub_questions", []))
            + f"\n\nEVIDENCE:\n{evidence}"
        )
        if feedback:
            user += (
                "\n\nA reviewer flagged these issues in your previous draft — fix them "
                "(remove or correct unsupported claims, add caveats to ambiguous ones):\n"
                + json.dumps(feedback.get("issues", []), indent=2, ensure_ascii=False)
            )
        language = plan.get("language", "the same language as the question")
        msg = llm.chat([
            {"role": "system", "content": WRITER_PROMPT.format(language=language)},
            {"role": "user", "content": user},
        ])
        return (msg.content or "").strip()

    def _critique(self, question: str, draft: str, store: EvidenceStore) -> dict[str, Any]:
        cited = {i.strip() for m in CITE_RE.finditer(draft) for i in m.group(1).split(",")}
        unknown = sorted(c for c in cited if c not in store.items)
        evidence = "\n\n".join(ev.render() for ev in store.items.values())
        review = llm.chat_json(
            CRITIC_PROMPT,
            f"QUESTION:\n{question}\n\nEVIDENCE:\n{evidence}\n\nDRAFT ANSWER:\n{draft}",
        )
        review.setdefault("verdict", "pass")
        review.setdefault("issues", [])
        if unknown:  # deterministic check on top of the LLM judge
            review["verdict"] = "revise"
            review["issues"].append(
                {"claim": ", ".join(unknown), "type": "citation", "detail": "Cites evidence ids that do not exist."}
            )
        return review

    # ------------------------------------------------------------------ output
    def _result(self, question, answer, plan, store: EvidenceStore, tools: Toolbox, review, started) -> dict[str, Any]:
        answer, sources = _renumber_citations(answer, store)
        return {
            "question": question,
            "answer": answer.rstrip() + "\n\n---\n" + DISCLAIMER,
            "sources": sources,
            "plan": plan,
            "tool_calls": tools.calls,
            "review": review,
            "disclaimer": DISCLAIMER,
            "latency_s": round(time.time() - started, 2),
        }


def _safe_json(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _renumber_citations(answer: str, store: EvidenceStore) -> tuple[str, list[dict[str, Any]]]:
    """Map internal evidence ids [E7] to user-facing source numbers [1].

    Passages from the same file+section share one number, in order of first citation.
    """
    numbers: dict[tuple[str, str], int] = {}
    sources: list[dict[str, Any]] = []

    def number_for(eid: str) -> int | None:
        ev = store.items.get(eid)
        if ev is None:
            return None
        key = (ev.file, ev.section)
        if key not in numbers:
            numbers[key] = len(numbers) + 1
            sources.append({"n": numbers[key], "file": ev.file, "section": ev.section, "excerpt": ev.text[:600]})
        return numbers[key]

    def repl(m: re.Match) -> str:
        nums = sorted({n for e in m.group(1).split(",") if (n := number_for(e.strip())) is not None})
        return "".join(f"[{n}]" for n in nums)

    return CITE_RE.sub(repl, answer), sources

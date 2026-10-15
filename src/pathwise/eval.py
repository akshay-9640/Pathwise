"""Evaluation: how well Pathwise finds the right sources and cites them.

Retrieval (free, local): for each search mode, hit@k (at least one expected
source in the top k), recall@k (share of expected sources in the top k), and
MRR (how high the first expected source ranks). Context recall then scores
everything the advisor shows Claude, which adds records, rule checks, and the
rules and prerequisites they point to.

End to end (--answers, calls Claude): citation verification rate (quotes found
word for word in their source), supported statements (at least one verified
citation), cited-source recall (expected sources among the footnotes), and
refusal accuracy (answerable flag matches the question set).

    pathwise-eval                      # retrieval only, all modes
    pathwise-eval --answers            # plus end to end, with PATHWISE_BACKEND for the model
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from pydantic import BaseModel

from .advisor import Advisor
from .catalog import Catalog, load_catalog
from .retrieval import MODES, Retriever

QUESTIONS = Path(__file__).parent / "eval_questions.yaml"


class Question(BaseModel):
    id: str
    category: str
    question: str
    student: str | None = None
    expect: list[str] = []
    answerable: bool = True

    @property
    def retrieval_expect(self) -> list[str]:
        """Record and rule-check sources are built per student, not retrieved."""
        return [e for e in self.expect if e.startswith(("course:", "handbook:"))]


def load_questions(path: Path = QUESTIONS) -> list[Question]:
    return [Question(**q) for q in yaml.safe_load(path.read_text())["questions"]]


def score_ranking(ranked: list[str], expected: list[str]) -> dict:
    found = [e for e in expected if e in ranked]
    first = min((ranked.index(e) + 1 for e in found), default=None)
    return {"hit": bool(found), "recall": len(found) / len(expected), "rr": 1 / first if first else 0.0,
            "missing": [e for e in expected if e not in ranked]}


def _mean(xs) -> float:
    xs = list(xs)
    return round(sum(xs) / len(xs), 3) if xs else 0.0


def eval_retrieval(retriever: Retriever, questions: list[Question], modes=MODES, k: int = 6) -> dict:
    out = {}
    scored = [q for q in questions if q.retrieval_expect]
    for mode in modes:
        rows = []
        for q in scored:
            ranked = [h.chunk.id for h in retriever.search(q.question, k=k, mode=mode)]
            rows.append({"id": q.id, "category": q.category, "ranked": ranked, **score_ranking(ranked, q.retrieval_expect)})
        out[mode] = {"hit": _mean(r["hit"] for r in rows), "recall": _mean(r["recall"] for r in rows),
                     "mrr": _mean(r["rr"] for r in rows), "rows": rows}
    return out


def eval_context(advisor, questions: list[Question]) -> dict:
    """Recall over everything the advisor hands Claude: retrieved chunks plus the student's record,
    rule checks, the rules those checks cite, and prerequisites of named courses. Free, no model call."""
    rows = []
    for q in questions:
        if not q.expect:
            continue
        shown = [s.id for s in advisor.gather(q.question, advisor.catalog.students.get(q.student) if q.student else None)]
        rows.append({"id": q.id, "shown": len(shown), **score_ranking(shown, q.expect)})
    return {"recall": _mean(r["recall"] for r in rows), "avg_sources": _mean(r["shown"] for r in rows), "rows": rows}


def eval_answers(advisor, questions: list[Question], workers: int = 4) -> dict:
    def run(q: Question) -> dict:
        start = time.time()
        try:
            a = advisor.ask(q.question, q.student)
        except Exception as e:  # one failed call should not sink the whole run
            return {"id": q.id, "category": q.category, "error": str(e)[:300]}
        cited = [s.id for s in a.sources]
        cites = [c for s in a.statements for c in s.citations]
        return {"id": q.id, "category": q.category, "seconds": round(time.time() - start, 1),
                "answerable": a.answerable, "refusal_ok": a.answerable == q.answerable,
                "citations": len(cites), "verified": sum(c.verified for c in cites),
                "statements": len(a.statements), "supported": sum(s.supported for s in a.statements),
                "cited": cited, "source_recall": score_ranking(cited, q.expect)["recall"] if q.expect and q.answerable else None,
                "missing": [e for e in q.expect if e not in cited],
                "answer": [s.text for s in a.statements]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(run, questions))
    ok = [r for r in rows if "error" not in r]
    recall = [r["source_recall"] for r in ok if r["source_recall"] is not None]
    return {"citation_rate": round(sum(r["verified"] for r in ok) / max(1, sum(r["citations"] for r in ok)), 3),
            "supported_rate": round(sum(r["supported"] for r in ok) / max(1, sum(r["statements"] for r in ok)), 3),
            "source_recall": _mean(recall), "refusal_accuracy": _mean(r["refusal_ok"] for r in ok),
            "errors": len(rows) - len(ok), "rows": rows}


def markdown(report: dict) -> str:
    k, n = report["k"], report["questions"]
    lines = [f"# Pathwise evaluation", "", f"{n} questions, top {k} sources. Generated by `pathwise-eval`.", "",
             "## Retrieval", "", f"| Mode | Hit@{k} | Recall@{k} | MRR |", "|---|---|---|---|"]
    for mode, r in report["retrieval"].items():
        lines.append(f"| {mode} | {r['hit']:.0%} | {r['recall']:.0%} | {r['mrr']:.2f} |")
    misses = [(m, row) for m, r in report["retrieval"].items() for row in r["rows"] if row["missing"]]
    if misses:
        lines += ["", "Missed sources:", ""]
        lines += [f"- {m}, `{row['id']}`: {', '.join(row['missing'])}" for m, row in misses]
    c = report["context"]
    lines += ["", f"## Context ({report['context_mode']} retrieval)", "",
              f"Expected sources among everything the advisor shows Claude: **{c['recall']:.0%}**, "
              f"{c['avg_sources']:.1f} sources per question on average."]
    lines += [f"- `{r['id']}`: {', '.join(r['missing'])} not shown" for r in c["rows"] if r["missing"]]
    if a := report.get("answers"):
        lines += ["", f"## Answers ({report['model']})", "", "| Metric | Score |", "|---|---|",
                  f"| Citations verified word for word | {a['citation_rate']:.0%} |",
                  f"| Statements with a verified citation | {a['supported_rate']:.0%} |",
                  f"| Expected sources cited | {a['source_recall']:.0%} |",
                  f"| Refusal accuracy | {a['refusal_accuracy']:.0%} |",
                  f"| Failed calls | {a['errors']} |"]
        notes = [r for r in a["rows"] if "error" in r or not r["refusal_ok"] or r["missing"] or r["verified"] < r["citations"]]
        if notes:
            lines += ["", "Notes:", ""]
            for r in notes:
                if "error" in r:
                    lines.append(f"- `{r['id']}`: call failed, {r['error']}")
                    continue
                bits = []
                if not r["refusal_ok"]:
                    bits.append(f"answerable was {r['answerable']}")
                if r["missing"] and r["answerable"]:
                    bits.append(f"did not cite {', '.join(r['missing'])}")
                if r["verified"] < r["citations"]:
                    bits.append(f"{r['citations'] - r['verified']} of {r['citations']} quotes not found")
                if bits:
                    lines.append(f"- `{r['id']}`: {'; '.join(bits)}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pathwise-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--answers", action="store_true", help="also run the full advisor (calls Claude)")
    p.add_argument("--mode", default="hybrid", choices=MODES, help="retrieval mode the advisor uses")
    p.add_argument("--only", nargs="+", help="question ids to run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("evals"))
    args = p.parse_args(argv)

    catalog: Catalog = load_catalog()
    questions = load_questions()
    if args.only:
        questions = [q for q in questions if q.id in args.only]
    retriever = Retriever(catalog.chunks(), mentions=catalog.mentions)
    report = {"k": args.k, "questions": len(questions), "retrieval": eval_retrieval(retriever, questions, args.modes, args.k)}
    advisor = Advisor(catalog, retriever, mode=args.mode, k=args.k)
    report["context_mode"], report["context"] = args.mode, eval_context(advisor, questions)
    if args.answers:
        report["model"] = f"{advisor.llm.model}, {'claude-code CLI' if advisor.llm.cli else 'API'}, {args.mode} retrieval"
        report["answers"] = eval_answers(advisor, questions, args.workers)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2))
    (args.out / "results.md").write_text(md := markdown(report))
    print(md)


if __name__ == "__main__":
    main()

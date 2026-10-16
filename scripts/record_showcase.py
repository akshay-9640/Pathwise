"""Records the showcase: runs every suggested question through the real advisor and
saves the answers, unedited, to showcase/demo.json, along with the latest eval
results. Also copies the app's script and styles, so the showcase replays answers
in the same interface the app uses.

    PATHWISE_BACKEND=claude-code uv run python scripts/record_showcase.py
"""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from pathwise.advisor import Advisor
from pathwise.app import SUGGESTIONS, create_app
from pathwise.catalog import load_catalog
from pathwise.retrieval import Retriever
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "showcase"
WEB = ROOT / "src" / "pathwise" / "web"


def main() -> None:
    catalog = load_catalog()
    advisor = Advisor(catalog, Retriever(catalog.chunks(), mentions=catalog.mentions))
    meta = TestClient(create_app(advisor)).get("/api/meta").json()
    jobs = [(student, q) for student, qs in SUGGESTIONS.items() for q in qs]

    def run(job):
        student, q = job
        reply = TestClient(create_app(advisor)).post("/api/ask", json={"question": q, "student": student or None})
        reply.raise_for_status()
        a = reply.json()
        print(f"{student or '-':6} {a['citation_rate']:.0%} {a['seconds']:5}s  {q}")
        return f"{student}|{q}", a

    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = dict(pool.map(run, jobs))

    evals = json.loads((ROOT / "evals" / "results.json").read_text())
    summary = {"k": evals["k"], "questions": evals["questions"], "model": evals.get("model"),
               "retrieval": {m: {k: r[k] for k in ("hit", "recall", "mrr")} for m, r in evals["retrieval"].items()},
               "context": {k: evals["context"][k] for k in ("recall", "avg_sources")},
               "answers": {k: v for k, v in evals.get("answers", {}).items() if k != "rows"}}

    OUT.mkdir(exist_ok=True)
    (OUT / "demo.json").write_text(json.dumps({"recorded": date.today().isoformat(), "model": advisor.llm.model,
                                               "meta": meta, "answers": answers, "evals": summary}, indent=1))
    for f in ("app.js", "app.css"):
        shutil.copy(WEB / f, OUT / f)
    print(f"Recorded {len(answers)} answers to {OUT / 'demo.json'}")


if __name__ == "__main__":
    main()

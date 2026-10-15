# Pathwise

A RAG course advisor for a (fictional) B.Tech CSE program. Students ask questions in plain language and get answers built only from the course catalog, the student handbook, their own record, and rule checks computed in code.

**Citations are the core of the design.** Claude answers in short statements, and every statement cites a source by id with a quote copied from it. Code then checks each quote against the source text. A statement without a verified citation is marked unsupported, so nothing uncited is presented as fact.

## How it works

- **Catalog**: 31 courses, a 10-section handbook, and 4 demo students, all fictional (`src/pathwise/catalog/`).
- **Retrieval**: BM25 plus local bge-small embeddings, merged with reciprocal rank fusion. Courses a question names (by code, title, or abbreviation like NLP) are always included, along with their direct prerequisites. Runs fully local.
- **Rules**: eligibility, prerequisites, load limits, overload, probation, and timetable clashes are checked in code, and every check names the handbook rule it applies.
- **Answering**: Claude gets the sources and returns a structured draft; `verify()` checks every quote and numbers the cited sources as footnotes.

## Evaluation

`uv run pathwise-eval` scores 35 questions (lookups, paraphrased questions, handbook rules, student-specific questions, and out-of-scope questions that should be declined). Add `--answers` to run the full advisor. Latest results are in [evals/results.md](evals/results.md).

| Retrieval mode | Recall@6 | MRR |
|---|---|---|
| BM25 | 95% | 0.82 |
| Dense | 99% | 0.94 |
| Hybrid (default) | 99% | 0.92 |
| Hybrid + cross-encoder rerank | 97% | 0.90 |

End to end with Claude: 100% of quotes verified word for word, 100% of statements supported, 100% of expected sources cited, 100% of out-of-scope questions declined.

The evals changed the design: hybrid search beat cross-encoder reranking, so hybrid is the default, and failures they surfaced led to prerequisite expansion, a fix for a misleading partial-plan warning, and a stricter rule to apply the student's record. The set is small and close to saturated, so treat these numbers as a regression check rather than a benchmark.

## Run it

```bash
uv sync
uv run pytest                       # offline, no model calls
uv run pathwise-eval                # retrieval evals, free and local
export ANTHROPIC_API_KEY=...        # or PATHWISE_BACKEND=claude-code to use the Claude Code CLI
uv run pathwise-eval --answers
```

Work in progress: web app with citation footnotes and a sources panel, and an MCP server.

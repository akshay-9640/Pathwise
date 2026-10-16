"""MCP server: use Pathwise from Claude Desktop or Claude Code.

`ask` returns a full cited answer (one Claude call, with every quote verified in
code). The other tools are free and local: they search the catalog, read
sources, and run the same rule checks the advisor uses, each naming the handbook
rule it applies so the host model can cite it too.
"""

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .advisor import Advisor
from .catalog import load_catalog, normalize_code
from .llm import LLMError
from .retrieval import MODES, Retriever
from . import rules

# Expected failures raise ToolError, so the reason reaches the client instead of a bare "Error executing tool".

mcp = MCPServer(
    "pathwise",
    instructions=(
        "Course advising for a fictional B.Tech CSE program. Source ids look like course:CS 345, handbook:2.2, "
        "record:riya. Rule checks name the handbook section they apply. When you answer from these tools, cite "
        "the source ids you used. Use ask for a complete answer whose quotes are already verified."
    ),
)
catalog = load_catalog()
retriever = Retriever(catalog.chunks(), mentions=catalog.mentions)
advisor = Advisor(catalog, retriever)


def _student(student_id: str):
    s = catalog.students.get(student_id)
    if s is None:
        raise ToolError(f"No student {student_id!r}. Known: {', '.join(catalog.students)}")
    return s


def _code(code: str) -> str:
    c = normalize_code(code)
    if c not in catalog.courses:
        raise ToolError(f"No course {code!r} in the catalog.")
    return c


@mcp.tool()
def ask(question: str, student_id: str | None = None) -> dict:
    """Answer a question from the catalog and handbook (and the student's record, if given) with Claude.
    Every statement comes with numbered sources and a quote verified word for word; unsupported statements are flagged."""
    try:
        a = advisor.ask(question, _student(student_id).id if student_id else None)
    except LLMError as e:
        raise ToolError(str(e))
    return {
        "answerable": a.answerable,
        "statements": [{"text": s.text, "sources": sorted({c.number for c in s.citations if c.verified}), "supported": s.supported}
                       for s in a.statements],
        "sources": [{"number": i + 1, "id": s.id, "label": s.label,
                     "quotes": [c.quote for st in a.statements for c in st.citations if c.verified and c.source == s.id]}
                    for i, s in enumerate(a.sources)],
        "citation_rate": a.citation_rate,
    }


@mcp.tool()
def search_catalog(query: str, k: int = 6, mode: str = "hybrid") -> list[dict]:
    """Search course entries and handbook sections. mode is bm25, dense, hybrid (default), or rerank. Free and local."""
    if mode not in MODES:
        raise ToolError(f"mode must be one of {', '.join(MODES)}")
    return [{"id": h.chunk.id, "label": h.chunk.label, "rank": h.rank, "text": h.chunk.text}
            for h in retriever.search(query, k=max(1, min(k, 20)), mode=mode)]


@mcp.tool()
def get_source(source_id: str) -> dict:
    """Full text of a source by id: course:CS 345, handbook:6.2, or record:<student_id>."""
    if source_id.startswith("record:"):
        from .advisor import record_source
        return record_source(_student(source_id.split(":", 1)[1]), catalog).model_dump()
    if source_id.startswith("course:"):
        source_id = f"course:{_code(source_id.split(':', 1)[1])}"
    chunk = retriever.by_id.get(source_id)
    if chunk is None:
        raise ToolError(f"No source {source_id!r}.")
    return chunk.model_dump()


@mcp.tool()
def list_students() -> list[dict]:
    """The demo students (all fictional), with their year, CGPA, and situation."""
    out = []
    for s in catalog.students.values():
        st = rules.standing(s, catalog)
        out.append({"id": s.id, "name": s.name, "year": s.year, "semester": s.semester, "cgpa": st.cgpa,
                    "on_probation": st.on_probation, "note": s.note})
    return out


@mcp.tool()
def check_course(student_id: str, course_code: str) -> dict:
    """Can this student register for this course next term? Lists every issue with the handbook rule behind it."""
    s, code = _student(student_id), _code(course_code)
    issues = rules.check_course(s, catalog, code)
    return {"course": code, "term": rules.standing(s, catalog).next_term, "eligible": not any(i.kind == "blocked" for i in issues),
            "issues": [i.model_dump() for i in issues]}


@mcp.tool()
def check_plan(student_id: str, course_codes: list[str]) -> dict:
    """Check a full semester plan for a student: each course, total load, overload, probation, and timetable clashes."""
    return rules.check_plan(_student(student_id), catalog, [_code(c) for c in course_codes]).model_dump()


@mcp.tool()
def student_progress(student_id: str) -> dict:
    """Standing (CGPA, credits, probation, load limit) and progress toward electives and specialization tracks."""
    s = _student(student_id)
    return {"standing": rules.standing(s, catalog).model_dump(), "progress": rules.progress(s, catalog)}


@mcp.tool()
def eligible_courses(student_id: str) -> list[dict]:
    """Courses offered next term and whether this student can take each one, with reasons."""
    return rules.options(_student(student_id), catalog)


def main():
    mcp.run()


if __name__ == "__main__":
    main()

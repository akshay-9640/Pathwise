"""The advisor: answers a student's question from cited sources only.

Citations are the center of the design. Every source Claude sees has an id and a
label (a catalog entry, a handbook section, the student's record, or a rule check
computed in code). Claude must answer in statements, and every statement must
cite at least one source with a short quote copied from it. Code then checks each
quote against the source text. A statement with no verified citation is marked
unsupported, so nothing uncited is ever presented as fact.
"""

import re
from difflib import SequenceMatcher

from pydantic import BaseModel

from .catalog import Catalog, Student
from .llm import LLM
from .retrieval import Retriever
from .rules import check_course, check_plan, progress, standing


class Source(BaseModel):
    id: str            # "course:CS 345", "handbook:2.2", "record:riya", "check:CS 345", "check:plan"
    kind: str          # course, handbook, record, check
    label: str
    text: str


# What Claude returns. Kept free of defaults so it works as a structured-output schema.
class Citation(BaseModel):
    source: str
    quote: str


class Statement(BaseModel):
    text: str
    citations: list[Citation]


class Draft(BaseModel):
    answerable: bool
    statements: list[Statement]


# What the app returns after verification.
class CheckedCitation(BaseModel):
    source: str
    quote: str
    verified: bool
    number: int | None = None   # footnote number, shared by every citation of the same source
    span: list[int] | None = None   # [start, end] of the quote in the source text, for highlighting


class CheckedStatement(BaseModel):
    text: str
    citations: list[CheckedCitation]
    supported: bool


class Answer(BaseModel):
    question: str
    student: str | None
    answerable: bool
    statements: list[CheckedStatement]
    sources: list[Source]           # only the cited ones, in footnote order
    considered: list[Source]        # everything Claude was shown
    citation_rate: float            # share of citations whose quote was found in the source


SYSTEM = """You are an academic advising assistant for {program} at {institution}.

You answer a student's question using only the SOURCES you are given. Citations are mandatory.

How to answer:
- Write the answer as a list of short statements, each one a complete sentence.
- Every statement cites at least one source. Each citation names the source id exactly as given
  and quotes a short passage (about 5 to 25 words) copied word for word from that source, which
  directly supports the statement. Do not paraphrase inside a quote.
- Rule checks were computed by code from the student's record and the handbook. They are
  authoritative for eligibility, credit limits, and timetable clashes: never contradict them.
- When a student record is given and the question is about the student ("I", "me", "my"), the answer
  must apply the record: cite it and say what is true for them specifically, not only the general rule.
- Answer what was asked. Skip sources and rule checks that do not bear on the question.
- If the sources do not answer the question, set answerable to false and return one statement
  saying the catalog and handbook do not cover it, citing the closest source if any.
- Do not add advice, facts, or opinions that no source supports. Mention that the faculty advisor
  gives final approval only when the student is deciding what to register for."""


_KEEP = re.compile(r"[a-z0-9.+%']")
_SAME = {"’": "'", "‘": "'"}


def _norm_map(s: str) -> tuple[str, list[int]]:
    """Lowercases, turns punctuation into spaces, and collapses spaces, remembering for every
    character of the result where it came from in the original, so a match maps back to it."""
    out, where = [], []
    for i, ch in enumerate(s):
        c = _SAME.get(ch, ch.lower())
        c = c if _KEEP.match(c) else " "
        if c == " " and (not out or out[-1] == " "):
            continue
        out.append(c)
        where.append(i)
    while out and out[-1] == " ":
        out.pop(), where.pop()
    return "".join(out), where


def quote_span(quote: str, text: str) -> tuple[int, int] | None:
    """Where the quote appears in the text (start, end in the original text), ignoring case,
    spacing, and punctuation. Allows a near match (one or two words off) for quotes of 6 words
    or more. None when the quote is not there."""
    t, where = _norm_map(text)
    q = _norm_map(quote)[0].strip(" .'")
    if not q or not t:
        return None
    pos = t.find(q)
    if pos < 0:
        if len(q.split()) < 6:
            return None
        m = SequenceMatcher(None, t, q, autojunk=False).find_longest_match(0, len(t), 0, len(q))
        pos = max(0, m.a - m.b)
        if SequenceMatcher(None, t[pos: pos + len(q) + 10], q, autojunk=False).ratio() < 0.9:
            return None
    end = min(len(t), pos + len(q)) - 1
    return where[pos], where[end] + 1


def quote_found(quote: str, text: str) -> bool:
    return quote_span(quote, text) is not None


def record_source(student: Student, catalog: Catalog) -> Source:
    st = standing(student, catalog)
    passed = ", ".join(f"{c} {catalog.courses[c].title} ({student.completed[c]})" for c in st.passed if c in catalog.courses)
    failed = ", ".join(f"{c} {catalog.courses[c].title}" for c in st.failed) or "none"
    now = ", ".join(f"{c} {catalog.courses[c].title}" for c in st.in_progress) or "none"
    p = progress(student, catalog)
    tracks = "; ".join(f"{t['track']} needs {t['needed']} electives, passed {', '.join(t['passed']) or 'none'}, in progress {', '.join(t['in_progress']) or 'none'}" for t in p["tracks"])
    text = (f"{student.name} is in Year {student.year}, Semester {student.semester}. Next term is Year {st.next_term[0]}, Semester {st.next_term[1]}. "
            f"CGPA {st.cgpa}. Earned credits {st.earned_credits}. On probation: {'yes' if st.on_probation else 'no'}. "
            f"Maximum load next term: {st.max_load} credits{'; may request an overload up to 24' if st.can_overload else ''}. "
            f"Passed: {passed}. Failed and not yet repeated: {failed}. In progress: {now}. "
            f"Electives passed: {p['electives_passed']} of {p['electives_needed']} needed. Specialization progress: {tracks}.")
    return Source(id=f"record:{student.id}", kind="record", label=f"Student record: {student.name}", text=text)


def check_sources(question: str, student: Student, catalog: Catalog, course_codes: list[str]) -> list[Source]:
    """Rule checks for the courses in play: each one is computed in code and names the handbook rule it applies."""
    st = standing(student, catalog)
    term = f"Year {st.next_term[0]}, Semester {st.next_term[1]}"
    out = []
    for code in course_codes:
        c = catalog.course(code)
        if c is None:
            continue
        issues = check_course(student, catalog, code)
        if not issues:
            verdict = f"{student.name} meets every requirement for {c.code} {c.title} in {term}."
        else:
            verdict = f"For {student.name} in {term}, {c.code} {c.title}: " + " ".join(f"{i.message} [{i.rule}]" for i in issues)
        out.append(Source(id=f"check:{c.code}", kind="check", label=f"Rule check: {c.code} for {student.name}", text=verdict))
    named = catalog.mentions(question)
    if len(named) >= 2:
        plan = check_plan(student, catalog, named)
        # The named courses are only part of a semester, so being under the minimum load says nothing.
        # Being over a limit still holds, since more courses only add credits.
        plan_issues = [i for i in plan.issues if i.code is None and i.rule != "handbook:2.3"]
        text = (f"Taking {', '.join(named)} together in {term} is {plan.credits} credits. "
                + (" ".join(f"{i.message} [{i.rule}]" for i in plan_issues) if plan_issues else "No load or timetable clash problems."))
        out.append(Source(id="check:plan", kind="check", label=f"Rule check: {' + '.join(named)} together", text=text))
    return out


class Advisor:
    def __init__(self, catalog: Catalog, retriever: Retriever, llm: LLM | None = None, mode: str = "hybrid", k: int = 6):
        self.catalog, self.retriever, self.mode, self.k = catalog, retriever, mode, k
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    def gather(self, question: str, student: Student | None) -> list[Source]:
        hits = self.retriever.search(question, k=self.k, mode=self.mode)
        sources = [Source(id=h.chunk.id, kind=h.chunk.kind, label=h.chunk.label, text=h.chunk.text) for h in hits]
        # The direct prerequisites of a course the question names, so "what do I need before Deep
        # Learning" can cite Machine Learning's own entry, not only the line that names it.
        have = {s.id for s in sources}
        for code in self.catalog.mentions(question):
            for p in self.catalog.courses[code].prerequisites:
                chunk = self.retriever.by_id.get(f"course:{p}")
                if chunk and chunk.id not in have:
                    have.add(chunk.id)
                    sources.append(Source(id=chunk.id, kind=chunk.kind, label=chunk.label, text=chunk.text))
        if student:
            codes = list(dict.fromkeys(self.catalog.mentions(question) + [s.id.split(":", 1)[1] for s in sources if s.kind == "course"]))
            checks = check_sources(question, student, self.catalog, codes)
            # Bring in every handbook rule a check refers to, so Claude can cite the rule itself.
            rules = {r for c in checks for r in re.findall(r"\[(handbook:[\d.]+)\]", c.text)}
            have = {s.id for s in sources}
            for chunk in self.retriever.chunks:
                if chunk.id in rules and chunk.id not in have:
                    sources.append(Source(id=chunk.id, kind="handbook", label=chunk.label, text=chunk.text))
            sources = [record_source(student, self.catalog)] + checks + sources
        return sources

    def ask(self, question: str, student_id: str | None = None) -> Answer:
        student = self.catalog.students.get(student_id) if student_id else None
        sources = self.gather(question, student)
        listing = "\n\n".join(f'<source id="{s.id}" label="{s.label}">\n{s.text}\n</source>' for s in sources)
        prompt = (f"<sources>\n{listing}\n</sources>\n\n"
                  + (f"The student asking is {student.name}.\n" if student else "The student has not shared their record.\n")
                  + f"<question>\n{question}\n</question>")
        draft = self.llm.parse(system=SYSTEM.format(program=self.catalog.program, institution=self.catalog.institution),
                               prompt=prompt, output_format=Draft, effort="medium")
        return verify(question, student, draft, sources)


def verify(question: str, student: Student | None, draft: Draft, sources: list[Source]) -> Answer:
    by_id = {s.id: s for s in sources}
    numbers: dict[str, int] = {}
    statements, total, good = [], 0, 0
    for st in draft.statements:
        checked = []
        for c in st.citations:
            src = by_id.get(c.source)
            span = quote_span(c.quote, src.text) if src else None
            ok = span is not None
            total += 1
            good += ok
            if ok:
                numbers.setdefault(c.source, len(numbers) + 1)
            checked.append(CheckedCitation(source=c.source, quote=c.quote, verified=ok, number=numbers.get(c.source) if ok else None, span=list(span) if span else None))
        statements.append(CheckedStatement(text=st.text, citations=checked, supported=any(c.verified for c in checked)))
    cited = sorted(numbers, key=numbers.get)
    return Answer(question=question, student=student.id if student else None, answerable=draft.answerable,
                  statements=statements, sources=[by_id[i] for i in cited], considered=sources,
                  citation_rate=round(good / total, 3) if total else 0.0)

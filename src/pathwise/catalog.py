"""Loads the course catalog, handbook, and demo students, and splits them into
the chunks that retrieval searches over. One chunk per course and one per
handbook section, each with a stable id that answers cite."""

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

DATA = Path(__file__).parent / "catalog"
GRADE_POINTS = {"O": 10, "A+": 9, "A": 8, "B+": 7, "B": 6, "C": 5, "P": 4, "F": 0}


class Course(BaseModel):
    code: str
    title: str
    credits: int
    year: int
    semester: int
    kind: str
    slot: str
    prerequisites: list[str]
    tracks: list[str] = []
    aliases: list[str] = []
    topics: list[str]
    description: str
    min_credits: int = 0


class Section(BaseModel):
    number: str
    title: str
    text: str


class Student(BaseModel):
    id: str
    name: str
    year: int
    semester: int
    note: str = ""
    completed: dict[str, str]
    in_progress: list[str]


class Chunk(BaseModel):
    id: str          # "course:CS 345" or "handbook:2.2"
    kind: str        # "course" or "handbook"
    label: str       # what a citation shows: "CS 345 Deep Learning" or "Handbook 2.2 Overload"
    text: str


class Catalog(BaseModel):
    program: str
    institution: str
    courses: dict[str, Course]
    sections: list[Section]
    students: dict[str, Student]

    def course(self, code: str) -> Course | None:
        return self.courses.get(normalize_code(code))

    def chunks(self) -> list[Chunk]:
        out = []
        for c in self.courses.values():
            prereqs = ", ".join(f"{p} {self.courses[p].title}" for p in c.prerequisites) or "none"
            tracks = f" (tracks: {', '.join(c.tracks)})" if c.tracks else ""
            extra = f" Also needs at least {c.min_credits} earned credits." if c.min_credits else ""
            known = f" Also called {', '.join(c.aliases)}." if c.aliases else ""
            text = (f"{c.code} {c.title}.{known} {c.credits} credits, {c.kind}{tracks}. Offered in Year {c.year}, Semester {c.semester}, "
                    f"timetable slot {c.slot}. Prerequisites: {prereqs}.{extra} Topics: {', '.join(c.topics)}. {c.description}")
            out.append(Chunk(id=f"course:{c.code}", kind="course", label=f"{c.code} {c.title}", text=text))
        for s in self.sections:
            out.append(Chunk(id=f"handbook:{s.number}", kind="handbook", label=f"Handbook {s.number} {s.title}", text=f"{s.number} {s.title}. {s.text}"))
        return out

    def chunk_ids(self) -> set[str]:
        return {c.id for c in self.chunks()}

    def mentions(self, text: str) -> list[str]:
        """Courses a question names, in order: by code ("cs345"), full title ("Deep Learning"),
        or abbreviation ("NLP"). Longer names win, so "Machine Learning Operations" is not
        also read as "Machine Learning"."""
        found: list[tuple[int, str]] = []
        taken = [False] * len(text)

        def claim(m: re.Match, code: str):
            if not any(taken[m.start():m.end()]):
                taken[m.start():m.end()] = [True] * (m.end() - m.start())
                found.append((m.start(), code))

        for m in re.finditer(r"\b([A-Za-z]{2})[\s-]?(\d{3})\b", text):
            code = f"{m.group(1).upper()} {m.group(2)}"
            if code in self.courses:
                claim(m, code)
        names = [(c.title, c.code, False) for c in self.courses.values()] + [(a, c.code, a.isupper() and len(a) <= 2) for c in self.courses.values() for a in c.aliases]
        for name, code, exact_case in sorted(names, key=lambda n: -len(n[0])):
            for m in re.finditer(rf"\b{re.escape(name)}\b", text, 0 if exact_case else re.I):
                claim(m, code)
        return list(dict.fromkeys(code for _, code in sorted(found)))


def normalize_code(code: str) -> str:
    """'cs345', 'CS-345', 'cs 345' all become 'CS 345'."""
    m = re.fullmatch(r"\s*([A-Za-z]{2})[\s-]*(\d{3})\s*", code)
    return f"{m.group(1).upper()} {m.group(2)}" if m else code.strip()


def parse_handbook(markdown: str) -> list[Section]:
    """Leaf sections: every '###' subsection, plus '##' sections that have no subsections."""
    sections, current = [], None
    for line in markdown.splitlines():
        h = re.match(r"^(#{2,3}) (\d+(?:\.\d+)?)\. (.+)$", line) or re.match(r"^(#{2,3}) (\d+\.\d+) (.+)$", line)
        if h:
            current = Section(number=h.group(2), title=h.group(3).strip(), text="")
            sections.append(current)
        elif current and line.strip():
            current.text = (current.text + " " + line.strip()).strip()
    # A '##' heading followed straight by '###' headings has no text of its own.
    return [s for s in sections if s.text]


@lru_cache
def load_catalog(data_dir: str = str(DATA)) -> Catalog:
    d = Path(data_dir)
    raw = yaml.safe_load((d / "courses.yaml").read_text())
    courses = {c["code"]: Course(**c) for c in raw["courses"]}
    students = {s["id"]: Student(**s) for s in yaml.safe_load((d / "students.yaml").read_text())["students"]}
    return Catalog(program=raw["program"], institution=raw["institution"], courses=courses,
                   sections=parse_handbook((d / "handbook.md").read_text()), students=students)

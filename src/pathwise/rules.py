"""Academic rules, enforced in code. The language model explains these results;
it never decides eligibility. Every rule here points at the handbook section it
comes from, so answers can cite it."""

from pydantic import BaseModel

from .catalog import GRADE_POINTS, Catalog, Course, Student, normalize_code

NORMAL_MAX, OVERLOAD_MAX, PROBATION_MAX, MIN_LOAD = 20, 24, 16, 9
OVERLOAD_CGPA, PROBATION_CGPA = 8.0, 5.0
ELECTIVES_NEEDED, ELECTIVE_CREDITS_NEEDED, GRAD_CREDITS, SPECIALIZATION_COUNT = 4, 13, 86, 3


class Issue(BaseModel):
    code: str | None          # course the issue is about, or None for the whole plan
    kind: str                 # "blocked" stops registration; "conditional" depends on passing courses in progress; "needs_approval"; "info"
    message: str
    rule: str                 # handbook section, e.g. "handbook:4.1"


class Standing(BaseModel):
    cgpa: float
    earned_credits: int
    passed: list[str]
    failed: list[str]          # failed and not yet passed on a later attempt
    in_progress: list[str]
    on_probation: bool
    max_load: int
    can_overload: bool
    next_term: tuple[int, int]


def next_term(student: Student) -> tuple[int, int]:
    return (student.year, 2) if student.semester == 1 else (student.year + 1, 1)


def standing(student: Student, catalog: Catalog) -> Standing:
    passed = [c for c, g in student.completed.items() if g != "F"]
    failed = [c for c, g in student.completed.items() if g == "F" and c not in passed]
    graded = [(catalog.courses[c].credits, GRADE_POINTS[g]) for c, g in student.completed.items() if c in catalog.courses]
    credits = sum(cr for cr, _ in graded)
    cgpa = round(sum(cr * gp for cr, gp in graded) / credits, 2) if credits else 0.0
    probation = bool(graded) and cgpa < PROBATION_CGPA
    can_overload = cgpa >= OVERLOAD_CGPA and not probation
    max_load = PROBATION_MAX if probation else NORMAL_MAX
    return Standing(cgpa=cgpa, earned_credits=sum(catalog.courses[c].credits for c in passed if c in catalog.courses),
                    passed=passed, failed=failed, in_progress=list(student.in_progress), on_probation=probation,
                    max_load=max_load, can_overload=can_overload, next_term=next_term(student))


def check_course(student: Student, catalog: Catalog, code: str, term: tuple[int, int] | None = None) -> list[Issue]:
    """Can this student register for this course in the given term (default: their next term)?"""
    code = normalize_code(code)
    course = catalog.course(code)
    if course is None:
        return [Issue(code=code, kind="blocked", message=f"{code} is not in the catalog.", rule="handbook:1")]
    st = standing(student, catalog)
    year, semester = term or st.next_term
    issues = []
    if code in st.passed:
        issues.append(Issue(code=code, kind="blocked", message=f"{code} is already passed and cannot be repeated to improve the grade.", rule="handbook:3.3"))
    if code in st.in_progress:
        issues.append(Issue(code=code, kind="blocked", message=f"{code} is in progress this semester.", rule="handbook:4.1"))
    if course.semester != semester:
        issues.append(Issue(code=code, kind="blocked", message=f"{code} is only offered in Semester {course.semester}.", rule="handbook:1"))
    for p in course.prerequisites:
        if p in st.passed:
            continue
        if p in st.in_progress:
            issues.append(Issue(code=code, kind="conditional", message=f"Needs {p} {catalog.courses[p].title}, which is in progress; it counts only once passed.", rule="handbook:4.1"))
        else:
            reason = "failed and not yet repeated" if p in st.failed else "not taken yet"
            issues.append(Issue(code=code, kind="blocked", message=f"Needs {p} {catalog.courses[p].title} ({reason}).", rule="handbook:4.1"))
    if course.min_credits:
        pending = sum(catalog.courses[c].credits for c in st.in_progress if c in catalog.courses)
        if st.earned_credits < course.min_credits:
            kind = "conditional" if st.earned_credits + pending >= course.min_credits else "blocked"
            issues.append(Issue(code=code, kind=kind, message=f"Needs {course.min_credits} earned credits; you have {st.earned_credits}"
                                + (f" ({st.earned_credits + pending} if this semester's courses are passed)." if kind == "conditional" else "."), rule="handbook:4.1"))
    return issues


class PlanResult(BaseModel):
    term: tuple[int, int]
    credits: int
    ok: bool                   # no blocking issues
    issues: list[Issue]


def check_plan(student: Student, catalog: Catalog, codes: list[str], term: tuple[int, int] | None = None) -> PlanResult:
    """Checks a whole semester plan: each course, plus load and timetable clashes."""
    st = standing(student, catalog)
    term = term or st.next_term
    codes = list(dict.fromkeys(normalize_code(c) for c in codes))
    issues = [i for c in codes for i in check_course(student, catalog, c, term)]
    known = [catalog.courses[c] for c in codes if c in catalog.courses]
    credits = sum(c.credits for c in known)

    by_slot: dict[str, list[Course]] = {}
    for c in known:
        if c.semester == term[1]:
            by_slot.setdefault(c.slot, []).append(c)
    for slot, cs in by_slot.items():
        if len(cs) > 1:
            names = " and ".join(f"{c.code} {c.title}" for c in cs)
            issues.append(Issue(code=None, kind="blocked", message=f"{names} both meet in slot {slot}; choose one.", rule="handbook:6.2"))

    if st.on_probation and credits > PROBATION_MAX:
        issues.append(Issue(code=None, kind="blocked", message=f"{credits} credits is over the {PROBATION_MAX}-credit limit while on probation (CGPA {st.cgpa}).", rule="handbook:9"))
    elif credits > OVERLOAD_MAX:
        issues.append(Issue(code=None, kind="blocked", message=f"{credits} credits is over the {OVERLOAD_MAX}-credit maximum, even with an overload.", rule="handbook:2.2"))
    elif credits > NORMAL_MAX:
        if st.can_overload:
            issues.append(Issue(code=None, kind="needs_approval", message=f"{credits} credits is an overload; allowed with CGPA {st.cgpa} but needs your faculty advisor's written approval.", rule="handbook:2.2"))
        else:
            issues.append(Issue(code=None, kind="blocked", message=f"{credits} credits is over {NORMAL_MAX}, and an overload needs a CGPA of {OVERLOAD_CGPA} (yours is {st.cgpa}).", rule="handbook:2.2"))
    if 0 < credits < MIN_LOAD:
        issues.append(Issue(code=None, kind="needs_approval", message=f"{credits} credits is under the {MIN_LOAD}-credit minimum and needs head of department approval.", rule="handbook:2.3"))
    if st.on_probation:
        issues.append(Issue(code=None, kind="needs_approval", message="On probation: your faculty advisor must sign off on this registration.", rule="handbook:9"))
    return PlanResult(term=term, credits=credits, ok=not any(i.kind == "blocked" for i in issues), issues=issues)


def options(student: Student, catalog: Catalog, term: tuple[int, int] | None = None) -> list[dict]:
    """Every course offered in the term, with whether this student can take it.
    Includes failed courses that must be repeated, and lower-year courses not taken yet."""
    st = standing(student, catalog)
    term = term or st.next_term
    out = []
    for c in catalog.courses.values():
        if c.semester != term[1] or c.code in st.passed or c.code in st.in_progress or c.year > term[0]:
            continue
        issues = check_course(student, catalog, c.code, term)
        status = "blocked" if any(i.kind == "blocked" for i in issues) else "conditional" if issues else "eligible"
        out.append({"code": c.code, "title": c.title, "credits": c.credits, "kind": c.kind, "tracks": c.tracks, "slot": c.slot,
                    "status": status, "repeat": c.code in st.failed, "issues": [i.model_dump() for i in issues]})
    return sorted(out, key=lambda o: (not o["repeat"], o["kind"] != "core", o["code"]))


def progress(student: Student, catalog: Catalog) -> dict:
    """Graduation and specialization progress (handbook 5 and 8)."""
    st = standing(student, catalog)
    electives = [catalog.courses[c] for c in st.passed if c in catalog.courses and catalog.courses[c].kind == "elective"]
    core = [c for c in catalog.courses.values() if c.kind == "core"]
    tracks: dict[str, dict] = {}
    for course in catalog.courses.values():
        for t in course.tracks:
            row = tracks.setdefault(t, {"track": t, "passed": [], "in_progress": [], "needed": SPECIALIZATION_COUNT})
            if course.code in st.passed:
                row["passed"].append(course.code)
            elif course.code in st.in_progress:
                row["in_progress"].append(course.code)
    return {
        "earned_credits": st.earned_credits, "credits_needed": GRAD_CREDITS,
        "core_passed": sum(c.code in st.passed for c in core), "core_total": len(core),
        "electives_passed": len(electives), "electives_needed": ELECTIVES_NEEDED,
        "elective_credits": sum(c.credits for c in electives), "elective_credits_needed": ELECTIVE_CREDITS_NEEDED,
        "cgpa": st.cgpa, "tracks": sorted(tracks.values(), key=lambda t: -len(t["passed"]) - 0.5 * len(t["in_progress"])),
    }

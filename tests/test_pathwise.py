"""Offline tests: no model downloads and no Claude calls (fake embedder and fake LLM)."""

import numpy as np
import pytest

from pathwise.advisor import Advisor, Citation, Draft, Statement, quote_found, verify
from pathwise.catalog import load_catalog, normalize_code, parse_handbook
from pathwise.retrieval import Retriever, mentioned_codes, tokenize
from pathwise.rules import check_course, check_plan, options, progress, standing

CAT = load_catalog()


class FakeEmbedder:
    """Bag-of-words vectors: enough to exercise the dense and hybrid code paths."""
    model_name = "fake"

    def __init__(self):
        self.vocab = {}

    def _vec(self, text):
        v = np.zeros(512)
        for t in tokenize(text):
            v[self.vocab.setdefault(t, len(self.vocab) % 512)] += 1
        return v

    def passages(self, texts):
        return np.array([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


class FakeReranker:
    def scores(self, query, texts):
        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) for t in texts]


class FakeLLM:
    def __init__(self, draft):
        self.draft, self.calls = draft, []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.draft


def retriever():
    return Retriever(CAT.chunks(), embedder=FakeEmbedder(), reranker=FakeReranker(), cache=None, mentions=CAT.mentions)


# Catalog
def test_catalog_is_consistent():
    codes = set(CAT.courses)
    assert len(codes) == 31
    assert all(p in codes for c in CAT.courses.values() for p in c.prerequisites)
    assert sum(c.credits for c in CAT.courses.values() if c.kind == "core") == 71
    ids = CAT.chunk_ids()
    assert "course:CS 345" in ids and "handbook:2.2" in ids and "handbook:2" not in ids  # '## 2' has no text of its own


def test_demo_students_only_took_courses_they_were_allowed_to():
    for s in CAT.students.values():
        for code in s.completed:
            assert all(p in s.completed for p in CAT.courses[code].prerequisites), (s.id, code)


def test_normalize_code_and_mentions():
    assert normalize_code("cs345") == normalize_code("CS-345") == "CS 345"
    assert mentioned_codes("Can I take cs345 and CS 380?") == ["CS 345", "CS 380"]


def test_handbook_parser_keeps_leaf_sections():
    secs = parse_handbook("## 1. Intro\nHello.\n## 2. Load\n### 2.1 Normal\nTwenty.\n")
    assert [(s.number, s.text) for s in secs] == [("1", "Hello."), ("2.1", "Twenty.")]


# Rules
def test_standing_for_each_demo_student():
    riya, dev = standing(CAT.students["riya"], CAT), standing(CAT.students["dev"], CAT)
    assert riya.can_overload and not riya.on_probation and riya.next_term == (3, 2)
    assert dev.on_probation and dev.max_load == 16 and set(dev.failed) == {"CS 250", "MA 201"}


def test_prerequisites_in_progress_are_conditional_and_missing_ones_block():
    riya, kabir = CAT.students["riya"], CAT.students["kabir"]
    assert [i.kind for i in check_course(riya, CAT, "CS 345")] == ["conditional"]  # CS 340 in progress
    assert any(i.kind == "blocked" and "MA 201" in i.message for i in check_course(kabir, CAT, "CS 340", term=(3, 1)))
    assert check_course(riya, CAT, "CS 101")[0].rule == "handbook:3.3"  # already passed
    assert check_course(riya, CAT, "CS 999")[0].kind == "blocked"


def test_plan_catches_clashes_overloads_and_probation():
    riya, dev = CAT.students["riya"], CAT.students["dev"]
    clash = check_plan(riya, CAT, ["CS 345", "CS 380"])
    assert not clash.ok and any(i.rule == "handbook:6.2" for i in clash.issues)
    meera = CAT.students["meera"]  # CGPA 8.06, so a 22-credit final semester is an overload she may request
    big = check_plan(meera, CAT, ["CS 492", "CS 380", "CS 450", "CS 460", "CS 499"])
    assert big.credits == 22 and any(i.kind == "needs_approval" and i.rule == "handbook:2.2" for i in big.issues)
    heavy = check_plan(dev, CAT, ["CS 250", "MA 201", "CS 375", "CS 385", "CS 345"])
    assert any(i.rule == "handbook:9" and i.kind == "blocked" for i in heavy.issues)


def test_options_put_repeats_first_and_progress_counts_tracks():
    dev_opts = options(CAT.students["dev"], CAT)
    assert [o["code"] for o in dev_opts[:2]] == ["CS 250", "MA 201"] and all(o["repeat"] for o in dev_opts[:2])
    tracks = {t["track"]: t for t in progress(CAT.students["meera"], CAT)["tracks"]}
    assert tracks["AI and Machine Learning"]["passed"] == ["CS 345"]
    assert set(tracks["AI and Machine Learning"]["in_progress"]) == {"CS 420", "CS 430"}


# Retrieval
def test_retrieval_modes_and_code_mentions():
    r = retriever()
    assert r.search("overload credits CGPA advisor approval", mode="bm25")[0].chunk.id == "handbook:2.2"
    for mode in ("dense", "hybrid", "rerank"):
        assert r.search("search engines ranking BM25 vector search", k=3, mode=mode)[0].chunk.id == "course:CS 420", mode
    assert r.search("cs 380?", k=3, mode="bm25")[0].chunk.id == "course:CS 380"
    with pytest.raises(ValueError):
        r.search("x", mode="nope")


# Citations
def test_quote_matching_is_strict_but_forgives_formatting():
    text = "Students with a CGPA of 8.0 or higher may request an overload of up to 24 credits in one semester."
    assert quote_found("CGPA of 8.0 or higher may request an overload", text)
    assert quote_found("cgpa of 8.0 or higher  may request an  overload.", text)
    assert quote_found("Students with a CGPA of 8.0 or higher may request overload of up to 24 credits", text)  # one word dropped
    assert not quote_found("CGPA of 7.0 or higher may request", text)
    assert not quote_found("overload", "Underload needs approval")
    assert not quote_found("", text)


def test_verify_numbers_sources_and_flags_unsupported_statements():
    sources = Advisor(CAT, retriever()).gather("Can I overload?", CAT.students["riya"])
    draft = Draft(answerable=True, statements=[
        Statement(text="You can request up to 24 credits.", citations=[Citation(source="handbook:2.2", quote="may request an overload of up to 24 credits")]),
        Statement(text="Your CGPA is 8.71.", citations=[Citation(source="record:riya", quote="CGPA 8.71")]),
        Statement(text="Overloads are free on Fridays.", citations=[Citation(source="handbook:2.2", quote="overloads are free on Fridays")]),
        Statement(text="Made up source.", citations=[Citation(source="handbook:99", quote="anything")]),
    ])
    ans = verify("Can I overload?", CAT.students["riya"], draft, sources)
    assert [s.supported for s in ans.statements] == [True, True, False, False]
    assert [s.id for s in ans.sources] == ["handbook:2.2", "record:riya"]  # footnote order
    assert ans.statements[0].citations[0].number == 1 and ans.statements[1].citations[0].number == 2
    assert ans.citation_rate == 0.5


def test_gather_adds_record_rule_checks_and_the_rules_they_cite():
    sources = Advisor(CAT, retriever()).gather("Can I take CS 345 and CS 380 next semester?", CAT.students["riya"])
    ids = [s.id for s in sources]
    assert ids[0] == "record:riya" and "check:CS 345" in ids and "check:plan" in ids
    assert "handbook:6.2" in ids and "handbook:4.1" in ids  # rules named by the checks are citable
    plan = next(s for s in sources if s.id == "check:plan")
    assert "slot D" in plan.text


def test_ask_sends_sources_and_returns_verified_answer():
    draft = Draft(answerable=True, statements=[Statement(text="CS 345 needs CS 340.", citations=[Citation(source="course:CS 345", quote="Prerequisites: CS 340 Machine Learning")])])
    llm = FakeLLM(draft)
    ans = Advisor(CAT, retriever(), llm=llm).ask("What do I need before CS 345?", "riya")
    prompt = llm.calls[0]["prompt"]
    assert '<source id="course:CS 345"' in prompt and "Riya Sharma" in prompt and llm.calls[0]["output_format"] is Draft
    assert ans.statements[0].supported and ans.sources[0].id == "course:CS 345"
    assert Advisor(CAT, retriever(), llm=FakeLLM(draft)).ask("What do I need before CS 345?").student is None


def test_mentions_find_codes_titles_and_abbreviations():
    assert CAT.mentions("Can I take Deep Learning and NLP together?") == ["CS 345", "CS 380"]
    assert CAT.mentions("Is Machine Learning Operations hard?") == ["CS 460"]  # not also CS 340
    assert CAT.mentions("ml vs cs 340 vs os") == ["CS 340"]  # lowercase "ml"/"os" are ordinary words, the code still counts
    assert CAT.mentions("Tell me about the OS course and ML") == ["CS 260", "CS 340"]
    assert retriever().search("is NLP hard", k=2, mode="bm25")[0].chunk.id == "course:CS 380"


def test_longer_abbreviations_match_in_any_case():
    assert CAT.mentions("can i take nlp and dbms?") == ["CS 380", "CS 220"]


# Evals
def test_eval_questions_name_real_sources():
    from pathwise.eval import load_questions
    qs = load_questions()
    assert len({q.id for q in qs}) == len(qs)
    ids = CAT.chunk_ids()
    for q in qs:
        assert all(e in ids for e in q.retrieval_expect), q.id
        assert q.student is None or q.student in CAT.students, q.id
        assert all(e.split(":", 1)[1] == q.student for e in q.expect if e.startswith("record:")), q.id
        assert q.answerable == bool(q.expect), q.id  # out-of-scope questions expect nothing


def test_eval_scores_rankings_and_runs_offline(tmp_path):
    from pathwise.eval import Question, eval_answers, eval_context, eval_retrieval, markdown, score_ranking
    s = score_ranking(["a", "b", "c"], ["b", "z"])
    assert s["hit"] and s["recall"] == 0.5 and s["rr"] == 0.5 and s["missing"] == ["z"]
    qs = [Question(id="nlp", category="lookup", question="What does NLP cover?", expect=["course:CS 380"]),
          Question(id="fee", category="refusal", question="hostel fee?", answerable=False)]
    report = {"k": 3, "questions": 2, "retrieval": eval_retrieval(retriever(), qs, ["bm25", "hybrid"], k=3)}
    assert report["retrieval"]["bm25"]["hit"] == 1.0 and len(report["retrieval"]["bm25"]["rows"]) == 1  # refusals are not scored for retrieval
    draft = Draft(answerable=True, statements=[Statement(text="NLP covers language models.", citations=[Citation(source="course:CS 380", quote="word embeddings, language models")])])
    report["context_mode"], report["context"] = "hybrid", eval_context(Advisor(CAT, retriever(), mode="hybrid"), qs)
    assert report["context"]["recall"] == 1.0 and len(report["context"]["rows"]) == 1
    report["model"] = "fake"
    report["answers"] = a = eval_answers(Advisor(CAT, retriever(), llm=FakeLLM(draft)), qs, workers=1)
    assert a["citation_rate"] == 0.5 and a["source_recall"] == 1.0 and a["refusal_accuracy"] == 0.5  # the fee answer cites a source it was never shown, so that quote fails
    assert "| bm25 | 100% |" in markdown(report) and "`fee`: answerable was True" in markdown(report)


def test_gather_adds_prerequisites_of_named_courses():
    ids = [s.id for s in Advisor(CAT, retriever(), mode="bm25", k=1).gather("What do I need before deep learning?", None)]
    assert ids[0] == "course:CS 345" and "course:CS 340" in ids


def test_partial_plan_check_does_not_flag_underload():
    from pathwise.advisor import check_sources
    plan = next(s for s in check_sources("Deep Learning and NLP together?", CAT.students["riya"], CAT, []) if s.id == "check:plan")
    assert "slot D" in plan.text and "minimum" not in plan.text

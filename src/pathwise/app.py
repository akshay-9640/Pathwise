"""The Pathwise web app: one page that asks the advisor and shows every citation.

Runs locally (`pathwise`, then open http://127.0.0.1:8765). Claude is reached
through the Anthropic API, or through the Claude Code CLI with
PATHWISE_BACKEND=claude-code.
"""

import os
import threading
import time
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .advisor import Advisor, Answer
from .catalog import load_catalog
from .llm import LLMError
from .retrieval import Retriever
from .rules import standing

WEB = Path(__file__).parent / "web"

# Starting points for each demo student, so the page shows what the advisor can do.
SUGGESTIONS = {
    "": ["Is there a course about how search engines work?", "What happens if I fail a core course?",
         "Can a prerequisite be waived, and who decides?", "Who teaches Machine Learning this year?"],
    "riya": ["Can I take Deep Learning and NLP together next semester?", "Am I allowed to take 24 credits next term?",
             "Which AI electives can I take next semester?"],
    "kabir": ["Can I take Operating Systems next semester?", "What happens with my failed MA 102?"],
    "meera": ["How close am I to the AI and Machine Learning specialization?", "Can I start Capstone Project II next semester?"],
    "dev": ["How many credits can I take while I am on probation?", "How do I get off probation?"],
}


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    student: str | None = None


class Reply(Answer):
    seconds: float


def create_app(advisor: Advisor | None = None) -> FastAPI:
    catalog = load_catalog()
    if advisor is None:
        advisor = Advisor(catalog, Retriever(catalog.chunks(), mentions=catalog.mentions))
        # Load the embedding model and catalog vectors in the background, so the first question is quick.
        threading.Thread(target=lambda: advisor.retriever.vectors, daemon=True).start()
    app = FastAPI(title="Pathwise", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/meta")
    def meta():
        students = []
        for s in catalog.students.values():
            st = standing(s, catalog)
            students.append({"id": s.id, "name": s.name, "first": s.name.split()[0],
                             "summary": f"Year {s.year}, Semester {s.semester}. CGPA {st.cgpa}. {s.note}"})
        return {"program": catalog.program, "institution": catalog.institution, "students": students,
                "suggestions": SUGGESTIONS, "courses": len(catalog.courses), "sections": len(catalog.sections)}

    @app.post("/api/ask", response_model=Reply)
    def ask(body: Ask):
        if body.student and body.student not in catalog.students:
            raise HTTPException(404, "Unknown student.")
        start = time.time()
        try:
            answer = advisor.ask(body.question.strip(), body.student or None)
        except LLMError as e:
            raise HTTPException(502, str(e))
        except anthropic.APIError as e:
            raise HTTPException(502, f"The Claude API returned an error: {getattr(e, 'message', e)}")
        return Reply(**answer.model_dump(), seconds=round(time.time() - start, 1))

    return app


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PATHWISE_PORT", "8765"))
    print(f"Pathwise on http://127.0.0.1:{port}")
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")

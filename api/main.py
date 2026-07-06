from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from ragguard.model import check_groundedness, load_model

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = load_model()
    yield


app = FastAPI(title="RagGuard", lifespan=lifespan)


class CheckRequest(BaseModel):
    context: str
    answer: str


class CheckResponse(BaseModel):
    grounded: bool
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/check", response_model=CheckResponse)
def check(request: CheckRequest):
    return check_groundedness(_model, request.context, request.answer)

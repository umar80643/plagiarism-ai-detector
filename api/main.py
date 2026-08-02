"""FastAPI backend for plagiarism + AI-text detection.

Separated from the Streamlit UI so the same detection logic can be called
by any client (a script, a mobile app, another service) over a versioned
HTTP API, not just from inside a Streamlit process. Run with:

    uvicorn api.main:app --reload
"""
from __future__ import annotations
import logging

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.auth import require_api_key
from api.deps import ai_detector_mode, get_plagiarism_index
from api.schemas import AITextResponse, HealthResponse, PlagiarismResponse, TextRequest
from models.ai_text_detector import detect_ai_text_detailed
from utils.file_reader import UnsupportedFileTypeError, extract_text_from_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Plagiarism & AI Content Detector API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to known origins in a real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})


@app.get("/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index = get_plagiarism_index()
    return HealthResponse(
        status="ok",
        plagiarism_index_documents=len(index.document_names),
        ai_detector_mode=ai_detector_mode(),
    )


@app.post("/v1/detect/plagiarism", response_model=PlagiarismResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def detect_plagiarism_text(request: Request, payload: TextRequest) -> PlagiarismResponse:
    index = get_plagiarism_index()
    matches = index.score_document(payload.text)
    score = matches[0]["similarity"] if matches else 0.0
    return PlagiarismResponse(score=score, matches=matches, sentence_matches=index.score_sentences(payload.text))


@app.post("/v1/detect/plagiarism/file", response_model=PlagiarismResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def detect_plagiarism_file(request: Request, file: UploadFile) -> PlagiarismResponse:
    data = await file.read()
    try:
        text = extract_text_from_bytes(data, file.filename or "")
    except UnsupportedFileTypeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    index = get_plagiarism_index()
    matches = index.score_document(text)
    score = matches[0]["similarity"] if matches else 0.0
    return PlagiarismResponse(score=score, matches=matches, sentence_matches=index.score_sentences(text))


@app.post("/v1/detect/ai-text", response_model=AITextResponse, dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
def detect_ai_text_endpoint(request: Request, payload: TextRequest) -> AITextResponse:
    result = detect_ai_text_detailed(payload.text)
    return AITextResponse(
        score=round(result.ai_probability * 100, 2),
        label=result.label,
        human_probability=result.human_probability,
        ai_probability=result.ai_probability,
        confidence=result.confidence,
        explanation=result.explanation,
        backend=result.backend,
    )

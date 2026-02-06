from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag_engine import answer_question

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="Document-Aware RAG API",
    description="Backend API for document-based question answering",
    version="1.0.0"
)

# -----------------------------
# CORS (allow frontend access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Constants
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Request / Response Models
# -----------------------------
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# -----------------------------
# Ask Question Endpoint
# -----------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    answer = answer_question(payload.question)
    return {"answer": answer}

# -----------------------------
# Upload PDF Endpoint
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "status": "uploaded",
        "filename": file.filename,
        "message": "File uploaded successfully. It will be indexed automatically."
    }

import os
# 1. SET ENV VARS BEFORE ANYTHING ELSE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Force CPU only

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd, pdfplumber, docx, io, re, string

# -----------------------------
# OPTIMIZED SENTENCE MODEL
# -----------------------------
from sentence_transformers import SentenceTransformer, util

print("🚀 Loading lightweight semantic model...")

# ⚡ SMALL, SUPER-FAST MODEL (~30MB)
# This fits easily in Render Free Tier RAM
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="/tmp/cache"
)

# Warmup to avoid cold first request
model.encode("warmup test", convert_to_tensor=True)

print("✅ Lightweight model loaded & warmed up!")


# -----------------------------
# FAST API APP
# -----------------------------
app = FastAPI(title="ATS Resume Ranking Portal")
templates = Jinja2Templates(directory="src/api/templates")

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text(file: UploadFile):
    filename = file.filename.lower()
    file.file.seek(0)
    content = file.file.read()
    if not content:
        return ""
    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(content))
            return "\n".join([p.text for p in doc.paragraphs])
        elif filename.endswith(".txt"):
            return content.decode("utf-8")
        else:
            return ""
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return ""

def semantic_score(resume_text, job_text):
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_job = model.encode(job_text, convert_to_tensor=True)

    score = util.cos_sim(emb_resume, emb_job).item()
    score_percentage = (score + 1) / 2 * 100
    return round(score_percentage, 2)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/start", response_class=HTMLResponse)
def start(request: Request):
    return templates.TemplateResponse("start.html", {"request": request})


@app.get("/recruiter", response_class=HTMLResponse)
def recruiter_page(request: Request):
    return templates.TemplateResponse("recruiter.html", {"request": request})


# CHANGED: Removed 'async' because semantic_score is CPU heavy.
# Using 'def' makes FastAPI run this in a threadpool, preventing blocking.
@app.post("/rank_resumes", response_class=HTMLResponse)
def rank_resumes(request: Request, job_text: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        resumes = []
        for file in files:
            raw_text = extract_text(file)
            cleaned = clean_text(raw_text)
            if cleaned.strip():
                resumes.append({"filename": file.filename, "text": cleaned})

        if not resumes:
            return templates.TemplateResponse(
                "result_rank.html",
                {"request": request, "error": "No valid resumes processed. Files might be empty or unreadable."}
            )

        df = pd.DataFrame(resumes)
        job_clean = clean_text(job_text)

        scores = [semantic_score(text, job_clean) for text in df["text"]]
        df["similarity"] = scores
        df = df.sort_values(by="similarity", ascending=False)

        table_html = df[["filename", "similarity"]].to_html(classes="styled-table", index=False)
        return templates.TemplateResponse("result_rank.html", {"request": request, "table": table_html})
    
    except Exception as e:
        return templates.TemplateResponse("result_rank.html", {"request": request, "error": f"An error occurred: {str(e)}"})


@app.get("/candidate", response_class=HTMLResponse)
def candidate_page(request: Request):
    return templates.TemplateResponse("candidate.html", {"request": request})


# CHANGED: Removed 'async' here too.
@app.post("/upload_resume", response_class=HTMLResponse)
def upload_resume(request: Request, resume_file: UploadFile = File(...), job_text: str = Form(...)):
    try:
        resume_text = extract_text(resume_file)
        if not resume_text:
             return templates.TemplateResponse("result_score.html", {"request": request, "error": "Could not read resume text."})

        job_clean = clean_text(job_text)
        resume_clean = clean_text(resume_text)

        score = semantic_score(resume_clean, job_clean)

        return templates.TemplateResponse("result_score.html", {"request": request, "score": score})
    
    except Exception as e:
        return templates.TemplateResponse("result_score.html", {"request": request, "error": f"An error occurred: {str(e)}"})


@app.get("/docs_redirect", response_class=HTMLResponse)
def docs_redirect(request: Request):
    return templates.TemplateResponse("docs_redirect.html", {"request": request})
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd, pdfplumber, docx, io, re, string, os


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
    if filename.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise ValueError("Unsupported file format")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/start", response_class=HTMLResponse)
def start(request: Request):
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/recruiter", response_class=HTMLResponse)
def recruiter_page(request: Request):
    return templates.TemplateResponse("recruiter.html", {"request": request})

@app.post("/rank_resumes", response_class=HTMLResponse)
async def rank_resumes(request: Request, job_text: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        resumes = []
        for file in files:
            raw_text = extract_text(file)
            text = clean_text(raw_text)
            if text.strip():
                resumes.append({"filename": file.filename, "text": text})

        if not resumes:
            return templates.TemplateResponse(
                "result_rank.html",
                {"request": request, "error": "No valid resumes processed. Please recheck uploads."}
            )

        df = pd.DataFrame(resumes)
        job_clean = clean_text(job_text)

        vectorizer = TfidfVectorizer(max_features=3000)
        vectorizer.fit([job_clean])

        job_vec = vectorizer.transform([job_clean]).toarray()
        resume_vecs = vectorizer.transform(df["text"]).toarray()

        sims = cosine_similarity(job_vec, resume_vecs)[0] * 100
        df["similarity"] = sims.round(2)
        df = df.sort_values(by="similarity", ascending=False)

        table_html = df[["filename", "similarity"]].to_html(classes="styled-table", index=False)
        return templates.TemplateResponse("result_rank.html", {"request": request, "table": table_html})

    except Exception as e:
        return templates.TemplateResponse("result_rank.html", {"request": request, "error": str(e)})

@app.get("/candidate", response_class=HTMLResponse)
def candidate_page(request: Request):
    return templates.TemplateResponse("candidate.html", {"request": request})

@app.post("/upload_resume", response_class=HTMLResponse)
async def upload_resume(request: Request, resume_file: UploadFile = File(...), job_text: str = Form(...)):
    try:
        resume_text = extract_text(resume_file)
        job_clean = clean_text(job_text)
        resume_clean = clean_text(resume_text)

        vectorizer = TfidfVectorizer(max_features=3000)
        vectorizer.fit([job_clean])

        job_vec = vectorizer.transform([job_clean]).toarray()
        resume_vec = vectorizer.transform([resume_clean]).toarray()

        score = cosine_similarity(job_vec, resume_vec)[0][0] * 100
        score = round(score, 2)

        return templates.TemplateResponse("result_score.html", {"request": request, "score": score})
    except Exception as e:
        return templates.TemplateResponse("result_score.html", {"request": request, "error": str(e)})

@app.get("/docs_redirect", response_class=HTMLResponse)
def docs_redirect(request: Request):
    return templates.TemplateResponse("docs_redirect.html", {"request": request})

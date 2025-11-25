import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd, pdfplumber, docx, io, re, string
import numpy as np
from openai import OpenAI


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY is missing! App will crash on request.")

client = OpenAI(api_key=api_key)


def get_embedding(text):
    """Generate OpenAI embedding for any text."""
    
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity_score(emb_a, emb_b):
    """Calculates similarity between two embeddings."""
    a = np.array(emb_a)
    b = np.array(emb_b)
    
    raw_score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    min_thresh = 0.60
    max_thresh = 0.85

    if raw_score < min_thresh:
        return 0.00
    elif raw_score > max_thresh:
        return 99.99
    else:
        normalized = (raw_score-min_thresh)/(max_thresh-min_thresh)
        return round(normalized*100,2)

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text(file: UploadFile):
    filename = file.filename.lower()
    try:
        file.file.seek(0)
        content = file.file.read()
        if not content: return ""
        
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

app = FastAPI(title="ATS Resume Ranking Portal")
templates = Jinja2Templates(directory="src/api/templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/start", response_class=HTMLResponse)
def start(request: Request):
    return templates.TemplateResponse("start.html", {"request": request})

@app.get("/recruiter", response_class=HTMLResponse)
def recruiter_page(request: Request):
    return templates.TemplateResponse("recruiter.html", {"request": request})

@app.get("/candidate", response_class=HTMLResponse)
def candidate_page(request: Request):
    return templates.TemplateResponse("candidate.html", {"request": request})

@app.get("/docs_redirect", response_class=HTMLResponse)
def docs_redirect(request: Request):
    return templates.TemplateResponse("docs_redirect.html", {"request": request})


@app.post("/rank_resumes", response_class=HTMLResponse)
def rank_resumes(request: Request, job_text: str = Form(...), files: list[UploadFile] = File(...)):
    try:
        
        job_clean = clean_text(job_text)
        job_emb = get_embedding(job_clean) 

        resumes = []
        for file in files:
            raw_text = extract_text(file)
            cleaned = clean_text(raw_text)
            
            if cleaned.strip():
    
                resume_emb = get_embedding(cleaned)
                
                score = cosine_similarity_score(resume_emb, job_emb)
                
                resumes.append({
                    "filename": file.filename, 
                    "similarity": score
                })

        if not resumes:
            return templates.TemplateResponse(
                "result_rank.html",
                {"request": request, "error": "No valid resumes processed."}
            )

        df = pd.DataFrame(resumes)
        df = df.sort_values(by="similarity", ascending=False)

        table_html = df[["filename", "similarity"]].to_html(classes="styled-table", index=False)
        return templates.TemplateResponse("result_rank.html", {"request": request, "table": table_html})
    
    except Exception as e:
        return templates.TemplateResponse("result_rank.html", {"request": request, "error": f"Error: {str(e)}"})


@app.post("/upload_resume", response_class=HTMLResponse)
def upload_resume(request: Request, resume_file: UploadFile = File(...), job_text: str = Form(...)):
    try:
        resume_text = extract_text(resume_file)
        if not resume_text:
             return templates.TemplateResponse("result_score.html", {"request": request, "error": "Could not read resume text."})

        
        job_clean = clean_text(job_text)
        resume_clean = clean_text(resume_text)

        job_emb = get_embedding(job_clean)
        resume_emb = get_embedding(resume_clean)

        score = cosine_similarity_score(resume_emb, job_emb)

        return templates.TemplateResponse("result_score.html", {"request": request, "score": score})
    
    except Exception as e:
        return templates.TemplateResponse("result_score.html", {"request": request, "error": f"Error: {str(e)}"})
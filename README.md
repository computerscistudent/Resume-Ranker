# 💼 ATS Resume Ranking System  
An AI-powered Applicant Tracking System (ATS) built using **FastAPI, SpaCy, and Scikit-Learn**, designed to help recruiters efficiently rank multiple resumes and allow candidates to evaluate how well their resume matches a job description.

This project demonstrates real-world NLP applications such as **text extraction, cleaning, TF-IDF vectorization, and cosine similarity scoring**.

---

## 🚀 Features

### 🔍 **1. Recruiter Mode**
Upload **multiple resumes at once**, paste a job description, and instantly receive:
- A **ranked table** of resumes  
- Each resume’s **match score (%)**  
- Beautiful UI with scrollable, responsive table  
- Job-description-based TF-IDF feature extraction for stable & consistent scoring

### 👩‍💻 **2. Candidate Mode**
Candidates can:
- Upload their resume  
- Paste a job description  
- Get an instant **Match Score (ATS Score)**  
- Clean visual UI optimized for mobile responsiveness  

### 🛠 **3. Developer Mode (Swagger Docs)**
Includes a dedicated page for developers with:
- Auto-generated API documentation (`/docs`)  
- Live testing of endpoints  
- Example requests & responses  
- Great for QA, integrations, and automation


## 🧠 Tech Stack

### **Backend**
- **FastAPI** – High-performance Python API framework  
- **SpaCy** – Lemmatization & NLP preprocessing  
- **Scikit-Learn** – TF-IDF vectorization & cosine similarity  
- **Python-Docx / PDFPlumber** – Resume text extraction  

### **Frontend**
- **HTML5, CSS3**  
- **Responsive design (mobile & tablet friendly)**  
- Modern UI with Glassmorphism + Gradient themes

---

## 📂 Project Structure

ATS_Checker/
|
|-- src/
| |-- api/
| |-- main.py # FastAPI backend
| |-- templates/
| |-- home.html
| |-- start.html
| |-- recruiter.html
| |-- result_rank.html
| |-- candidate.html
| |-- result_score.html
| |-- redirect_docs.html
|
|-- static/
|-- README.md
|-- requirements.txt


## 2) Create a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

## 🧪 API Endpoints
POST /rank_resumes

Upload multiple resumes & job description → get ranking

POST /upload_resume

Upload one resume → get ATS match score

GET /docs

Swagger documentation

## 🧼 Text Preprocessing Includes

Lowercasing

Removing punctuation & numbers

Lemmatization

Removing stopwords

Token filtering (short words removed)

## 📊 Resume Matching Logic
✔ Extract resume text
✔ Clean with SpaCy NLP
✔ Build TF-IDF vectors (job description as vocabulary)
✔ Calculate cosine similarity
✔ Rank based on highest similarity percentage

This is the same technique used by real ATS systems.

## 📱 Fully Responsive UI

Works smoothly on phone, tablet, and laptop

Dynamic viewport height fixes mobile shrinking issue

Tables become scrollable on small screens

Clean, centered card views

## 🎯 Future Improvements (Optional)

Add Weighted Keyword Matching

Job role classification (ML model)

Section-wise resume scoring

Export PDF report for candidates

Add authentication for recruiter access


## ❤️ Credits

Built by Abhimanyu Singh with modern NLP & FastAPI.

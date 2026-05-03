# ResuMatch AI — Agentic Resume Analyzer

**Stack:** FastAPI · Streamlit · LangChain · ChromaDB · Docling · OpenAI

---

## Setup (do this once)

```bash
pip install -r requirements.txt
```

---

## Run

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Usage

1. Enter your OpenAI API key in the sidebar
2. Upload your resume (PDF or DOCX)
3. Paste the job description
4. Click **Analyze**
5. View match score, skill gaps, recommendations
6. Switch to **RAG Chatbot** tab to ask questions about your analysis

---

## Project Structure

```
resumatch/
├── backend/
│   └── main.py          # FastAPI — all AI logic
├── frontend/
│   └── app.py           # Streamlit UI
├── requirements.txt
└── README.md
```

---

## How it works

Same pipeline as the IBM SkillBuilds notebook:
- **Docling** parses the resume PDF/DOCX
- **LLM (GPT-3.5)** extracts skills from resume and JD
- **Skill matching** calculates match score + gaps
- **LLM** generates recommendations for missing skills
- **ChromaDB RAG** builds a chatbot over your analysis results

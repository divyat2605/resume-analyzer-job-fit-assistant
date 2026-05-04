# 🎯 ResuMatch AI
### *Agentic Resume Analyzer & Job-Fit Assistant*

<div align="center">

![ResuMatch AI Demo](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcDd4bXJ6aWRhcXVxbmN5dWpjaW5vMXFydTBxcWc1b2FiYWZ0aXI2NiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qgQUggAC3Pfv687qPC/giphy.gif)

> **Upload your resume. Paste a job description. Get instant AI-powered insights on your fit — and chat with your results.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)

![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=flat-square)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

</div>

---

## ✨ What It Does

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExamx3NjdxM3dpNHgzNmU3bDM3NTdqZjN4dHZmamx3dHM5NmZkcTliNSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LaVp0AyqR5bGsC5Cbm/giphy.gif" width="400" alt="AI analyzing documents"/>
</div>

ResuMatch AI is a full-stack AI application that helps job seekers understand **how well their resume matches a job description** — and what to do about it.

| Feature | Description |
|---|---|
| 📄 **Resume Parsing** | Extracts text from PDF and DOCX resumes |
| 🧠 **LLM Skill Extraction** | Uses GPT-4o-mini to identify skills from both resume and JD |
| 📊 **Match Scoring** | Calculates a percentage fit score based on skill overlap |
| 💡 **Actionable Recommendations** | Generates 5 specific career coaching tips for improvement |
| 💬 **RAG Chatbot** | Chat with your results using ChromaDB + HuggingFace embeddings |
| ⬇️ **Export** | Download your full analysis as a JSON file |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│                  Streamlit Frontend                         │
│                                                             │
│   ┌─────────────────────┐   ┌─────────────────────────┐    │
│   │   📄 Analyze Tab    │   │   💬 RAG Chatbot Tab    │    │
│   │  Upload PDF/DOCX    │   │  Ask questions about    │    │
│   │  Paste JD text      │   │  your resume & analysis │    │
│   └──────────┬──────────┘   └────────────┬────────────┘    │
└──────────────┼──────────────────────────-┼─────────────────┘
               │ POST /analyze             │ POST /chat
               ▼                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI BACKEND                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ parse_resume │  │extract_skills│  │   match_skills   │  │
│  │  (PDF/DOCX)  │─▶│    (LLM)     │─▶│  (set overlap)   │  │
│  └──────────────┘  └──────────────┘  └────────┬─────────┘  │
│                                               │             │
│  ┌────────────────────────────────────────────▼──────────┐  │
│  │              generate_recommendations (LLM)           │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  build_rag_chain                       │  │
│  │  Resume + JD + Summary → Chunks → Embeddings → Chroma │  │
│  │  Question → Retriever → GPT-4o-mini → Answer          │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────┬──────────────────────────────┘
                               │
              ┌────────────────┼─────────────────┐
              ▼                ▼                  ▼
       ┌─────────────┐  ┌──────────────┐  ┌────────────────┐
       │  OpenAI API │  │  ChromaDB    │  │  HuggingFace   │
       │ GPT-4o-mini │  │ (in-memory   │  │  Embeddings    │
       │             │  │  vectorstore)│  │ all-MiniLM-L6  │
       └─────────────┘  └──────────────┘  └────────────────┘
```

### 🔄 Analyze Flow

```
PDF/DOCX Upload
      │
      ▼
 parse_resume()          ← PdfReader / raw decode
      │
      ▼
extract_skills_llm()     ← GPT-4o-mini on resume text
      │
extract_skills_llm()     ← GPT-4o-mini on JD text
      │
      ▼
  match_skills()         ← Set intersection & difference
      │
      ▼
generate_recommendations() ← GPT-4o-mini career coach prompt
      │
      ▼
  AnalysisResult JSON    → Streamlit UI
```

### 💬 RAG Chat Flow

```
Resume + JD + Analysis Summary
            │
            ▼
  RecursiveCharacterTextSplitter (500 chars, 50 overlap)
            │
            ▼
  HuggingFace Embeddings (all-MiniLM-L6-v2)
            │
            ▼
       ChromaDB VectorStore
            │
      User Question
            │
            ▼
     Retriever (top-3 chunks)
            │
            ▼
   GPT-4o-mini (context + question)
            │
            ▼
          Answer
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| 🖥️ **Frontend** | Streamlit | Interactive UI with tabs, chat, file upload |
| ⚡ **Backend** | FastAPI | REST API with async endpoints |
| 🧠 **LLM** | OpenAI GPT-4o-mini | Skill extraction, recommendations, Q&A |
| 🔗 **Orchestration** | LangChain | LLM chains, text splitting, retrievers |
| 🗄️ **Vector Store** | ChromaDB | In-memory semantic search for RAG |
| 📐 **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local, free sentence embeddings |
| 📄 **PDF Parsing** | pypdf | Extract text from resume PDFs |

---

## 🚀 Getting Started

<div align="center">
<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzV0dGpveXlkMm1iMHRrb3Z2NGJ6NThsMHplb3l6dW9zanNhc3Y5NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/du3J3cXyzhj75IOgvA/giphy.gif" width="350" alt="rocket launch"/>
</div>

### Prerequisites

- Python 3.10+
- An [OpenAI API Key](https://platform.openai.com/api-keys)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/resumatch-ai.git
cd resumatch-ai
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Key dependencies</summary>

```
fastapi
uvicorn
streamlit
langchain
langchain-openai
langchain-community
langchain-huggingface
langchain-text-splitters
chromadb
pypdf
sentence-transformers
requests
python-multipart
```
</details>

### 3️⃣ Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Visit `/docs` for the interactive Swagger UI.

### 4️⃣ Start the Streamlit frontend

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 5️⃣ Use the app

1. Enter your **OpenAI API key** in the sidebar
2. Upload your **resume** (PDF or DOCX)
3. Paste the **job description**
4. Click **🔍 Analyze**
5. Switch to the **💬 RAG Chatbot** tab to ask follow-up questions

---

## 📡 API Reference

### `POST /analyze`

Analyzes a resume against a job description.

| Field | Type | Description |
|---|---|---|
| `file` | `UploadFile` | Resume file (PDF or DOCX) |
| `jd_text` | `string` | Job description text |
| `api_key` | `string` | OpenAI API key |

**Response:**
```json
{
  "match_score": 72.5,
  "matched_skills": ["python", "fastapi", "machine learning"],
  "missing_skills": ["kubernetes", "terraform"],
  "recommendations": [
    "1. Add Kubernetes experience to your projects section...",
    "..."
  ],
  "resume_text": "...",
  "jd_text": "..."
}
```

### `POST /chat`

Ask a contextual question about your resume analysis.

**Query param:** `api_key`

**Request body:**
```json
{
  "question": "Which skills should I prioritize?",
  "resume_text": "...",
  "jd_text": "...",
  "analysis_summary": "..."
}
```

**Response:**
```json
{
  "answer": "Based on the job description, you should prioritize..."
}
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 📁 Project Structure

```
resumatch-ai/
├── main.py           # FastAPI backend — all core logic & endpoints
├── app.py            # Streamlit frontend — UI & API calls
├── requirements.txt  # Python dependencies
└── README.md         # You are here 📍
```

---

## 🔮 Future Improvements

- [ ] 🗂️ Support DOCX parsing via `python-docx`
- [ ] 🔄 Persistent ChromaDB storage across sessions
- [ ] 📊 Radar chart visualization of skill coverage
- [ ] 🌐 Hosted deployment (Render / Railway / HuggingFace Spaces)
- [ ] 🤖 LangGraph agentic loop for iterative resume rewriting
- [ ] 🔑 Support for other LLM providers (Gemini, Claude, etc.)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

MIT License — feel free to use, fork, and build on top of this.

---

<div align="center">

<img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHF6bHozemM2MzZ4aHdwdm51OGZhZm9xbjFkcWo3amRmbzV3cWZhaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LnQjpWaON8nhr21vNW/giphy.gif" width="60"/>

Made with ❤️ and too much ☕

*If this helped you land the job, give it a ⭐ — it means a lot!*

</div>

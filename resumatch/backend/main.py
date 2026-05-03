from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile

# LangChain + OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Docling for resume parsing
from docling.document_converter import DocumentConverter

app = FastAPI(title="ResuMatch AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class AnalysisResult(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendations: List[str]
    resume_text: str
    jd_text: str

class ChatRequest(BaseModel):
    question: str
    resume_text: str
    jd_text: str
    analysis_summary: str

# ---------- Core Logic (same as notebook) ----------

def parse_resume_docling(file_path: str) -> str:
    """Parse resume using Docling (same as notebook)"""
    try:
        converter = DocumentConverter()
        result = converter.convert(file_path)
        return result.document.export_to_markdown()
    except Exception:
        # fallback: read raw text
        with open(file_path, "rb") as f:
            return f.read().decode(errors="ignore")


def extract_skills_llm(text: str, context: str, api_key: str) -> List[str]:
    """Use LLM to extract skills from text"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)
    prompt = f"""Extract a list of technical and professional skills from the following {context}.
Return ONLY a comma-separated list of skills, nothing else.

Text:
{text[:3000]}

Skills:"""
    response = llm.invoke(prompt)
    skills = [s.strip().lower() for s in response.content.split(",") if s.strip()]
    return skills


def match_skills(resume_skills: List[str], jd_skills: List[str]):
    """Skill matching logic - same as notebook"""
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    matched = list(resume_set & jd_set)
    missing = list(jd_set - resume_set)
    score = (len(matched) / len(jd_set)) * 100 if jd_set else 0
    return round(score, 2), matched, missing


def generate_recommendations(missing_skills: List[str], resume_text: str, jd_text: str, api_key: str) -> List[str]:
    """LLM-powered recommendations - same as notebook"""
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.3)
    prompt = f"""You are a career coach. Based on the missing skills below, give 5 specific, actionable recommendations to improve this resume for the job.

Missing skills: {', '.join(missing_skills[:10])}

Job Description (excerpt): {jd_text[:1000]}

Give exactly 5 recommendations as a numbered list. Be specific and concise."""
    response = llm.invoke(prompt)
    lines = [l.strip() for l in response.content.split("\n") if l.strip() and l.strip()[0].isdigit()]
    return lines if lines else [response.content]


def build_rag_chain(resume_text: str, jd_text: str, analysis_summary: str, api_key: str):
    """Build RAG chatbot over analysis results - same as notebook"""
    combined = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

ANALYSIS SUMMARY:
{analysis_summary}
"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([combined])

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = Chroma.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0.2)

    prompt_template = PromptTemplate(
        template="""You are a helpful career assistant. Use the resume, job description, and analysis below to answer the question.

Context:
{context}

Question: {question}

Answer helpfully and concisely:""",
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template}
    )
    return chain


# ---------- API Endpoints ----------

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume(
    file: UploadFile = File(...),
    jd_text: str = Form(...),
    api_key: str = Form(...)
):
    # Save uploaded file temporarily
    suffix = "." + file.filename.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        resume_text = parse_resume_docling(tmp_path)
        resume_skills = extract_skills_llm(resume_text, "resume", api_key)
        jd_skills = extract_skills_llm(jd_text, "job description", api_key)
        score, matched, missing = match_skills(resume_skills, jd_skills)
        recommendations = generate_recommendations(missing, resume_text, jd_text, api_key)

        return AnalysisResult(
            match_score=score,
            matched_skills=matched,
            missing_skills=missing,
            recommendations=recommendations,
            resume_text=resume_text,
            jd_text=jd_text
        )
    finally:
        os.unlink(tmp_path)


@app.post("/chat")
async def chat(req: ChatRequest, api_key: str):
    chain = build_rag_chain(req.resume_text, req.jd_text, req.analysis_summary, api_key)
    result = chain.run(req.question)
    return {"answer": result["result"]}


@app.get("/health")
def health():
    return {"status": "ok"}

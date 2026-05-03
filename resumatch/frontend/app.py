import streamlit as st
import requests
import json

# ---------- Page Config ----------
st.set_page_config(
    page_title="ResuMatch AI",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 ResuMatch AI")
st.caption("Agentic Resume Analyzer & Job-Fit Assistant")

# ---------- Sidebar - API Key ----------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    backend_url = st.text_input("Backend URL", value="http://localhost:8000")
    st.divider()
    st.markdown("**Tech Stack**")
    st.markdown("- FastAPI backend\n- LangChain + LangGraph\n- ChromaDB RAG\n- Docling parser\n- OpenAI GPT-3.5")

# ---------- Session State ----------
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Main Tabs ----------
tab1, tab2 = st.tabs(["📄 Analyze Resume", "💬 RAG Chatbot"])

# ---- Tab 1: Analysis ----
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"])

    with col2:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste JD here", height=200, placeholder="Paste the job description...")

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_btn:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload a resume.")
        elif not jd_text.strip():
            st.error("Please paste a job description.")
        else:
            with st.spinner("Analyzing your resume... (this may take 20-30 seconds)"):
                try:
                    response = requests.post(
                        f"{backend_url}/analyze",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data={"jd_text": jd_text, "api_key": api_key},
                        timeout=120
                    )
                    if response.status_code == 200:
                        st.session_state.analysis = response.json()
                        st.session_state.chat_history = []
                        st.success("Analysis complete!")
                    else:
                        st.error(f"Backend error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure FastAPI is running on the backend URL.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display Results
    if st.session_state.analysis:
        a = st.session_state.analysis
        st.divider()

        # Score
        score = a["match_score"]
        col_score, col_empty = st.columns([1, 2])
        with col_score:
            color = "green" if score >= 70 else "orange" if score >= 40 else "red"
            st.markdown(f"""
            <div style="text-align:center; padding:20px; border-radius:12px; background:#f0f2f6;">
                <div style="font-size:48px; font-weight:700; color:{color};">{score}%</div>
                <div style="font-size:16px; color:#666;">Match Score</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Skills
        col_matched, col_missing = st.columns(2)
        with col_matched:
            st.subheader("✅ Matched Skills")
            if a["matched_skills"]:
                for skill in a["matched_skills"]:
                    st.markdown(f"- `{skill}`")
            else:
                st.info("No direct skill matches found.")

        with col_missing:
            st.subheader("❌ Missing Skills")
            if a["missing_skills"]:
                for skill in a["missing_skills"]:
                    st.markdown(f"- `{skill}`")
            else:
                st.success("No skill gaps found!")

        st.divider()

        # Recommendations
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(a["recommendations"], 1):
            st.markdown(f"{rec}")

        # Export
        st.divider()
        result_json = json.dumps({
            "match_score": a["match_score"],
            "matched_skills": a["matched_skills"],
            "missing_skills": a["missing_skills"],
            "recommendations": a["recommendations"]
        }, indent=2)
        st.download_button(
            "⬇️ Download Analysis (JSON)",
            data=result_json,
            file_name="resumatch_analysis.json",
            mime="application/json"
        )

# ---- Tab 2: RAG Chatbot ----
with tab2:
    st.subheader("💬 Ask about your analysis")

    if not st.session_state.analysis:
        st.info("Run an analysis first in the 'Analyze Resume' tab.")
    else:
        a = st.session_state.analysis
        analysis_summary = f"""
Match Score: {a['match_score']}%
Matched Skills: {', '.join(a['matched_skills'])}
Missing Skills: {', '.join(a['missing_skills'])}
Recommendations: {' | '.join(a['recommendations'])}
"""
        # Example questions
        st.caption("Try asking:")
        cols = st.columns(3)
        example_qs = [
            "Why is my match score not higher?",
            "Which skills should I prioritize?",
            "How can I improve my resume for this role?"
        ]
        for i, q in enumerate(example_qs):
            if cols[i].button(q, key=f"eq_{i}"):
                st.session_state.pending_question = q

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Handle example question click
        prefill = st.session_state.get("pending_question", "")
        if prefill:
            del st.session_state.pending_question

        # Chat input
        user_input = st.chat_input("Ask anything about your resume or job fit...")
        question = prefill or user_input

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = requests.post(
                            f"{backend_url}/chat",
                            data={
                                "api_key": api_key,
                            },
                            json={
                                "question": question,
                                "resume_text": a["resume_text"],
                                "jd_text": a["jd_text"],
                                "analysis_summary": analysis_summary
                            },
                            timeout=60
                        )
                        # Note: need to send api_key differently since we have both json and form
                        # Retry with proper format
                        payload = {
                            "question": question,
                            "resume_text": a["resume_text"],
                            "jd_text": a["jd_text"],
                            "analysis_summary": analysis_summary,
                            "api_key": api_key
                        }
                        response = requests.post(
                            f"{backend_url}/chat",
                            params={"api_key": api_key},
                            json={
                                "question": question,
                                "resume_text": a["resume_text"],
                                "jd_text": a["jd_text"],
                                "analysis_summary": analysis_summary
                            },
                            timeout=60
                        )
                        if response.status_code == 200:
                            answer = response.json()["answer"]
                            st.write(answer)
                            st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        else:
                            st.error(f"Error: {response.text}")
                    except Exception as e:
                        st.error(f"Chat error: {str(e)}")

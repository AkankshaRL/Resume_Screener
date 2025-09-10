import streamlit as st
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class State(dict):
    jd: str
    resume: str
    score: int
    reasons: list

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

def scoring_node(state: State):
    prompt = f"""
    You are a resume screening assistant.
    Compare the following Job Description and Resume.
    Give a single integer score (0–100) based on how well the resume matches the JD.

    Job Description:
    {state['jd']}

    Resume:
    {state['resume']}

    Output only the number.
    """
    response = llm.invoke(prompt)
    state["score"] = int("".join([c for c in response.content if c.isdigit()]))
    return state

def reasoning_node(state: State):
    prompt = f"""
    You scored this resume {state['score']} out of 100 compared to the JD.
    Now explain in exactly 3 concise bullet points why you gave this score.

    Job Description:
    {state['jd']}

    Resume:
    {state['resume']}
    """
    response = llm.invoke(prompt)
    state["reasons"] = [line.strip("-• ") for line in response.content.split("\n") if line.strip()][:3]
    return state

# ----------------------------
# LangGraph Workflow
# ----------------------------
graph = StateGraph(State)

graph.add_node("score", scoring_node)
graph.add_node("reason", reasoning_node)

graph.add_edge(START, "score")
graph.add_edge("score", "reason")
graph.add_edge("reason", END)

app = graph.compile()

st.set_page_config(page_title="Resume vs JD Matcher", layout="centered")

st.title("Resume–JD Match Scorer")
st.markdown("Upload/paste a **Job Description** and a **Resume** to get a match score (0–100) and 3 reasons.")

jd_text = st.text_area("Job Description", height=200, placeholder="Paste the job description here...")
resume_text = st.text_area("Resume", height=200, placeholder="Paste the candidate's resume here...")

if st.button("Evaluate Match"):
    if jd_text.strip() and resume_text.strip():
        with st.spinner("Evaluating match..."):
            result = app.invoke({"jd": jd_text, "resume": resume_text})
        
        st.success(f"Match Score: **{result['score']} / 100**")
        st.subheader("Reasons")
        for r in result["reasons"]:
            st.write(f"- {r}")
    else:
        st.warning("Please enter both Job Description and Resume.")

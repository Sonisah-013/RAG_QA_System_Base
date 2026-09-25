import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from src.ingestion.loader import load_single_pdf
from src.ingestion.chunker import split_documents
from src.retrieval.vectorstore import build_vectorstore
from src.generation.qa_chain import build_qa_chain, ask_question

load_dotenv()

st.set_page_config(page_title="RAG Q&A System", page_icon="📄", layout="centered")
st.title("📄 RAG Q&A System")
st.caption("Upload a PDF, ask questions, get answers grounded in the document.")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_filename" not in st.session_state:
    st.session_state.indexed_filename = None


def build_index_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        documents = load_single_pdf(tmp_path)
        chunks = split_documents(documents)
        vectorstore = build_vectorstore(chunks)
        st.session_state.qa_chain = build_qa_chain(vectorstore)
        st.session_state.indexed_filename = uploaded_file.name
        st.session_state.chat_history = []
        st.success(f"Indexed {uploaded_file.name} ({len(chunks)} chunks). Ask away!")
    finally:
        os.unlink(tmp_path)


with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file is not None and uploaded_file.name != st.session_state.indexed_filename:
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        else:
            build_index_from_pdf(uploaded_file)
    if st.session_state.indexed_filename:
        st.info(f"Currently indexed: {st.session_state.indexed_filename}")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.qa_chain is None:
    st.info("Upload a PDF from the sidebar to get started.")
else:
    question = st.text_input("Ask a question about the document", key="question_input")
    if st.button("Ask", type="primary") and question.strip():
        answer, sources = ask_question(st.session_state.qa_chain, question)
        st.session_state.chat_history.append((question, answer, sources))
    for q, a, srcs in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.write(a)
        with st.expander("Sources"):
            for i, s in enumerate(srcs, start=1):
                st.write(f"{i}. {s}")
        st.divider()

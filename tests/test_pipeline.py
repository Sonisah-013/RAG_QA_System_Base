"""
test_pipeline.py
Offline tests for the RAG pipeline. No model downloads, no API keys.

Run from the project root:
    pip install pytest
    python -m pytest tests -v
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from src.ingestion.chunker import split_documents
from src.ingestion.loader import load_documents
from src.generation.prompts import qa_prompt
from src.retrieval import vectorstore as vs_module
from src.retrieval.retriever import retrieve_docs


# ---------- helpers / fixtures ----------

def make_docs(n=3, words=400):
    """Fake multi-page documents with source/page metadata."""
    return [
        Document(
            page_content=" ".join(f"word{i}_{j}" for j in range(words)),
            metadata={"source": "sample.pdf", "page": i},
        )
        for i in range(n)
    ]


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    """A small FAISS index built with fake embeddings in a temp folder.
    Never touches your real faiss_index/."""
    monkeypatch.setattr(vs_module, "get_embeddings", lambda *a, **k: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "faiss_index")
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=30)
    store = vs_module.create_vectorstore(chunks)
    return store, chunks


# ---------- chunker ----------

def test_split_empty_returns_empty_list():
    assert split_documents([]) == []


def test_chunks_respect_size_limit():
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=30)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_chunks_keep_source_and_page_metadata():
    chunks = split_documents(make_docs(), chunk_size=300, chunk_overlap=30)
    for c in chunks:
        assert c.metadata["source"] == "sample.pdf"
        assert "page" in c.metadata


# ---------- loader ----------

def test_load_documents_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does_not_exist")


def test_load_documents_reads_txt(tmp_path):
    (tmp_path / "notes.txt").write_text("RAG combines retrieval with generation.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "retrieval" in docs[0].page_content


def test_load_documents_empty_dir_returns_empty_list(tmp_path):
    assert load_documents(tmp_path) == []


@pytest.mark.xfail(reason="loader.py stores the full file path in metadata['source']; clean it to the file name")
def test_loaded_source_is_clean_file_name(tmp_path):
    (tmp_path / "notes.txt").write_text("hello world", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert docs[0].metadata["source"] == "notes.txt"


# ---------- vectorstore ----------

def test_create_and_load_roundtrip(fake_index):
    store, chunks = fake_index
    loaded = vs_module.load_vectorstore()
    assert loaded.index.ntotal == len(chunks)


def test_load_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(vs_module, "get_embeddings", lambda *a, **k: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr(vs_module, "INDEX_PATH", tmp_path / "nothing_here")
    with pytest.raises(FileNotFoundError):
        vs_module.load_vectorstore()


# ---------- retriever ----------

def test_retrieve_empty_query_raises(fake_index):
    store, _ = fake_index
    with pytest.raises(ValueError):
        retrieve_docs("   ", vectorstore=store)


def test_retrieve_returns_k_docs_with_metadata(fake_index):
    store, _ = fake_index
    results = retrieve_docs("word1_5", vectorstore=store, k=3)
    assert len(results) == 3
    assert all("source" in d.metadata for d in results)


# ---------- prompt ----------

def test_prompt_has_no_stray_text_after_context_label():
    text = qa_prompt.format(**{v: "x" for v in qa_prompt.input_variables})
    assert "Context:A" not in text


def test_prompt_matches_retrieval_chain_input_key():
    # create_retrieval_chain passes the user's text as "input",
    # so the prompt must use {input}, not {question}.
    # This test FAILS until prompts.py is fixed.
    assert "input" in qa_prompt.input_variables
    assert "context" in qa_prompt.input_variables


# ---------- qa_chain (LLM mocked) ----------

class FakeChain:
    def invoke(self, payload):
        return {
            "answer": "RAG stands for Retrieval-Augmented Generation.",
            "context": [
                Document(
                    page_content="RAG stands for Retrieval-Augmented Generation.",
                    metadata={"source": "intro.pdf", "page": 2},
                )
            ],
        }


def test_ask_question_returns_answer_and_sources():
    from src.generation.qa_chain import ask_question

    answer, sources = ask_question(FakeChain(), "What is RAG?")
    assert "Retrieval-Augmented" in answer
    assert len(sources) == 1
    assert "intro.pdf" in sources[0]


def test_ask_question_empty_raises():
    from src.generation.qa_chain import ask_question

    with pytest.raises(ValueError):
        ask_question(FakeChain(), "")
import os
import streamlit as st
import pdfplumber
import docx
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers embedding model
LLM_MODEL   = "distilgpt2"        # HuggingFace text-generation model
# ────────────────────────────────────────────────────────────────────────────────

# Initialize embedding model
st_model = SentenceTransformer(EMBED_MODEL)

class STEEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.tolist()
    def embed_query(self, text: str) -> list[float]:
        emb = st_model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return emb[0].tolist()

embedder = STEEmbedder()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Configure HuggingFace LLM pipeline using max_new_tokens instead of max_length
hf_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL,
    device=-1,
    max_new_tokens=200,
    do_sample=False,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Persist FAISS in session state
if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = None

# ── Helper functions ────────────────────────────────────────────────────────────

def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        pages = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    pages.append(txt)
        return "\n".join(pages)
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def ingest_docs():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No .pdf/.docx files found in 'documents/'.")
        return

    all_texts, metadatas = [], []
    for fname in files:
        raw = extract_text(os.path.join(DATA_DIR, fname))
        chunks = splitter.split_text(raw)
        all_texts.extend(chunks)
        metadatas.extend([{"source": fname}] * len(chunks))

    st.session_state['vectordb'] = FAISS.from_texts(
        texts=all_texts,
        embedding=embedder,
        metadatas=metadatas
    )
    st.success(f"Ingested {len(files)} files → {len(all_texts)} chunks.")


def query_docs(question: str, top_k: int = 5) -> str:
    vectordb = st.session_state['vectordb']
    if vectordb is None:
        return "⚠️ Please click **Ingest Documents** first."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": top_k})
    )
    return qa.run(question)

# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Free Local RAG", layout="wide")
st.title("Intelligent RFQ Repository")

if st.sidebar.button("Ingest Documents"):
    with st.spinner("Building FAISS index..."):
        ingest_docs()

k = st.sidebar.slider("Top K passages", min_value=1, max_value=10, value=5)

query = st.text_input("🔍 Ask a question about your documents:")
if st.button("Search"):
    if not query:
        st.warning("Enter a question to search.")
    else:
        with st.spinner("Generating answer..."):
            answer = query_docs(query, k)
            st.subheader("Answer")
            st.write(answer)

st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Place your `.pdf` and `.docx` files in the `documents/` folder.
    2. Click **Ingest Documents** to build the index (only needs to be done once).
    3. Type your question and hit **Search** to retrieve answers.
    """
)

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
# Embedding model
EMBED_MODEL = "all-MiniLM-L6-v2"
# LLM model (use a small model for CPU inference)
LLM_MODEL   = "distilgpt2"
# ────────────────────────────────────────────────────────────────────────────────

# 1) SentenceTransformer for embeddings
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

# 2) HuggingFace pipeline as LLM
hf_pipeline = pipeline(
    "text-generation",
    model=LLM_MODEL,
    device=-1,
    max_length=200,
    do_sample=False
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# 3) FAISS vector store
vectordb = None

# ── HELPERS ─────────────────────────────────────────────────────────────────────
def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        texts = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    texts.append(txt)
        return "\n".join(texts)
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def ingest_docs():
    global vectordb
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No .pdf/.docx files found in 'documents/'.")
        return

    texts, metas = [], []
    for fname in files:
        raw = extract_text(os.path.join(DATA_DIR, fname))
        chunks = splitter.split_text(raw)
        texts.extend(chunks)
        metas.extend([{"source": fname}] * len(chunks))

    vectordb = FAISS.from_texts(texts, embedder, metadatas=metas)
    st.success(f"Ingested {len(files)} files into FAISS ({len(texts)} chunks)")


def query_docs(q: str, k: int = 5) -> str:
    if vectordb is None:
        return "Please click 'Ingest Documents' first."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": k})
    )
    return qa.run(q)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Free Local RAG (HF)", layout="wide")
st.title("📚 Free Local RAG with FAISS & HuggingFace")

if st.sidebar.button("Ingest Documents"):
    with st.spinner("Building index..."):
        ingest_docs()

k = st.sidebar.slider("Top K passages", 1, 10, 5)

query = st.text_input("Ask a question about your documents:")
if st.button("Search"):
    if not query:
        st.warning("Enter a question.")
    else:
        with st.spinner("Generating answer..."):
            ans = query_docs(query, k)
            st.subheader("Answer")
            st.write(ans)

st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Place your `.pdf` and `.docx` files in `documents/`.
    2. Click **Ingest Documents** to embed and index.
    3. Ask your question and hit **Search**.
    """
)

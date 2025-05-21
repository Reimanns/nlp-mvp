import os
import streamlit as st
import pdfplumber
import docx
import numpy as np

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "documents"
MODEL_PATH  = "models/ggml-gpt4all-j-v1.3-groovy.bin"   # your GPT4All .bin
EMBED_MODEL = "all-MiniLM-L6-v2"                       # sentence-transformers model
# ────────────────────────────────────────────────────────────────────────────────

# 1) Load SentenceTransformer once
st_model = SentenceTransformer(EMBED_MODEL)

# 2) Embedder wrapper for LangChain
class STEEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embs = st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs.tolist()
    def embed_query(self, text: str) -> list[float]:
        emb = st_model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return emb[0].tolist()

embedder = STEEmbedder()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# 3) GPT4All LLM
llm = GPT4All(model=MODEL_PATH, n_threads=4, verbose=False)

# 4) Hold FAISS in memory
vectordb = None

# ── HELPERS ─────────────────────────────────────────────────────────────────────
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
    global vectordb
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No .pdf/.docx in `documents/`.")
        return

    texts, metas = [], []
    for fname in files:
        raw = extract_text(os.path.join(DATA_DIR, fname))
        chunks = splitter.split_text(raw)
        texts.extend(chunks)
        metas.extend([{"source": fname}] * len(chunks))

    vectordb = FAISS.from_texts(texts, embedder, metadatas=metas)
    st.success(f"Ingested {len(files)} files → {len(texts)} chunks.")

def query_docs(q: str, k: int=5) -> str:
    if vectordb is None:
        return "⚠️ Please click **Ingest Documents** first."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": k})
    )
    return qa.run(q)

# ── UI ─────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Free Local RAG", layout="wide")
st.title("📚 Free Local RAG with FAISS & GPT4All")

# Sidebar
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Building FAISS index..."):
        ingest_docs()

top_k = st.sidebar.slider("Top K passages", 1, 10, 5)

# Main
query = st.text_input("🔍 Ask a question about your documents:")
if st.button("Search"):
    if not query:
        st.warning("Enter a question to search.")
    else:
        with st.spinner("Generating answer..."):
            ans = query_docs(query, top_k)
            st.subheader("Answer")
            st.write(ans)

st.markdown("---")
st.markdown(
    """
    **Instructions**  
    1. Put your `.pdf`/`.docx` into the `documents/` folder.  
    2. Download the GPT4All `.bin` to `models/ggml-gpt4all-j-v1.3-groovy.bin`.  
    3. Click **Ingest Documents**.  
    4. Ask questions and enjoy—fully offline!
    """
)

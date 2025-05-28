import os
import streamlit as st
from PIL import Image
import pdfplumber
import docx

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
LLM_MODEL   = "distilgpt2"         # HuggingFace text-generation model
LOGO_PATH   = "logo.png"           # Path to your Citadel logo
# ────────────────────────────────────────────────────────────────────────────────

# Page setup
st.set_page_config(page_title="Intelligent RFQ Repository", layout="wide")

# Header with logo
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(LOGO_PATH):
        logo = Image.open(LOGO_PATH)
        st.image(logo, width=120)
with col2:
    st.title("Intelligent RFQ Repository")

# Load embedding model once
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)

st_model = load_embedding_model()

class STEEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
    def embed_query(self, text: str) -> list[float]:
        return st_model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()

embedder = STEEmbedder()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Load LLM pipeline once
@st.cache_resource
def load_llm_pipeline():
    hf_pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        device=-1,
        max_new_tokens=200,
        do_sample=False,
        truncation=True
    )
    return HuggingFacePipeline(pipeline=hf_pipe)

llm = load_llm_pipeline()

# Initialize FAISS store in session
if 'vectordb' not in st.session_state:
    st.session_state['vectordb'] = None

def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = []
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                txt = p.extract_text()
                if txt:
                    text.append(txt)
        return "\n".join(text)
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    return ""

def ingest_docs():
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No PDF/DOCX files found in the 'documents/' folder.")
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
    st.success(f"Ingested {len(files)} files into {len(all_texts)} text chunks.")

def query_docs(query: str, top_k: int = 5) -> str:
    vectordb = st.session_state['vectordb']
    if vectordb is None:
        return "⚠️ Please click **Ingest Documents** first."
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": top_k})
    )
    return qa.run(query)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Actions")
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Indexing documents..."):
        ingest_docs()

top_k = st.sidebar.slider("Top K passages", 1, 10, 5)

# ── Main ─────────────────────────────────────────────────────────────
query = st.text_input("🔍 Ask a question about your RFQ documents:", "")
if st.button("Search"):
    if not query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching..."):
            answer = query_docs(query, top_k)
            st.subheader("Answer")
            st.write(answer)

st.markdown("---")
st.markdown(
    """
    **Instructions:**  
    1. Place your `.pdf` and `.docx` files in the `documents/` folder.  
    2. Click **Ingest Documents** once to build the FAISS index.  
    3. Type your question and hit **Search** to retrieve an answer.
    """
)

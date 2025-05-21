import os
import streamlit as st
import pdfplumber
import docx
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA

# --- Configuration ---
DATA_DIR = "documents"
MODEL_PATH = "models/ggml-gpt4all-j-v1.3-groovy.bin"  # path to your GPT4All binary
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- Initialize components ---
# Embedding model (free, local)
embedder = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"}
)
# Text splitter for chunking
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# LLM (free, local GPT4All)
llm = GPT4All(
    model=MODEL_PATH,
    n_threads=4,
    verbose=False
)
# Placeholder for vector store
vectordb = None

# --- Helper functions ---
def extract_text(path: str) -> str:
    """Extract text from PDF or DOCX."""
    if path.lower().endswith(".pdf"):
        texts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    texts.append(txt)
        return "\n".join(texts)
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def ingest_docs():
    """Load docs, split, embed, and build FAISS index."""
    global vectordb
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No .pdf or .docx files found in 'documents/'")
        return

    texts, metadatas = [], []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        raw = extract_text(path)
        chunks = splitter.split_text(raw)
        texts.extend(chunks)
        metadatas.extend([{"source": fname}] * len(chunks))

    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas
    )
    st.success(f"Ingested {len(files)} files ({len(texts)} chunks) into FAISS.")


def query_docs(question: str, top_k: int = 5) -> str:
    """Run RetrievalQA: retrieve passages and generate answer."""
    if vectordb is None:
        return "Please ingest documents first."

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": top_k})
    )
    return qa_chain.run(question)

# --- Streamlit UI ---
st.set_page_config(page_title="Free RAG App", layout="wide")
st.title("📚 Free Local RAG with FAISS & GPT4All")

# Sidebar controls
st.sidebar.header("Actions")
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Ingesting documents into FAISS..."):
        ingest_docs()

# Sidebar for top_k
top_k = st.sidebar.slider("Top K passages", min_value=1, max_value=10, value=5)

# Main input
query = st.text_input("Ask a question about your documents:")
if st.button("Search"):
    if not query:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            answer = query_docs(query, top_k)
            st.subheader("Answer")
            st.write(answer)

# Footer instructions
st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Put your `.pdf` and `.docx` files in the `documents/` folder.
    2. Download a GPT4All binary and place it at `models/ggml-gpt4all-j-v1.3-groovy.bin`.
    3. Click **Ingest Documents** to build the FAISS index.
    4. Enter a question and click **Search** to see results powered by GPT4All.
    """
)

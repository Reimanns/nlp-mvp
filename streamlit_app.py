import os
import streamlit as st
import pdfplumber
import docx
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Constants
DATA_DIR = "documents"

# Initialize components
embedder = OpenAIEmbeddings()
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# FAISS vector store placeholder
vectordb = None

# Helper: extract text from files
def extract_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text.append(txt)
        return "\n".join(text)
    elif path.lower().endswith(".docx"):
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

# Ingest: build FAISS index in memory
def ingest_docs():
    global vectordb
    files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".pdf", ".docx"))]
    if not files:
        st.warning("No .pdf or .docx files found in 'documents/' folder.")
        return

    all_texts = []
    metadatas = []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        raw = extract_text(path)
        chunks = splitter.split_text(raw)
        all_texts.extend(chunks)
        metadatas.extend([{"source": fname}] * len(chunks))

    vectordb = FAISS.from_texts(
        texts=all_texts,
        embedding=embedder,
        metadatas=metadatas
    )
    st.success(f"Ingested {len(files)} files into FAISS (total chunks: {len(all_texts)})")

# Query: retrieve and answer
def query_docs(query: str, top_k: int = 5) -> str:
    if vectordb is None:
        return "Please click 'Ingest Documents' first."
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(model_name="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": top_k})
    )
    return qa.run(query)

# Streamlit UI
st.set_page_config(page_title="RAG Streamlit (FAISS)", layout="wide")
st.title("📚 RAG Document Search with FAISS")

# Sidebar controls
st.sidebar.header("Options")
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Ingesting documents into FAISS..."):
        ingest_docs()

top_k = st.sidebar.number_input("Top K results", min_value=1, max_value=20, value=5)

# Main query input
query = st.text_input("Ask a question about your documents:")
if st.button("Search") and query:
    with st.spinner("Searching..."):
        answer = query_docs(query, top_k=top_k)
        st.subheader("Answer")
        st.write(answer)

# Instructions
st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Place your `.pdf` and `.docx` files in the `documents/` folder.
    2. Click **Ingest Documents** to process and embed files using FAISS.
    3. Enter a question and hit **Search** to see LLM-powered answers.
    """
)

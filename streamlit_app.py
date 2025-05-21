import os
import streamlit as st
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pdfplumber
import docx

# Constants
DATA_DIR = "documents"
PERSIST_DIR = "vector_store"

# Initialize
embedder = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Load or create Chroma vectorstore
vectordb = Chroma(
    collection_name="streamlit_rag",
    embedding_function=embedder,
    persist_directory=PERSIST_DIR,
    client_settings=Settings(
        chroma_db_impl="duckdb+parquet",  # use DuckDB+Parquet store
        persist_directory=PERSIST_DIR
    )
)
# Build QA chain (lazy init after ingest)
llm = OpenAI(model_name="gpt-4o-mini", temperature=0)

# Helper functions

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_pdf(path: str) -> str:
    full_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def ingest_docs():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.pdf', '.docx'))]
    texts, metadatas = [], []
    for fname in files:
        path = os.path.join(DATA_DIR, fname)
        raw = extract_text_from_pdf(path) if fname.endswith('.pdf') else extract_text_from_docx(path)
        chunks = text_splitter.split_text(raw)
        texts.extend(chunks)
        metadatas.extend([{"source": fname}] * len(chunks))
    vectordb.add_texts(texts=texts, metadatas=metadatas)
    vectordb.persist()
    st.success(f"Ingested {len(files)} files with {len(texts)} chunks.")


def query_docs(query: str, top_k: int = 5):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": top_k})
    )
    return qa_chain.run(query)

# Streamlit UI
st.set_page_config(page_title="RAG Streamlit App", layout="wide")
st.title("📚 RAG Document Search")

# Sidebar controls
st.sidebar.header("Actions")
if st.sidebar.button("Ingest Documents"):
    with st.spinner("Ingesting documents..."):
        ingest_docs()

st.sidebar.text_input("Top K", value="5", key="top_k")

# Query input
query = st.text_input("Enter your question:", "")
if st.button("Search") and query:
    with st.spinner("Searching..."):
        try:
            k = int(st.session_state.top_k)
        except ValueError:
            k = 5
        answer = query_docs(query, top_k=k)
        st.subheader("Answer:")
        st.write(answer)

# Instructions
st.markdown("---")
st.markdown(
    """
    **Instructions:**
    1. Place your `.pdf` and `.docx` files in the `documents/` folder.
    2. Click **Ingest Documents** to extract, chunk, and embed your files.
    3. Type a question in the search box and click **Search**.
    4. Results will appear with relevant excerpts.
    """
)

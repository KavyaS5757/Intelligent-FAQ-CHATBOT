import streamlit as st
import fitz  # PyMuPDF
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import tempfile
import os
import pickle

st.set_page_config(page_title="📚 FAQ Chatbot", layout="centered")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "doc_uploaded" not in st.session_state:
    st.session_state.doc_uploaded = False

# Load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    return embedder, qa_model

embedder, qa_model = load_models()

# Read file
def read_file(file):
    if file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        return "\n".join([page.get_text() for page in doc])
    elif file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    else:
        return None

# Text splitter
def split_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

# Embeddings & FAISS
def get_embeddings(chunks):
    return embedder.encode(chunks, show_progress_bar=True)

def build_faiss_index(embeddings):
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index

# Get answer
def get_answer(query, chunks, index):
    query_embedding = embedder.encode([query])
    D, I = index.search(np.array(query_embedding), k=3)
    context = " ".join([chunks[i] for i in I[0]])
    return qa_model(question=query, context=context)["answer"]

# Sidebar
with st.sidebar:
    st.title("📁 Document Panel")
    uploaded_file = st.file_uploader("➕ Add Document", type=["pdf", "txt"])

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []

    if uploaded_file:
        with st.spinner("Processing document..."):
            text = read_file(uploaded_file)
            chunks = split_text(text)
            embeddings = get_embeddings(chunks)
            index = build_faiss_index(embeddings)

            st.session_state.chunks = chunks
            st.session_state.faiss_index = index
            st.session_state.doc_uploaded = True
            st.success("✅ Document processed!")

    # Show collapsible chat history
    st.markdown("---")
    st.subheader("💬 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        with st.expander(f"🧑‍💬 {q}", expanded=False):
            st.markdown(f"**🤖 Answer:** {a}")

# Main Chat UI
st.title("📚 Intelligent FAQ Chatbot")
st.markdown("Upload a `.pdf` or `.txt` and ask your questions!")

user_query = st.chat_input("Ask a question about the document")

if user_query and st.session_state.doc_uploaded:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            answer = get_answer(user_query, st.session_state.chunks, st.session_state.faiss_index)
            st.markdown(f"**Answer:** {answer}")
            st.session_state.chat_history.append((user_query, answer))
elif user_query:
    st.warning("⚠️ Please upload a document before asking questions.")

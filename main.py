import os
import streamlit as st
import faiss
import PyPDF2
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import requests

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Read and split PDF
def load_pdf(path):
    pdf_reader = PyPDF2.PdfReader(open(path, "rb"))
    texts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
    return texts

# Embed chunks and create FAISS index
def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, embeddings

# Get top-k similar chunks
def retrieve_context(query, chunks, embeddings, index, k=3):
    query_emb = embedder.encode([query])
    distances, indices = index.search(np.array(query_emb), k)
    return "\n".join([chunks[i] for i in indices[0]])

# Call Groq LLaMA-3
def call_groq(query, context):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are a helpful assistant. Use the context to answer the query.

Context:
{context}

Query: {query}
Answer:"""

    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama3-8b-8192",
        "temperature": 0.5,
        "max_tokens": 300,
        "top_p": 1
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Load PDF
pdf_chunks = load_pdf("Resume.pdf")
faiss_index, chunk_embeddings = create_faiss_index(pdf_chunks)

# ---------------- UI ----------------

# Hide sidebar and Streamlit style junk
st.set_page_config(page_title="Resume RAG", layout="centered")
hide_st_style = """
            <style>
            #MainMenu, header, footer {visibility: hidden;}
            .block-container { padding-top: 2rem; padding-bottom: 2rem; }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Clean Title
st.markdown("<h1 style='text-align: center;'>Resume Chatbot</h1>", unsafe_allow_html=True)

# Input Box
query = st.text_input("", placeholder="Ask something about the resume...")

if query:
    context = retrieve_context(query, pdf_chunks, chunk_embeddings, faiss_index)
    answer = call_groq(query, context)
    st.markdown("### Answer:")
    st.write(answer)

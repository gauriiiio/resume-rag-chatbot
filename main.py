import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import requests
os.environ["CHROMA_TELEMETRY"] = "FALSE"


load_dotenv()

#  Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#  Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#  Groq call function
def call_groq(prompt: str, context: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": f"You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

#  Text extraction
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

#  UI and workflow
st.title(" Resume Chatbot")
uploaded_file = st.file_uploader("Upload your Resume PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]

    #  Use temporary path for Chroma
    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory=tmpdir)
        query = st.text_input("Ask something about the resume")

        if query:
            results = vectorstore.similarity_search(query, k=3)
            context = "\n".join([res.page_content for res in results])
            answer = call_groq(query, context)
            st.write("### 💬 Answer:")
            st.write(answer)



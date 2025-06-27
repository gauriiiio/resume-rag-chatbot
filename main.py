import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume Chatbot", layout="wide")
st.title("📄 Resume RAG Chatbot (Groq + ChromaDB)")

# PDF Upload
pdf = st.file_uploader("Upload your resume PDF", type="pdf")

if pdf:
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Chunking
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Embedding & Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embeddings)

    # Question
    query = st.text_input("Ask a question about the resume:")

    if query:
        docs = vectorstore.similarity_search(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Call Groq LLaMA 3
        client = Groq(api_key=GROQ_API_KEY)

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an assistant helping answer questions from resume content."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
        )

        st.subheader("Answer:")
        st.write(response.choices[0].message.content)

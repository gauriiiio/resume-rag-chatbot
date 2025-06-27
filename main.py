import os
import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Resume RAG Chatbot", layout="centered")
st.title("📄 Resume RAG Chatbot")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF resume", type="pdf")

if uploaded_file:
    # Extract text
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Split into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Embed & store in Chroma
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

    # Set up Groq LLM
    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

    # RAG chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff"
    )

    query = st.text_input("Ask a question about the resume:")
    if query:
        result = qa.run(query)
        st.markdown("### Answer:")
        st.write(result)



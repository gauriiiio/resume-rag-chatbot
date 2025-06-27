import os
os.environ["CHROMA_TELEMETRY"] = "FALSE"

import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
import tempfile

# UI
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("📄 Resume RAG Chatbot")
st.write("Upload a resume (PDF), ask questions, and get insights.")

uploaded_file = st.file_uploader("Upload a Resume PDF", type="pdf")

query = st.text_input("Ask a question about the resume:")

if uploaded_file and query:
    with st.spinner("Processing..."):
        # Read PDF
        reader = PdfReader(uploaded_file)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        texts = text_splitter.split_text(raw_text)

        # Embed and index using ChromaDB
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        with tempfile.TemporaryDirectory() as tmpdir:
            vectorstore = Chroma.from_texts(texts, embedding=embeddings, persist_directory=tmpdir)

            # Set up Groq model
            llm = ChatGroq(
                groq_api_key=os.environ.get("GROQ_API_KEY"),
                model_name="mixtral-8x7b-32768"
            )

            chain = load_qa_chain(llm, chain_type="stuff")

            docs = vectorstore.similarity_search(query)

            # Answer
            answer = chain.run(input_documents=docs, question=query)

            st.success("Answer:")
            st.write(answer)




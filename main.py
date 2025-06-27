import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import tempfile

st.set_page_config(page_title="Resume RAG Chatbot 💬")
st.title("Resume Chatbot")

uploaded_file = st.file_uploader("Upload a resume (PDF)", type="pdf")

if uploaded_file:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    raw_text = ""
    for page in pdf_reader.pages:
        raw_text += page.extract_text() or ""

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Use temp directory for Chroma persistence
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(texts, embedding=embeddings, persist_directory=tmpdir)

        # Simple QA chain using HuggingFace model (open source)
        llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature":0.5, "max_length":512})
        chain = load_qa_chain(llm, chain_type="stuff")

        query = st.text_input("Ask something about the resume:")
        if query:
            docs = vectorstore.similarity_search(query)
            response = chain.run(input_documents=docs, question=query)
            st.write("🤖", response)





import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from phi.agent import Agent
from phi.llm.groq import Groq

# Load Groq API key from Streamlit secrets
groq_api_key = st.secrets["groq_api_key"]

# Set up LLM with Phi model
llm = Groq(
    api_key=groq_api_key,
    model="phi-2"
)

# Set up Streamlit UI
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title(" Resume Chatbot")

pdf = st.file_uploader("Upload a resume (PDF)", type="pdf")

if pdf:
    pdf_reader = PdfReader(pdf)
    raw_text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    # Split text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Set up embedding and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_texts(texts, embedding=embeddings, persist_directory=tmpdir)

        # Set up Assistant with vectorstore
        agent = Agent(llm=llm, vectorstore=vectorstore)

        query = st.text_input("Ask something about this resume:")
        if query:
            result = agent.run(query)
            st.markdown("### 💬 Response")
            st.write(result)

import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms.groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain

# Load Groq API key from Streamlit secrets
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="Resume RAG Chatbot", layout="wide")
st.title("🤖 Resume RAG Chatbot")
st.write("Ask any question about your uploaded resume!")

uploaded_file = st.file_uploader("📄 Upload your resume (PDF)", type="pdf")

query = st.text_input("🔍 Ask a question about the resume")

# Only proceed if both file and query are present
if uploaded_file and query:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Extract text from PDF
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embeddings,
            persist_directory=tmpdir
        )

        # Load retriever and chain
        docs = vectorstore.similarity_search(query)
        llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
        chain = load_qa_chain(llm, chain_type="stuff")

        # Get answer
        response = chain.run(input_documents=docs, question=query)

        # Display
        st.subheader("📤 Answer")
        st.write(response)

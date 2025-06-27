from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from phi.embedder.sentence_transformers import SentenceTransformerEmbedder
from dotenv import load_dotenv
import os

# Load your environment variables if needed
load_dotenv()

# Optional: if you store your API key in a custom file like secret_key.py
from secret_key import GROQ_API_KEY

# Set the API key so Groq knows what to use
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Define your vector DB connection and embedder
db_url = "postgresql+psycopg://postgres:ai@localhost:5532/postgres"
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")

vector_db = PgVector(
    table_name="resume",
    db_url=db_url,
    search_type=SearchType.hybrid,
    embedder=embedder
)

# Load your resume
knowledge_base = PDFKnowledgeBase(
    path="/Users/apple/Desktop/Gauri_Vats_Resume.pdf",
    vector_db=vector_db,
)

# Create DB table and insert data
knowledge_base.load(upsert=True)

# Create agent with Groq model
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)

# Ask a sample question
agent.print_response("What are Gauri's technical skills?")

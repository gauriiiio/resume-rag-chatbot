from phi.agent import Agent
from phi.model.groq import Groq
from phi.embedder.groq import GroqEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from secret_keys import GROQ_API_KEY

db_url = "postgresql+psycopg://postgres:ai@localhost:5532/postgres"

# ✅ Use a smaller, fast embedding model
embedder = GroqEmbedder(
    api_key=GROQ_API_KEY,
    model="llama-3-8b-8192"
)

knowledge_base = PDFKnowledgeBase(
    path="/Users/apple/Desktop/Gauri_Vats_Resume.pdf",
    embedder=embedder,
    vector_db=PgVector(
        table_name="resume",
        db_url=db_url,
        search_type=SearchType.similarity
    ),
)

# ✅ Use your original 70B LLM for chatting
agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
)

# Load your resume PDF into vector DB
knowledge_base.load(upsert=True)

# Ask questions
agent.print_response("What are Gauri's projects?")

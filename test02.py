from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from secret_key import GROQ_API_KEY

db_url = "postgresql+psycopg://postgres:ai@localhost:5532/postgres"

knowledge_base = PDFKnowledgeBase(
    path="/Users/apple/Desktop/Gauri_Vats_Resume.pdf",
    vector_db=PgVector(
        table_name="resume",
        db_url=db_url,
        search_type=SearchType.hybrid
    ),
)

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    knowledge=knowledge_base,
    add_context=True,
    search_knowledge=True,  # Set to True so it uses your PDF knowledge
    markdown=True,
)

knowledge_base.load(upsert=True)

agent.print_response("What are Gauri's skills?")

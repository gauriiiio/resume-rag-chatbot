from phi.agent import Agent
from phi.model.groq import Groq
from phi.knowledge.pdf import PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector, SearchType
from dotenv import load_dotenv
from secret_key import GROQ_API_KEY


load_dotenv()

db_url = "postgresql+psycopg://postgres:ai@localhost:5532/postgres"
knowledge_base = PDFKnowledgeBase(
    #urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    path="/Users/apple/Desktop/Gauri_Vats_Resume.pdf",
    vector_db=PgVector(table_name="resume", db_url=db_url, search_type=SearchType.hybrid),
)

#knowledge_base.load(upsert=True)

agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    knowledge=knowledge_base,
    add_context=True,
    
    search_knowledge=True,
    markdown=True,
)
agent.print_response("What are Gauri's skills?")

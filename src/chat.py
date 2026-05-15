import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from src.ingest import get_index

load_dotenv()

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def chat(question: str, top_k: int = 3) -> dict:
    """
    Take a user question, find the most relevant chunks
    from the vector DB, and generate a grounded answer.
    """
    index = get_index()

    # Query engine retrieves top_k most relevant chunks
    # then passes them to Llama 3 to generate the answer
    query_engine = index.as_query_engine(similarity_top_k=top_k)
    response = query_engine.query(question)

    # Extract the source documents used
    sources = []
    for node in response.source_nodes:
        sources.append({
            "source": node.metadata.get("source", "unknown"),
            "type": node.metadata.get("type", "unknown"),
            "score": round(node.score, 3) if node.score else None,
            "excerpt": node.text[:200] + "..."  # first 200 chars
        })

    return {
        "answer": str(response),
        "sources": sources,
        "chunks_used": len(sources)
    }

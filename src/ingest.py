import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi
from firecrawl import FirecrawlApp
import urllib.parse

load_dotenv()

# ---- Global Settings ----
# Tell LlamaIndex to use our free local embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# Tell LlamaIndex to use Groq for answer generation
Settings.llm = Groq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))

# ---- Vector Database Setup ----
def get_index():
    """
    Connect to (or create) our Chroma vector database
    and return a LlamaIndex VectorStoreIndex.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_documents")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context
    )
    return index

# ---- Source 1: PDF ----
def ingest_pdf(file_path: str) -> dict:
    """
    Read a PDF file, chunk it, embed it, store in Chroma.
    """
    from llama_index.core import SimpleDirectoryReader

    print(f"📄 Ingesting PDF: {file_path}")
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    index = get_index()
    for doc in documents:
        index.insert(doc)

    return {
        "status": "success",
        "source": "pdf",
        "file": file_path,
        "chunks_ingested": len(documents)
    }

# ---- Source 2: Web URL (via Firecrawl) ----
def ingest_url(url: str) -> dict:
    """
    Scrape a web page using Firecrawl (returns clean markdown),
    chunk it, embed it, store in Chroma.
    """
    print(f"🌐 Ingesting URL: {url}")
    app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    result = app.scrape_url(url, params={"formats": ["markdown"]})

    # Firecrawl returns clean markdown — perfect for LLMs
    content = result.get("markdown", "")
    if not content:
        return {"status": "error", "message": "No content scraped from URL"}

    document = Document(
        text=content,
        metadata={"source": url, "type": "webpage"}
    )

    index = get_index()
    index.insert(document)

    return {
        "status": "success",
        "source": "url",
        "url": url,
        "characters_ingested": len(content)
    }

# ---- Source 3: YouTube ----
def ingest_youtube(video_url: str) -> dict:
    """
    Pull the transcript from a YouTube video,
    chunk it, embed it, store in Chroma.
    """
    print(f"🎥 Ingesting YouTube: {video_url}")

    # Extract video ID from URL
    parsed = urllib.parse.urlparse(video_url)
    video_id = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]

    if not video_id:
        return {"status": "error", "message": "Could not extract video ID from URL"}

    # Fetch transcript
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    full_text = " ".join([entry["text"] for entry in transcript_list])

    document = Document(
        text=full_text,
        metadata={"source": video_url, "type": "youtube", "video_id": video_id}
    )

    index = get_index()
    index.insert(document)

    return {
        "status": "success",
        "source": "youtube",
        "video_id": video_id,
        "characters_ingested": len(full_text)
    }

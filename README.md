# 📚 Multi-Source RAG System

A Retrieval-Augmented Generation system that ingests PDFs, web pages, and YouTube transcripts into a vector database and exposes a chat API.

## Tech Stack
- **LlamaIndex** — RAG framework
- **Chroma** — local vector database
- **Nomic Embed** (via Ollama) — local embedding model
- **Groq (Llama 3)** — answer generation
- **FastAPI** — chat API endpoint
- **Firecrawl** — web scraping
- **YouTube Transcript API** — video transcripts

## Sources Supported
- PDFs
- Web URLs
- YouTube videos

## Setup
1. Clone repo
2. Copy `.env.example` to `.env` and fill in keys
3. `pip install -r requirements.txt`
4. `ollama pull nomic-embed-text`
5. `uvicorn src.main:app --reload`

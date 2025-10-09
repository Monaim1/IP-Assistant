# Docker Setup Guide for IP Assistant API

## Prerequisites

1. Docker and Docker Compose installed
2. `.env` file (optional). For local Ollama you do not need any API keys.

## Quick Start

### 1. Create your `.env` file

```bash
cp .env.example .env
```

### 2. Start services

```bash
# Lightweight (API + Chat; Ollama runs on host)
docker compose up -d

# With RAG stack (Milvus/MinIO/etcd)
docker compose --profile rag up -d
```

This will start:
- Always-on: **api** (FastAPI), **chat** (Streamlit UI)
- With `--profile rag`: **etcd**, **minio**, **milvus-standalone**

### 3. Check service health

```bash
# Check all services are running
docker compose ps

# Check API health
curl http://localhost:8000/health

# Check Ollama (running locally) is ready
curl http://localhost:11434/api/tags
```

### 4. Pull a model in Ollama (first-time, on host)

```bash
# Recommended small model
ollama pull qwen2.5:1.5b
```

### 5. Test the API

```bash
# Simple query without RAG (local LLM)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "solar panel technology", "use_rag": false, "model": "qwen2.5:1.5b"}'

# Streamed response (tokens as they generate)
curl -N -X POST "http://localhost:8000/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "solar panel technology", "use_rag": false, "model": "qwen2.5:1.5b"}'

# Query with RAG (requires data in Milvus)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "search all patent involving solar panel technology", "use_rag": true, "top_k": 5}'

# Search only (no LLM)
curl -X POST "http://localhost:8000/search?query=solar%20panel&top_k=3"
```

## API Endpoints

- `GET /` - API information and status
- `GET /health` - Health check
- `POST /query` - Query with RAG pipeline
- `POST /search` - Search patents without LLM

## Stopping Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes all data)
docker compose down -v
```

## Rebuilding the API

After code changes:

```bash
docker compose up -d --build api
```

## Ingestion
```bash
docker compose exec api uv run python -m ip_assistant.ingestion
```

## testing opensearch & milvus Databases

```bash
##opensearch

curl -X GET "localhost:9200/_cat/indices?v"

curl -X GET "localhost:9200/ip_chunks_bm25/_search?size=5&pretty"


## milvus
curl -X POST "http://localhost:8000/search?top_k=3&query=photovoltaic%20solar%20panels"
```
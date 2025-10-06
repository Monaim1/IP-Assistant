# Docker Setup Guide for IP Assistant API

## Prerequisites

1. Docker and Docker Compose installed
2. `.env` file with your `OPENROUTER_API_KEY`

## Quick Start

### 1. Create your `.env` file

```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

### 2. Start all services

```bash
docker-compose up -d
```

This will start:
- **etcd**: Metadata storage for Milvus
- **minio**: Object storage for Milvus
- **milvus-standalone**: Vector database
- **api**: Your RAG API service

### 3. Check service health

```bash
# Check all services are running
docker-compose ps

# Check API health
curl http://localhost:8000/health
```

### 4. Test the API

```bash
# Simple query without RAG
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "solar panel technology", "use_rag": false}'

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
docker-compose down

# Stop and remove volumes (⚠️ deletes all data)
docker-compose down -v
```

## Rebuilding the API

After code changes:

```bash
docker-compose up -d --build api
```

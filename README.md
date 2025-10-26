# IP Assistant System

## Overview
A Retrieval-Augmented Generation (RAG) system for patent analysis, implementing contextual embeddings and hybrid retrieval methods.

## Key Features
- **Contextual Embeddings**: Enrich chunks with patent-specific context
- **Hybrid Retrieval**: Combines vector similarity (Milvus) and lexical BM25 (Elasticsearch/OpenSearch)
- **Re-ranking**: Implements cross-encoder or LLM-based scoring for improved relevance


## Getting Started

### 1. Prerequisites
- Docker and Docker Compose
- OpenRouter API key or ollama

### 2. Set Up Environment
```bash
# Clone the repository
git clone repo_url
cd IP-Assistant

# Create .env file with your OpenRouter API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

```
# Start all services (Milvus, etcd, MinIO, API)
docker-compose up -d
```

```
## Run the ingestion pipeline, it's fixed at 1000 patents, change as desired (using the API service container)
docker-compose exec api uv run python -m ip_assistant.ingestion
```

### Access the API

```
# Test search endpoint
curl "http://localhost:8000/search?query=solar+panel&top_k=3"

# Test full RAG pipeline
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "search all patent involving solar panel technology", "use_rag": true}'
```


## System Architecture

### 1. Data Processing
#### Data Ingestion & Normalization
- Source: HUPD dataset (abstract, claims, summary, full_description, metadata fields)
- Process:
  - Normalize records into structured JSON
  - Store metadata (application number, filing date, CPC labels, etc.)

#### Chunking Strategy
- Chunk by semantic sections (abstract, claims, background, summary, full_description)
- Chunk size: 300–500 tokens with 50-token overlap
- Preserve section and patent metadata with each chunk

### 2. Contextualization
- **Method**: Anthropic's contextual retrieval approach
- **Process**:
  - Input: Whole patent + chunk to Claude/LLM
  - Output: 50–100 token contextual summary
  - Format: Prepend context to chunk text as "contextualized chunk"

**Example**:
```
Context: This chunk is from Patent 20160012345, Abstract, about solar panel efficiency improvements.  
Chunk: "The system improves photon capture by embedding nanostructures in the substrate layer."
```

### 3. Indexing
#### Vector Index (Milvus)
- Stores embeddings of contextualized chunks
  - sentence-transformers

#### BM25 Index (ElasticSearch/OpenSearch)
- Indexes contextualized chunk text
- Enables exact term matching (e.g., "US20160234A1")

### 4. Retrieval Pipeline (Current Implementation)
- Process user query
- Vector search using Milvus
- Return top-K chunks (e.g., 5–10)

Planned Enhancements:
- Add BM25 keyword search (Elasticsearch/OpenSearch)
- Implement result re-ranking with cross-encoders
- Add lightweight LLM relevance scoring

### 5. Answer Generation
- **Input**:
  - User query
  - Retrieved contextualized chunks (as citations)
- **Process**:
  - Construct prompt with query and context
  - Send to LLM (Claude/GPT/Openrouter)
  - Generate structured output (answer + cited patent IDs/chunks)

## Observability (MLflow)

- Basic metrics and artifacts are logged per request using MLflow:
  - Metrics: `retrieval_latency_ms`, `llm_latency_ms`, `request_latency_ms`, `chunks_retrieved`, `tokens_used`
  - Artifacts: `query.txt`, `prompt.txt`, `context.txt` (if any), `output.txt`, `retrieved.json` (chunk summary)
- Configuration:
  - By default logs to local file store at `./mlruns` (set via `MLFLOW_TRACKING_URI=file:./mlruns`).
  - To use an existing tracking server (e.g., with SQLite db `mlruns.db`), set `MLFLOW_TRACKING_URI` accordingly, e.g. `sqlite:///mlruns.db` (requires running `mlflow server`).
  - Change experiment name with `MLFLOW_EXPERIMENT` (default: `IP-Assistant`).

View runs locally:
```
mlflow ui --backend-store-uri file:./mlruns --port 5000
# or if you run a server with SQLite
mlflow ui --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts --port 5000
```

Docker note: If the API runs in Docker and your MLflow server runs on the host, use `MLFLOW_TRACKING_URI=http://host.docker.internal:5000` (not `http://localhost:5000`). If the URI is unreachable, the API auto‑falls back to `file:./mlruns` with a short timeout.

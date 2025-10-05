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
- OpenRouter API key

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

# Verify all services are running
docker-compose ps
```

# Run the ingestion pipeline (using the API service container)
docker-compose exec api python ingestion.py --input-dir /path/to/patents

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
  - Send to LLM (Claude/GPT)
  - Generate structured output (answer + cited patent IDs/chunks)

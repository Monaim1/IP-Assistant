# Patent RAG System

## Overview
A Retrieval-Augmented Generation (RAG) system for patent analysis, implementing contextual embeddings and hybrid retrieval methods.

## Key Features
- **Contextual Embeddings**: Enrich chunks with patent-specific context
- **Hybrid Retrieval**: Combines vector similarity (Milvus) and lexical BM25 (Elasticsearch/OpenSearch)
- **Re-ranking**: Implements cross-encoder or LLM-based scoring for improved relevance

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
- Supported models:
  - sentence-transformers
  - OpenAI/Anthropic embeddings

#### BM25 Index (ElasticSearch/OpenSearch)
- Indexes contextualized chunk text
- Enables exact term matching (e.g., "US20160234A1")

### 4. Retrieval Pipeline
1. Process user query
2. Execute parallel searches:
   - Vector search (Milvus)
   - Keyword search (BM25 via Elasticsearch/OpenSearch)
3. Merge and deduplicate results
4. Re-rank using:
   - Cross-encoder (e.g., ms-marco-MiniLM-L-6-v2) or
   - Lightweight LLM relevance scoring
5. Return top-K chunks (e.g., 5–10)

### 5. Answer Generation
- **Input**:
  - User query
  - Retrieved contextualized chunks (as citations)
- **Process**:
  - Construct prompt with query and context
  - Send to LLM (Claude/GPT)
  - Generate structured output (answer + cited patent IDs/chunks)

## Monitoring & Observability
- **Performance Metrics**:
  - Query latency
  - Recall @K
  - Re-ranker confidence scores
- **Cost Tracking**:
  - Embedding costs
  - Contextualization costs
- **System Health**:
  - Milvus cluster status
  - Elasticsearch/OpenSearch cluster health
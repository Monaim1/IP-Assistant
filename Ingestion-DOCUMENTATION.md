# IP Assistant - Documentation

## Overview

The IP Assistant is a Retrieval-Augmented Generation (RAG) system for intellectual property (patent) documents. It ingests patent data, chunks text content, generates embeddings, and stores everything in a Milvus vector database for semantic search and retrieval.

## Architecture

```
┌─────────────────┐
│  Patent JSON    │
│     Files       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │  ← Filters relevant fields
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Chunking   │  ← Splits into 1200-char chunks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  ← sentence-transformers/all-MiniLM-L6-v2
│   Generation    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Milvus Vector  │  ← Stores embeddings + metadata
│    Database     │
└─────────────────┘
```

## System Components

### 1. Docker Services

The system runs three Docker containers:

- **etcd**: Distributed key-value store for Milvus metadata
- **minio**: Object storage for Milvus data persistence
- **milvus-standalone**: Vector database for embeddings and metadata

### 2. Data Pipeline

#### Input Data Format
Patent documents in JSON format with the following fields:

**Identifiers:**
- `publication_number` - Primary external ID
- `application_number` - Application tracking number
- `patent_number` - Patent number (if granted)

**Dates:**
- `date_published` - Publication date
- `filing_date` - Filing date
- `patent_issue_date` - Issue date (nullable)
- `abandon_date` - Abandonment date (nullable)

**Classification:**
- `decision` - Status (granted/pending/withdrawn)
- `main_cpc_label` - Main CPC classification
- `main_ipcr_label` - Main IPCR classification

**Text Content:**
- `title` - Patent title
- `abstract` - Patent abstract
- `summary` - Patent summary

you can add other fields to the `RELEVANT_FIELDS` list in the `ingestion.py` file

#### Processing Pipeline

1. **Load Data** (`get_IP_data()`)
   - Reads JSON files from `RawData/2018/`
   - Filters to only relevant fields
   - Returns list of filtered documents

2. **Extract Metadata** (`process_document()`)
   - Extracts identifiers, dates, and classifications
   - Converts dates to epoch timestamps
   - Normalizes string fields

3. **Chunk Text** (`chunk_text()`)
   - Splits text into 1200-character chunks
   - Uses 150-character overlap for context preservation
   - Processes: title, abstract, summary

4. **Generate Embeddings** (`compute_embeddings()`)
   - Uses `sentence-transformers/all-MiniLM-L6-v2`
   - Generates 384-dimensional vectors
   - Normalizes embeddings for cosine similarity

5. **Batch Insert** (`flush_batch()`)
   - Inserts data in batches of 512 chunks
   - Column-major format (Milvus requirement)
   - Auto-generates primary keys

## Configuration

### Environment Variables

```bash
# OpenRouter API Key (for LLM queries)
OPENROUTER_API_KEY=your_key_here
```

### Pipeline Configuration

Located in `ingestion.py`:

```python
# Milvus Connection
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION = "ip_chunks"

# Embedding Model
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIM = 384
METRIC = "COSINE"

# Chunking Parameters
CHUNK_SIZE = 1200        # characters
CHUNK_OVERLAP = 150      # characters
BATCH_SIZE = 512         # chunks per batch

# Text Fields to Process
TEXT_FIELDS = ["title", "abstract", "summary"]
```

## Milvus Schema

### Collection: `ip_chunks`

| Field Name | Type | Description |
|------------|------|-------------|
| `pk` | INT64 | Primary key (auto-generated) |
| `application_number` | VARCHAR(128) | Application number |
| `publication_number` | VARCHAR(128) | Publication number |
| `patent_number` | VARCHAR(128) | Patent number |
| `section` | VARCHAR(64) | Source field (title/abstract/summary) |
| `decision` | VARCHAR(64) | Patent status |
| `main_cpc_label` | VARCHAR(128) | CPC classification |
| `main_ipcr_label` | VARCHAR(128) | IPCR classification |
| `date_published_ts` | INT64 | Publication date (epoch) |
| `filing_date_ts` | INT64 | Filing date (epoch) |
| `patent_issue_date_ts` | INT64 | Issue date (epoch) |
| `abandon_date_ts` | INT64 | Abandon date (epoch) |
| `text` | VARCHAR(8192) | Text chunk content |
| `embedding` | FLOAT_VECTOR(384) | Embedding vector |

### Indexes

**Vector Index:**
- Type: HNSW (Hierarchical Navigable Small World)
- Metric: COSINE
- Parameters: M=16, efConstruction=200

**Scalar Indexes:**
- **Trie** (for VARCHAR fields): `section`, `decision`, `main_cpc_label`
- **STL_SORT** (for INT64 fields): `date_published_ts`, `filing_date_ts`

## Installation & Setup

### Prerequisites

- Docker & Docker Compose
- Python 3.13+
- uv (Python package manager)

### Step 1: Start Docker Services

```bash
cd /Users/mounselam/Desktop/IP-Assistant
docker compose up -d
```

Verify services are running:
```bash
docker compose ps
```

Expected output:
```
NAME                   STATUS
ip-assistant-etcd-1    Up (healthy)
ip-assistant-minio-1   Up (healthy)
milvus-standalone      Up (healthy)
```

### Step 2: Install Python Dependencies

```bash
uv sync
```

### Step 3: Prepare Data

Place patent JSON files in:
```
RawData/2018/*.json
```
the directory should have the json HUPD files

### Step 4: Run Ingestion

```bash
uv run ingestion.py
```

## Usage


### Custom Configuration

```python
# Modify configuration before importing
import ingestion

# Change batch size
ingestion.BATCH_SIZE = 256

# Change text fields
ingestion.TEXT_FIELDS = ["title", "abstract", "summary", "claims"]

# Run ingestion
ingestion.ingest_patents(ip_limit=100)
```

### Querying the Collection

```python
from pymilvus import Collection, connections

# Connect to Milvus
connections.connect("default", host="127.0.0.1", port="19530")
coll = Collection("ip_chunks")
coll.load()

# Search by vector
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query = "methods for packaging materials"
query_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

results = coll.search(
    data=[query_vec],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    output_fields=["text", "publication_number", "section", "decision"]
)

for hits in results:
    for hit in hits:
        print(f"Score: {hit.score}")
        print(f"Text: {hit.entity.get('text')[:200]}...")
        print(f"Patent: {hit.entity.get('publication_number')}")
        print("---")
```

### Filtering by Metadata

```python
# Search with filters
results = coll.search(
    data=[query_vec],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 64}},
    limit=5,
    expr='decision == "GRANTED" and date_published_ts > 1514764800',  # After 2018-01-01
    output_fields=["text", "publication_number", "main_cpc_label"]
)
```

## Utility Functions

### LLM Integration (`utils.py`)

```python
from utils import get_LLM_response

# Get AI response for a query
response = get_LLM_response(
    prompt="Summarize this patent abstract: ...",
    model="moonshotai/kimi-k2",
    max_tokens=2000,
    temperature=0.7
)
```

## Performance Considerations

### Ingestion Speed

- **20 patents**: ~1 second
- **100 patents**: ~5 seconds
- **1000 patents**: ~50 seconds

Bottlenecks:
1. Embedding generation (GPU recommended)
2. Disk I/O for Milvus writes

### Search Performance

- **Top-5 search**: <50ms
- **Top-100 search**: <200ms
- **Filtered search**: <100ms (with proper indexes)

### Scaling Recommendations

**For Large Datasets (>10K patents):**
1. Increase `BATCH_SIZE` to 1024
2. Use GPU for embedding generation
3. Consider Milvus cluster mode
4. Add connection pooling

**For Production:**
1. Use environment variables for all config
2. Add retry logic for Milvus operations
3. Implement incremental ingestion
4. Add monitoring and logging

## Troubleshooting

### Milvus Connection Failed

```
MilvusException: Fail connecting to server on 127.0.0.1:19530
```

**Solution:**
```bash
# Check if containers are running
docker compose ps

# Restart services
docker compose down
docker compose up -d

# Wait for health checks
sleep 30
```

### No Entities After Ingestion

**Cause:** Missing `coll.flush()` call

**Solution:** The pipeline now includes automatic flushing. If you see this issue:
```python
coll.flush()  # Persist data
coll.load()   # Reload for search
```

### Out of Memory

**Cause:** Large batch size or too many embeddings in memory

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 128

# Or process in smaller chunks
ingest_patents(ip_limit=10)
```

### Index Creation Failed

**Cause:** Wrong index type for field type

**Solution:**
- Use `Trie` for VARCHAR fields
- Use `STL_SORT` for INT64 fields
- Use `HNSW` or `IVF_FLAT` for FLOAT_VECTOR fields


#### `to_epoch(date_str: str) -> int`
Converts date string to epoch timestamp.

**Parameters:**
- `date_str`: Date in format YYYY-MM-DD

**Returns:**
- Unix timestamp (seconds since epoch)

---

#### `process_document(data: Dict) -> None`
Processes a single patent document and adds chunks to batch.

**Parameters:**
- `data`: Patent document dictionary

---

#### `ingest_patents(ip_limit: int = 20) -> None`
Main ingestion pipeline.

**Parameters:**
- `ip_limit`: Number of patents to ingest

---

#### `init_milvus_collection() -> Collection`
Initializes or loads existing Milvus collection.

**Returns:**
- Milvus Collection object


## Future Enhancements

### Planned Features
1. **Incremental Ingestion**: Only process new/updated patents
2. **Multi-language Support**: Handle patents in different languages
3. **Advanced Chunking**: Semantic chunking based on document structure
4. **Hybrid Search**: Combine vector search with keyword search
5. **Query Interface**: Web UI for searching patents
6. **Analytics Dashboard**: Visualize patent trends and classifications

### Integration Ideas
1. **RAG Pipeline**: Combine with LLM for question answering
2. **Citation Analysis**: Link related patents
3. **Prior Art Search**: Find similar existing patents
4. **Patent Summarization**: Auto-generate summaries
5. **Classification Prediction**: Predict CPC/IPCR labels

## Support & Contributing

For issues or questions:
1. Check this documentation
2. Review error logs in Docker containers
3. Verify Milvus collection status

## License

[Add your license information here]

## Changelog

### v1.0.0 (2025-10-04)
- Initial release
- Basic ingestion pipeline
- Milvus integration
- Docker Compose setup
- Semantic search capability

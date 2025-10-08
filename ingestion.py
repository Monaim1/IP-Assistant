import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
import orjson
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from pymilvus import (
    connections, FieldSchema, CollectionSchema, DataType, Collection,
    utility
)

# Config
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION = "ip_chunks"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d
EMB_DIM = 384
DATA_DIR = Path("./ip_json")                # folder with your JSON files
BATCH_SIZE = 512
CHUNK_SIZE = 256         
CHUNK_OVERLAP = 50       
METRIC = "COSINE"        # Similarity metric

# Which long text fields to chunk/ingest as retrievable content:
TEXT_FIELDS = ["abstract", "summary"] 

# Fields to extract from raw JSON data
RELEVANT_FIELDS = [
    # Identifiers & Linking
    "publication_number",
    "application_number",
    "patent_number",
    # Dates (as epoch ints)
    "date_published",
    "filing_date",
    "patent_issue_date",
    "abandon_date",
    # Status & Classes
    "decision",
    "main_cpc_label",
    "main_ipcr_label",
    # Retrievable Text
    "title",
    "abstract",
    "summary",
]
def get_IP_data(ip_limit=20):
    """Load and filter IP data from JSON files."""
    ip_files = [json.load(open("RawData/2018/" + file)) for file in os.listdir(r"RawData/2018")[:ip_limit]] 
    ip_files = [{key: value for key, value in file.items() if key in RELEVANT_FIELDS} for file in ip_files]
    return ip_files

def to_epoch(date_str: str) -> int:
    """
    Convert 'YYYY-MM-DD' (or close) to epoch seconds; returns 0 if missing.
    """
    if not date_str:
        return 0
    ds = date_str.strip()
    if not ds:
        return 0
    # Try common formats
    
    fmts = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m", "%Y"]
    for f in fmts:
        try:
            dt = datetime.strptime(ds, f)
            return int(dt.timestamp())
        except ValueError:
            continue
    try:
        ds2 = "".join(ch for ch in ds if ch.isdigit() or ch in "-/")
        for f in ["%Y-%m-%d", "%Y/%m/%d", "%Y-%m"]:
            try:
                dt = datetime.strptime(ds2, f)
                return int(dt.timestamp())
            except ValueError:
                pass
    except Exception:
        pass
    return 0

def chunk_text(s: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks based on tokens."""
    if not s:
        return []
    s = s.strip()
    if len(s) <= size:
        return [s]
    
    # Encode text to token IDs
    token_ids = tokenizer.encode(s, add_special_tokens=False)
    
    chunks = []
    i = 0
    while i < len(token_ids):
        chunk_ids = token_ids[i:i+size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        i += size - overlap
    
    return chunks

def select_scalar(v, maxlen=None) -> str:
    """Convert value to string, handling None, lists, and dicts."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        s = orjson.dumps(v).decode("utf-8")
        return s[:maxlen] if maxlen else s
    s = str(v)
    return s[:maxlen] if maxlen else s


# MILVUS SETUP

def create_milvus_schema() -> List[FieldSchema]:
    """Defining Milvus collection schema."""
    return [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="application_number", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="publication_number", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="patent_number", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="decision", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="main_cpc_label", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="main_ipcr_label", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="date_published_ts", dtype=DataType.INT64),
        FieldSchema(name="filing_date_ts", dtype=DataType.INT64),
        FieldSchema(name="patent_issue_date_ts", dtype=DataType.INT64),
        FieldSchema(name="abandon_date_ts", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMB_DIM),
    ]


def init_milvus_collection() -> Collection:
    """Initializing or loading existing Milvus collection."""
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    if utility.has_collection(COLLECTION):
        return Collection(COLLECTION)
    
    fields = create_milvus_schema()
    schema = CollectionSchema(fields, description="Chunked IP documents (RAG-ready)")
    coll = Collection(COLLECTION, schema=schema)
    
    # Create vector index
    coll.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": METRIC,
            "params": {"M": 16, "efConstruction": 200}
        }
    )
    
    # Create scalar indexes for filtering
    # Trie index for VARCHAR fields
    for field in ["section", "decision", "main_cpc_label"]:
        coll.create_index(field, index_params={"index_type": "Trie"})
    # STL_SORT for INT64 date fields
    for field in ["date_published_ts", "filing_date_ts"]:
        coll.create_index(field, index_params={"index_type": "STL_SORT"})
    
    coll.load()
    return coll

# INGESTION PIPELINE

coll = None
model = None
tokenizer = None
rows = None

def compute_embeddings():
    """Compute embeddings for all None values in rows['embedding']."""
    idxs = [i for i, v in enumerate(rows["embedding"]) if v is None]
    if not idxs:
        return
    embeds = model.encode([rows["text"][i] for i in idxs], normalize_embeddings=True)
    for j, emb in zip(idxs, embeds):
        rows["embedding"][j] = emb.astype(np.float32).tolist()


def flush_batch():
    """Insert accumulated rows into Milvus and clear buffers."""
    if not rows["text"]:
        return
    
    # Compute any pending embeddings
    compute_embeddings()
    
    data_to_insert = [rows[field.name] for field in coll.schema.fields if field.name != "pk"]
    coll.insert(data_to_insert)
    
    # Clear all buffers
    for k in rows:
        rows[k].clear()


def process_document(data: Dict) -> int:
    """Process a single patent document and add chunks to batch.
    
    Returns:
        Number of chunks created from this document.
    """
    # Extract metadata
    metadata = {
        "application_number": select_scalar(data.get("application_number"), 128),
        "publication_number": select_scalar(data.get("publication_number"), 128),
        "patent_number": select_scalar(data.get("patent_number"), 128),
        "decision": select_scalar(data.get("decision"), 64),
        "main_cpc_label": select_scalar(data.get("main_cpc_label"), 128),
        "main_ipcr_label": select_scalar(data.get("main_ipcr_label"), 128),
        "date_published_ts": to_epoch(data.get("date_published", "")),
        "filing_date_ts": to_epoch(data.get("filing_date", "")),
        "patent_issue_date_ts": to_epoch(data.get("patent_issue_date", "")),
        "abandon_date_ts": to_epoch(data.get("abandon_date", "")),
    }
    
    chunks_created = 0
    # Chunk text fields and add to batch
    for field in TEXT_FIELDS:
        text_value = data.get(field, "")
        for chunk in chunk_text(text_value):
            # Add metadata for this chunk
            for key, value in metadata.items():
                rows[key].append(value)
            
            rows["section"].append(field)
            rows["text"].append(chunk)
            rows["embedding"].append(None)  # Compute later in batch
            chunks_created += 1
            
            # Flush if batch is full
            if len(rows["text"]) >= BATCH_SIZE:
                flush_batch()
    
    return chunks_created


def clear_collection() -> None:
    connections.connect("default", host="127.0.0.1", port="19530")
    
    if utility.has_collection(COLLECTION):
        print(f"  Clearing existing collection '{COLLECTION}'...")
        utility.drop_collection(COLLECTION)
        print(f"  ✓ Collection '{COLLECTION}' cleared successfully")


def ingest_patents(ip_limit: int = 20) -> None:
    """Main ingestion pipeline."""
    global coll, model, tokenizer, rows
    
    clear_collection()
    coll = init_milvus_collection()
    
    # Initialize models
    model = SentenceTransformer(EMB_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL_NAME)
    
    # Initialize data buffers
    rows = { field.name: [] for field in coll.schema.fields}
    
    ip_files = get_IP_data(ip_limit)
    
    total_chunks = 0
    for data in tqdm(ip_files, desc="Ingesting patents"):
        try:
            chunks_added = process_document(data)
            total_chunks += chunks_added
        except Exception as e:
            print(f"Failed to process document: {e}")
            continue
    
    # Final flush for remaining data
    if rows["text"]:
        flush_batch()
    
    # Flush to persist data to storage
    coll.flush()
    
    # Load collection for search
    coll.load()
    print(f"✓ Ingestion complete!")
    print(f"  • Processed {total_chunks} text chunks from {len(ip_files)} patents")
    print(f"  • Collection '{COLLECTION}' now has {coll.num_entities} entities")


if __name__ == "__main__":
    ingest_patents(ip_limit=1000)

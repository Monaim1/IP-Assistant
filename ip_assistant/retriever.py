from typing import List, Dict, Optional, Tuple
import os
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from math import isfinite

# Optional OpenSearch for BM25
try:
    from opensearchpy import OpenSearch  # type: ignore
except Exception:  # pragma: no cover
    OpenSearch = None  # type: ignore

class PatentRetriever:
    """
    A simple retriever for searching patent documents using Milvus vector database.
    """
    
    def __init__(
        self,
        collection_name: str = "ip_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "127.0.0.1",
        port: str = "19530",
        # Hybrid retrieval settings
        bm25_enabled: Optional[bool] = None,
        opensearch_host: Optional[str] = None,
        opensearch_user: Optional[str] = None,
        opensearch_password: Optional[str] = None,
        opensearch_index: Optional[str] = None,
        hybrid_alpha: Optional[float] = None,
    ):
        self.model = SentenceTransformer(model_name)
        self.host = host #  Milvus server host
        self.port = port # Milvus server port
        self.collection_name = collection_name
        self.collection = None
        # Hybrid config via env/defaults
        self.bm25_enabled = (
            os.getenv("BM25_ENABLED").lower() in {"1", "true", "yes"}
            if os.getenv("BM25_ENABLED") is not None
            else True if bm25_enabled is None else bm25_enabled
        )
        self.os_host = (
            opensearch_host
            or os.getenv("OPENSEARCH_HOST")
            or os.getenv("ELASTICSEARCH_HOST")
            or "http://localhost:9200"
        )
        self.os_user = opensearch_user or os.getenv("OPENSEARCH_USER", "")
        self.os_pass = opensearch_password or os.getenv("OPENSEARCH_PASSWORD", "")
        self.os_index = opensearch_index or os.getenv("OPENSEARCH_INDEX", "ip_chunks_bm25")
        try:
            self.hybrid_alpha = float(hybrid_alpha if hybrid_alpha is not None else os.getenv("HYBRID_ALPHA", 0.6))
        except Exception:
            self.hybrid_alpha = 0.6
        self.os_client = None
        self._connect()
        self._connect_opensearch()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def _connect_opensearch(self):
        if not self.bm25_enabled or OpenSearch is None:
            self.os_client = None
            return
        try:
            auth = (self.os_user, self.os_pass) if (self.os_user or self.os_pass) else None
            self.os_client = OpenSearch(self.os_host, http_auth=auth)  # type: ignore
            # Optional ping to verify
            try:
                if not self.os_client.ping():
                    print("⚠ OpenSearch ping failed; BM25 disabled")
                    self.os_client = None
            except Exception:
                pass
        except Exception as e:
            print(f"⚠ Could not init OpenSearch: {e}")
            self.os_client = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict]:
        if return_fields is None:
            return_fields = ["text", "publication_number", "section", "decision"]

        # Dense/vector search (Milvus)
        vec_results = self._vector_search(query, top_k=top_k, return_fields=return_fields, min_score=min_score)

        # Lexical/BM25 search (OpenSearch)
        bm25_results = self._bm25_search(query, top_k=top_k*2, return_fields=return_fields) if self.os_client else []

        # If BM25 not available, return dense results only
        if not bm25_results:
            return vec_results[:top_k]

        # Hybrid combine
        combined = self._combine_results(vec_results, bm25_results, top_k)
        return combined

    def _vector_search(self, query: str, top_k: int, return_fields: List[str], min_score: float) -> List[Dict]:
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=return_fields,
        )
        default_fields = {
            "publication_number": None,
            "text": "",
            "section": None,
            "decision": None,
        }
        out = []
        for hits in results:
            for hit in hits:
                if hit.score < min_score:
                    continue
                row = {**default_fields}
                for field in return_fields:
                    row[field] = hit.entity.get(field, default_fields.get(field))
                row["score"] = float(hit.score)
                row["_v_score"] = row["score"]
                out.append(row)
        return out

    def _bm25_search(self, query: str, top_k: int, return_fields: List[str]) -> List[Dict]:
        try:
            body = {
                "size": top_k,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text"],
                        "type": "best_fields"
                    }
                },
                "_source": return_fields,
            }
            res = self.os_client.search(index=self.os_index, body=body)  # type: ignore
            out = []
            hits = res.get("hits", {}).get("hits", [])
            for h in hits:
                src = h.get("_source", {})
                row = {k: src.get(k) for k in return_fields}
                row["score"] = float(h.get("_score", 0.0) or 0.0)
                row["_bm25_score"] = row["score"]
                out.append(row)
            return out
        except Exception:
            return []

    def _combine_results(self, vec: List[Dict], bm25: List[Dict], top_k: int) -> List[Dict]:
        # Helper to normalize a list of scores
        def normalize(scores: List[float]) -> Tuple[float, float]:
            if not scores:
                return (0.0, 0.0)
            mn = min(scores)
            mx = max(scores)
            return (mn, mx)

        # Build maps by unique key (chunk-level)
        def key_of(d: Dict) -> str:
            return f"{d.get('publication_number')}|{d.get('section')}|{d.get('text')}"

        m: Dict[str, Dict] = {}
        for r in vec:
            m[key_of(r)] = {**r}
        for r in bm25:
            k = key_of(r)
            if k in m:
                m[k].update(r)
            else:
                m[k] = {**r}

        # Normalize scores
        v_scores = [d.get("_v_score", 0.0) for d in m.values() if isfinite(d.get("_v_score", 0.0))]
        b_scores = [d.get("_bm25_score", 0.0) for d in m.values() if isfinite(d.get("_bm25_score", 0.0))]
        v_mn, v_mx = normalize(v_scores)
        b_mn, b_mx = normalize(b_scores)

        out: List[Dict] = []
        for d in m.values():
            v = d.get("_v_score", None)
            b = d.get("_bm25_score", None)
            v_norm = ((v - v_mn) / (v_mx - v_mn)) if (v is not None and v_mx > v_mn) else (1.0 if v is not None else 0.0)
            b_norm = ((b - b_mn) / (b_mx - b_mn)) if (b is not None and b_mx > b_mn) else (1.0 if b is not None else 0.0)
            # Weighted combine; if one side missing, use the other
            if v is not None and b is not None:
                final = self.hybrid_alpha * v_norm + (1.0 - self.hybrid_alpha) * b_norm
            elif v is not None:
                final = v_norm
            else:
                final = b_norm
            d2 = {k: v for k, v in d.items() if not k.startswith("_")}
            d2["score"] = float(final)
            out.append(d2)

        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return out[:top_k]

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'collection') and self.collection:
            try:
                self.collection.release()
                connections.disconnect("default")
            except:
                pass

# Example usage
if __name__ == "__main__":
    retriever = PatentRetriever()
    results = retriever.search("solar panel tech", top_k=2)
    
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Patent: {result['publication_number']}")
        print(f"Score: {result['score']:.3f}")
        print(f"Text: {result['text'][:150]}...")

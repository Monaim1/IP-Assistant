from typing import List, Dict, Optional, Tuple
import os
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer
from math import isfinite
from opensearchpy import OpenSearch  


class BM25Client:
    def __init__(self, client: "OpenSearch", index: str) -> None:  
        self._client = client
        self._index = index

    @classmethod
    def build_from_env(cls) -> Optional["BM25Client"]:
        enabled = os.getenv("BM25_ENABLED")
        bm25_enabled = (
            enabled.lower() in {"1", "true", "yes"}
            if enabled is not None
            else True
        )
        if not bm25_enabled or OpenSearch is None:
            return None

        host = os.getenv("OPENSEARCH_HOST") or os.getenv("ELASTICSEARCH_HOST") or "http://localhost:9200"
        user = os.getenv("OPENSEARCH_USER", "")
        password = os.getenv("OPENSEARCH_PASSWORD", "")
        index = os.getenv("OPENSEARCH_INDEX", "ip_chunks_bm25")
        auth = (user, password) if (user or password) else None

        try:
            client = OpenSearch(host, http_auth=auth)  # type: ignore
            try:
                if not client.ping():
                    return None
            except Exception:
                pass
            return cls(client, index)
        except Exception:
            return None

    def search(self, query: str, top_k: int, return_fields: List[str]) -> List[Dict]:
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text"],
                    "type": "best_fields",
                }
            },
            "_source": return_fields,
        }
        try:
            res = self._client.search(index=self._index, body=body)  # type: ignore[attr-defined]
            hits = res.get("hits", {}).get("hits", [])
            out: List[Dict] = []
            for h in hits:
                src = h.get("_source", {})
                row = {k: src.get(k) for k in return_fields}
                row["score"] = float(h.get("_score", 0.0) or 0.0)
                row["_bm25_score"] = row["score"]
                out.append(row)
            return out
        except Exception:
            return []

class MilvusRetriever:
    def __init__(
        self,
        collection_name: str = "ip_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "127.0.0.1",
        port: str = "19530",
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
        self._connect()

    def _connect(self) -> None:
        connections.connect("default", host=self.host, port=self.port)
        self.collection = Collection(self.collection_name)
        self.collection.load()

    def search(
        self,
        query: str,
        top_k: int,
        return_fields: List[str],
        min_score: float = 0.0,
    ) -> List[Dict]:
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()
        results = self.collection.search(  # type: ignore[union-attr]
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
        out: List[Dict] = []
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

    def __del__(self):
        if hasattr(self, "collection") and self.collection:
            try:
                self.collection.release()
                connections.disconnect("default")
            except Exception:
                pass


class PatentRetriever:

    def __init__(
        self,
        collection_name: str = "ip_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "127.0.0.1",
        port: str = "19530",
        milvus: Optional[MilvusRetriever] = None,
        bm25: Optional[BM25Client] = None,
    ):
        self.hybrid_alpha = 0.6
        # Components
        self.milvus = (
            milvus
            if milvus is not None
            else MilvusRetriever(
                collection_name=collection_name,
                model_name=model_name,
                host=host,
                port=port,
            )
        )
        self.bm25 = BM25Client.build_from_env()

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
        vec_results = self.milvus.search(query=query, top_k=top_k, return_fields=return_fields, min_score=min_score)

        # Lexical/BM25 search (OpenSearch)
        bm25_results = (
            self.bm25.search(query, top_k=top_k * 2, return_fields=return_fields)
            if self.bm25 is not None
            else []
        )

        # If BM25 not available, return dense results only
        if not bm25_results:
            return vec_results[:top_k]

        # Hybrid combine
        combined = self._combine_results(vec_results, bm25_results, top_k)
        return combined, vec_results, bm25_results

    def _combine_results(self, vec: List[Dict], bm25: List[Dict], top_k: int) -> List[Dict]:
        """Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = sum_i 1 / (k + rank_i), where `rank_i` is 1-based
        rank within each list and `k` (typically 60) dampens tail influence.
        """

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

        # Compute ranks within each list (defensive sort by their own scores)
        vec_sorted = sorted(vec, key=lambda d: d.get("_v_score", d.get("score", 0.0)), reverse=True)
        bm25_sorted = sorted(bm25, key=lambda d: d.get("_bm25_score", d.get("score", 0.0)), reverse=True)
        rank_vec = {key_of(d): i for i, d in enumerate(vec_sorted, start=1)}
        rank_bm25 = {key_of(d): i for i, d in enumerate(bm25_sorted, start=1)}

        rrf_k = 60

        out: List[Dict] = []
        for k, d in m.items():
            score = 0.0
            if k in rank_vec:
                score += 1.0 / (rrf_k + rank_vec[k])
            if k in rank_bm25:
                score += 1.0 / (rrf_k + rank_bm25[k])
            d2 = {kk: vv for kk, vv in d.items() if not kk.startswith("_")}
            d2["score"] = float(score)
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

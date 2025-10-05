from typing import List, Dict, Optional
from pymilvus import Collection, connections
from sentence_transformers import SentenceTransformer

class PatentRetriever:
    """
    A simple retriever for searching patent documents using Milvus vector database.
    """
    
    def __init__(
        self,
        collection_name: str = "ip_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "127.0.0.1",
        port: str = "19530"
    ):
        self.model = SentenceTransformer(model_name)
        self.host = host #  Milvus server host
        self.port = port # Milvus server port
        self.collection_name = collection_name
        self.collection = None
        self._connect()

    def _connect(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            self.collection = Collection(self.collection_name)
            self.collection.load()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        return_fields: Optional[List[str]] = None
    ) -> List[Dict]:

        if return_fields is None:
            return_fields = ["text", "publication_number", "section", "decision"]
        
        # Encode query
        query_embedding = self.model.encode(
            [query], 
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].tolist()

        # Search in Milvus
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=return_fields
        )

        # Process results more efficiently
        default_fields = {
            "score": 0.0,
            "publication_number": None,
            "text": "",
            "section": None,
            "decision": None
        }
        
        search_results = [
            {
                **default_fields,
                "score": float(hit.score),
                **{field: hit.entity.get(field, default_fields.get(field, None))
                for field in return_fields}
            }
            for hits in results
            for hit in hits
            if hit.score >= min_score
        ]

        return search_results

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
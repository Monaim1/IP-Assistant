"""
Evaluation helpers for RAG retrieval pipeline.

This module provides metrics and utilities to evaluate the quality of your retriever:
- Recall@K: Proportion of relevant documents retrieved in top-K results
- Precision@K: Proportion of retrieved documents that are relevant
- MRR (Mean Reciprocal Rank): Average of reciprocal ranks of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain (accounts for ranking position)
- Hit Rate@K: Whether at least one relevant document appears in top-K

Usage:
    1. Create a test dataset with queries and ground truth relevant documents
    2. Run your retriever on the queries
    3. Use these helpers to compute metrics
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection


class RetrievalEvaluator:
    """
    Evaluates retrieval quality using standard IR metrics.
    
    Example:
        evaluator = RetrievalEvaluator()
        
        # Define test queries with ground truth
        test_cases = [
            {
                "query": "solar panel efficiency improvements",
                "relevant_doc_ids": {"US20160012345", "US20150098765"}
            }
        ]
        
        # Get retrieval results from your system
        results = retriever.search(test_cases[0]["query"], k=10)
        retrieved_ids = [r["patent_number"] for r in results]
        
        # Compute metrics
        metrics = evaluator.evaluate_single(
            retrieved_ids,
            test_cases[0]["relevant_doc_ids"],
            k=10
        )
    """
    
    def __init__(self):
        self.metrics_history = []
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Recall@K: What proportion of relevant documents were retrieved in top-K?
        
        Formula: |Retrieved ∩ Relevant| / |Relevant|
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall score between 0 and 1
            
        Example:
            retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            relevant = {"doc2", "doc5", "doc10"}
            recall_at_k(retrieved, relevant, k=5)  # Returns 2/3 = 0.667
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k.intersection(relevant)
        return len(hits) / len(relevant)
    
    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Precision@K: What proportion of retrieved documents are relevant?
        
        Formula: |Retrieved ∩ Relevant| / K
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision score between 0 and 1
            
        Example:
            retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            relevant = {"doc2", "doc5", "doc10"}
            precision_at_k(retrieved, relevant, k=5)  # Returns 2/5 = 0.4
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k.intersection(relevant)
        return len(hits) / k
    
    def mean_reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """
        MRR: Average of reciprocal ranks of the first relevant document.
        
        Formula: 1 / rank_of_first_relevant_doc
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of ground truth relevant document IDs
            
        Returns:
            MRR score between 0 and 1
            
        Example:
            retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            relevant = {"doc3", "doc5"}
            mean_reciprocal_rank(retrieved, relevant)  # Returns 1/3 = 0.333
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain.
        Accounts for position - relevant docs at top rank higher.
        
        Formula: DCG@K / IDCG@K
        where DCG = Σ(relevance_i / log2(i + 1))
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            NDCG score between 0 and 1
            
        Example:
            retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
            relevant = {"doc2", "doc5"}
            ndcg_at_k(retrieved, relevant, k=5)  # Higher if relevant docs are ranked higher
        """
        if not relevant:
            return 0.0
        
        # DCG: Discounted Cumulative Gain
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                # Binary relevance: 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG: Ideal DCG (if all relevant docs were at top)
        idcg = 0.0
        for rank in range(1, min(len(relevant), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """
        Hit Rate@K: Is at least one relevant document in top-K?
        
        Args:
            retrieved: List of retrieved document IDs (in rank order)
            relevant: Set of ground truth relevant document IDs
            k: Number of top results to consider
            
        Returns:
            1.0 if at least one relevant doc in top-K, else 0.0
            
        Example:
            retrieved = ["doc1", "doc2", "doc3"]
            relevant = {"doc2", "doc10"}
            hit_rate_at_k(retrieved, relevant, k=3)  # Returns 1.0
        """
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if retrieved_at_k.intersection(relevant) else 0.0
    
    def evaluate_single(
        self, 
        retrieved: List[str], 
        relevant: Set[str], 
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single query.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of ground truth relevant document IDs
            k_values: List of K values to evaluate at
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        for k in k_values:
            metrics[f"recall@{k}"] = self.recall_at_k(retrieved, relevant, k)
            metrics[f"precision@{k}"] = self.precision_at_k(retrieved, relevant, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(retrieved, relevant, k)
            metrics[f"hit_rate@{k}"] = self.hit_rate_at_k(retrieved, relevant, k)
        
        metrics["mrr"] = self.mean_reciprocal_rank(retrieved, relevant)
        
        return metrics
    
    def evaluate_batch(
        self,
        test_cases: List[Dict],
        retriever_fn,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retriever on multiple test cases and return average metrics.
        
        Args:
            test_cases: List of dicts with 'query' and 'relevant_doc_ids'
            retriever_fn: Function that takes (query, k) and returns list of doc IDs
            k_values: List of K values to evaluate at
            
        Returns:
            Dictionary with averaged metrics across all test cases
            
        Example:
            test_cases = [
                {
                    "query": "solar panel efficiency",
                    "relevant_doc_ids": {"US20160012345", "US20150098765"}
                },
                {
                    "query": "battery storage systems",
                    "relevant_doc_ids": {"US20170054321"}
                }
            ]
            
            def my_retriever(query, k):
                # Your retrieval logic here
                results = collection.search(...)
                return [r["patent_number"] for r in results]
            
            metrics = evaluator.evaluate_batch(test_cases, my_retriever)
            print(f"Average Recall@5: {metrics['recall@5']:.3f}")
        """
        all_metrics = defaultdict(list)
        
        max_k = max(k_values)
        
        for test_case in test_cases:
            query = test_case["query"]
            relevant = test_case["relevant_doc_ids"]
            
            # Get retrieval results
            retrieved = retriever_fn(query, max_k)
            
            # Compute metrics for this query
            metrics = self.evaluate_single(retrieved, relevant, k_values)
            
            # Accumulate
            for metric_name, value in metrics.items():
                all_metrics[metric_name].append(value)
        
        # Average across all queries
        avg_metrics = {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
        }
        
        # Store for later analysis
        self.metrics_history.append(avg_metrics)
        
        return avg_metrics
    
    def print_metrics(self, metrics: Dict[str, float]) -> None:
        """Pretty print evaluation metrics."""
        print("\n" + "="*50)
        print("RETRIEVAL EVALUATION METRICS")
        print("="*50)
        
        # Group by K value
        k_values = sorted(set(
            int(k.split("@")[1]) 
            for k in metrics.keys() 
            if "@" in k
        ))
        
        for k in k_values:
            print(f"\n📊 Top-{k} Results:")
            print(f"  • Recall@{k}:    {metrics.get(f'recall@{k}', 0):.3f}")
            print(f"  • Precision@{k}: {metrics.get(f'precision@{k}', 0):.3f}")
            print(f"  • NDCG@{k}:      {metrics.get(f'ndcg@{k}', 0):.3f}")
            print(f"  • Hit Rate@{k}:  {metrics.get(f'hit_rate@{k}', 0):.3f}")
        
        print(f"\n🎯 Overall:")
        print(f"  • MRR: {metrics.get('mrr', 0):.3f}")
        print("="*50 + "\n")


class MilvusRetrieverWrapper:
    """
    Wrapper for Milvus retriever to use with evaluation.
    """
    
    def __init__(
        self,
        collection_name: str = "ip_chunks",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        host: str = "127.0.0.1",
        port: str = "19530"
    ):
        """Initialize connection to Milvus and load embedding model."""
        connections.connect("default", host=host, port=port)
        self.collection = Collection(collection_name)
        self.collection.load()
        self.model = SentenceTransformer(model_name)
    
    def search(
        self,
        query: str,
        k: int = 10,
        filters: str = None
    ) -> List[str]:
        """
        Search and return list of patent numbers.
        
        Args:
            query: Search query text
            k: Number of results to return
            filters: Optional Milvus filter expression
            
        Returns:
            List of patent numbers in rank order
        """
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        # Execute search
        results = self.collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=filters,
            output_fields=["patent_number", "application_number", "text"]
        )
        
        # Extract patent numbers
        patent_numbers = []
        for hits in results:
            for hit in hits:
                patent_num = hit.entity.get("patent_number")
                if patent_num:
                    patent_numbers.append(patent_num)
        
        return patent_numbers


# ============================================================================
# EXAMPLE TEST DATASET CREATION
# ============================================================================

def create_synthetic_test_set() -> List[Dict]:
    """
    Create a synthetic test dataset for evaluation.
    
    In practice, you should:
    1. Manually curate queries with known relevant patents
    2. Use existing query logs with relevance judgments
    3. Have domain experts label relevant documents
    
    Returns:
        List of test cases with queries and relevant document IDs
    """
    return [
        {
            "query": "solar panel efficiency improvements using nanostructures",
            "relevant_doc_ids": {
                # Add actual patent numbers from your collection
                "US20160012345",
                "US20150098765"
            }
        },
        {
            "query": "battery storage systems for electric vehicles",
            "relevant_doc_ids": {
                "US20170054321",
                "US20180076543"
            }
        },
        {
            "query": "machine learning for image classification",
            "relevant_doc_ids": {
                "US20190123456",
                "US20190234567"
            }
        }
    ]


def analyze_retrieval_failures(
    evaluator: RetrievalEvaluator,
    test_cases: List[Dict],
    retriever: MilvusRetrieverWrapper,
    k: int = 10
) -> None:
    """
    Analyze cases where retrieval failed to find relevant documents.
    
    This helps identify:
    - Queries that need better embeddings
    - Missing documents in the index
    - Need for query expansion or rewriting
    """
    print("\n" + "="*50)
    print("FAILURE ANALYSIS")
    print("="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        relevant = test_case["relevant_doc_ids"]
        
        retrieved = retriever.search(query, k=k)
        retrieved_set = set(retrieved)
        
        # Check what was missed
        missed = relevant - retrieved_set
        
        if missed:
            print(f"\n❌ Query {i}: {query}")
            print(f"   Missed {len(missed)}/{len(relevant)} relevant docs:")
            for doc_id in missed:
                print(f"     - {doc_id}")
            print(f"   Retrieved: {retrieved[:5]}...")


# ============================================================================
# MAIN EVALUATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of evaluation helpers.
    """
    
    print("🔍 Initializing Retrieval Evaluator...")
    
    # 1. Initialize evaluator and retriever
    evaluator = RetrievalEvaluator()
    retriever = MilvusRetrieverWrapper()
    
    # 2. Create or load test dataset
    # NOTE: Replace with your actual test dataset
    test_cases = create_synthetic_test_set()
    
    print(f"📝 Loaded {len(test_cases)} test cases")
    
    # 3. Define retriever function
    def retriever_fn(query: str, k: int) -> List[str]:
        return retriever.search(query, k=k)
    
    # 4. Run evaluation
    print("\n🚀 Running evaluation...")
    metrics = evaluator.evaluate_batch(
        test_cases=test_cases,
        retriever_fn=retriever_fn,
        k_values=[1, 3, 5, 10]
    )
    
    # 5. Print results
    evaluator.print_metrics(metrics)
    
    # 6. Analyze failures
    analyze_retrieval_failures(evaluator, test_cases, retriever, k=10)
    
    print("\n✅ Evaluation complete!")
    print("\n💡 Next steps:")
    print("  1. Create a real test dataset with domain expert labels")
    print("  2. Experiment with different embedding models")
    print("  3. Try different chunking strategies")
    print("  4. Add hybrid retrieval (BM25 + vector)")
    print("  5. Implement re-ranking with cross-encoders")

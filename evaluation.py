from typing import List, Dict, Callable, Iterable
import math
import json
import pandas as pd
from retriever import PatentRetriever


# Helper functions for metric computation
def compute_precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Compute Precision@K: fraction of retrieved documents that are relevant.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0 or len(retrieved) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    num_relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return num_relevant_retrieved / k


def compute_recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Compute Recall@K: fraction of relevant documents that are retrieved.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    num_relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
    return num_relevant_retrieved / len(relevant)


def compute_f1_at_k(precision: float, recall: float) -> float:
    """
    Compute F1@K: harmonic mean of precision and recall.
    
    Args:
        precision: Precision@K score
        recall: Recall@K score
    
    Returns:
        F1@K score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def compute_success_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """
    Compute Success@K: binary indicator if any relevant document is in top-k.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        k: Number of top results to consider
    
    Returns:
        1.0 if any relevant document is found, 0.0 otherwise
    """
    retrieved_at_k = retrieved[:k]
    return 1.0 if any(doc_id in relevant for doc_id in retrieved_at_k) else 0.0


def compute_rank(retrieved: List[str], relevant: set, max_rank: int = 100) -> int:
    """
    Compute the rank of the first relevant document.
    
    Args:
        retrieved: List of retrieved document IDs (in ranked order)
        relevant: Set of relevant document IDs
        max_rank: Maximum rank to return if no relevant document is found
    
    Returns:
        Rank of first relevant document (1-indexed), or max_rank if not found
    """
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return i
    return max_rank


def evaluate_retriever(
    df: 'pd.DataFrame',
    retriever: Callable[[str, int], List[str]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate a retriever using standard IR metrics.
    
    Args:
        df: DataFrame with 'query' and 'document_id' columns
        retriever: Function that takes (query: str, k: int) and returns list of document IDs
        ks: List of k values to evaluate at (e.g., [1, 3, 5, 10])
        
    Returns:
        Dictionary mapping k to metric scores
    """
    ks = sorted(set(ks))
    max_k = max(ks) if ks else 10
    
    # Initialize accumulators for micro-averaged metrics
    metrics = {k: {
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "success_count": 0,
        "total_rank": 0,
        "query_count": 0
    } for k in ks}
    
    # Convert document_id to string for consistent comparison
    df = df.copy()
    df['document_id'] = df['document_id'].astype(str)
    
    # Group by query to handle multiple relevant documents
    query_groups = df.groupby('query')
    num_queries = len(query_groups)
    
    for query, group in query_groups:
        # Get all relevant document IDs for this query
        relevant_docs = set(group['document_id'].unique())
        
        # Get ranked list of documents from retriever
        retrieved_docs = [str(doc_id) for doc_id in retriever(query, max_k)]
        
        # For each k, calculate metrics
        for k in ks:
            retrieved_at_k = retrieved_docs[:k]
            
            # Count true positives, false positives, false negatives
            true_positives = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_docs)
            false_positives = len(retrieved_at_k) - true_positives
            false_negatives = len(relevant_docs) - true_positives
            
            # Accumulate counts for micro-averaging
            metrics[k]["true_positives"] += true_positives
            metrics[k]["false_positives"] += false_positives
            metrics[k]["false_negatives"] += false_negatives
            
            # Success@K (binary)
            success = compute_success_at_k(retrieved_docs, relevant_docs, k)
            metrics[k]["success_count"] += success
            
            # Rank of first relevant document
            rank = compute_rank(retrieved_docs, relevant_docs, max_k + 1)
            metrics[k]["total_rank"] += rank
            
            metrics[k]["query_count"] += 1
    
    # Calculate final metrics using micro-averaging
    results = {}
    for k in ks:
        m = metrics[k]
        tp = m["true_positives"]
        fp = m["false_positives"]
        fn = m["false_negatives"]
        
        # Micro-averaged Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = compute_f1_at_k(precision, recall)
        
        # Success rate and average rank
        success_rate = m["success_count"] / m["query_count"] if m["query_count"] > 0 else 0.0
        avg_rank = m["total_rank"] / m["query_count"] if m["query_count"] > 0 else max_k + 1
        
        results[k] = {
            "Success@K": round(success_rate, 4),
            "Precision@K": round(precision, 4),
            "Recall@K": round(recall, 4),
            "F1@K": round(f1, 4),
            "AvgRank": round(avg_rank, 2)
        }
    
    # Print results
    print("\n" + "="*80)
    print("Retriever Evaluation Results (Micro-Averaged)")
    print("="*80)
    print(f"Number of queries: {num_queries}")
    print("-"*80)
    print("K     | " + 
          "Success@K  | " + 
          "Precision@K  | " + 
          "Recall@K   | " + 
          "F1@K     | " + 
          "AvgRank")
    print("-"*80)
    
    for k in ks:
        m = results[k]
        print(f"{str(k).ljust(6)}| " +
              f"{m['Success@K']:.4f}   | " +
              f"{m['Precision@K']:.4f}     | " +
              f"{m['Recall@K']:.4f}  | " +
              f"{m['F1@K']:.4f} | " +
              f"{m['AvgRank']:.2f}")
    
    return results

def search_wrapper(query: str, k: int) -> List[str]:
    results = retriever.search(query, top_k=k)
    return [str(result['publication_number']) for result in results]

if __name__ == "__main__":
    retriever = PatentRetriever()
    test_dataset = pd.read_json('retrieval_dataset.jsonl', lines=True)
    evaluate_retriever(test_dataset, search_wrapper)

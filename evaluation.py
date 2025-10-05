from typing import List, Dict, Callable, Iterable
import pandas as pd
from retriever import PatentRetriever


def _unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _first_hit_rank(retrieved: List[str], relevant: str, k_cap: int) -> int:
    """1-indexed rank of the first relevant doc within top-k_cap; k_cap+1 if not found."""
    for i, doc_id in enumerate(retrieved[:k_cap], 1):
        if doc_id == relevant:
            return i
    return k_cap + 1

def evaluate_retriever(
    df: "pd.DataFrame",
    retriever: Callable[[str, int], List[str]],
    ks: Iterable[int] = (1, 3, 5, 10),
    id_col: str = "document_id",
    query_col: str = "query",
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate a retriever assuming each row is ONE independent (query, relevant_id) pair.
    Required columns: `query_col` and `id_col`.
    """
    assert query_col in df.columns and id_col in df.columns, \
        f"DataFrame must contain '{query_col}' and '{id_col}'"

    ks = sorted(set(int(k) for k in ks))
    max_k = max(ks) if ks else 10

    # Micro-accumulators
    acc = {
        k: dict(tp=0, fp=0, fn=0, success=0, mrr_sum=0.0, total_rank=0, n=0)
        for k in ks
    }

    n_rows = len(df)
    for _, row in df.iterrows():
        q = str(row[query_col])
        rel = str(row[id_col])

        retrieved = [str(x) for x in retriever(q, max_k)]
        retrieved = _unique_preserve_order(retrieved)

        for k in ks:
            topk = retrieved[:k]
            hit = rel in topk
            tp = 1 if hit else 0
            fp = (len(set(topk)) - tp)  # duplicates already removed
            fn = 0 if hit else 1

            # rank & MRR@K
            rank = _first_hit_rank(retrieved, rel, k)
            mrr = 1.0 / rank if hit else 0.0

            a = acc[k]
            a["tp"] += tp
            a["fp"] += fp
            a["fn"] += fn
            a["success"] += 1 if hit else 0
            a["mrr_sum"] += mrr
            a["total_rank"] += rank
            a["n"] += 1

    # Finalize micro-averaged metrics
    results = {}
    for k in ks:
        a = acc[k]
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        n = max(a["n"], 1)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        success_rate = a["success"] / n
        avg_rank = a["total_rank"] / n
        mrr_at_k = a["mrr_sum"] / n

        # Small numeric guard
        recall = min(recall, 1.0)

        results[k] = {
            "Success@K": round(success_rate, 4),
            "Precision@K": round(precision, 4),
            "Recall@K": round(recall, 4),
            "F1@K": round(f1, 4),
            "MRR@K": round(mrr_at_k, 4),
            "AvgRank": round(avg_rank, 2),
        }

    # Pretty print
    print("\n" + "="*80)
    print("Retriever Evaluation Results (Micro-Averaged, row = independent query)")
    print("="*80)
    print(f"Number of rows (queries): {n_rows}")
    print("-"*80)
    print("K     | Success@K | Precision@K | Recall@K | F1@K  | MRR@K | AvgRank")
    print("-"*80)
    for k in ks:
        m = results[k]
        print(f"{str(k).ljust(6)}| "
              f"{m['Success@K']:.4f}   | "
              f"{m['Precision@K']:.4f}      | "
              f"{m['Recall@K']:.4f}   | "
              f"{m['F1@K']:.4f} | "
              f"{m['MRR@K']:.4f} | "
              f"{m['AvgRank']:.2f}")
    return results


def search_wrapper(query: str, k: int) -> List[str]:
    results = retriever.search(query, top_k=k)
    return [str(result['publication_number']) for result in results]

if __name__ == "__main__":
    retriever = PatentRetriever()
    test_dataset = pd.read_json('evals/retrieval_dataset.jsonl', lines=True)
    evaluate_retriever(test_dataset, search_wrapper)

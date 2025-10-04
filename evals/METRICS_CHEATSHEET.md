# RAG Evaluation Metrics - Quick Reference

## 📊 The 5 Key Metrics

### 1. Recall@K
```
What: Did we find the relevant documents?
Formula: (# relevant found) / (# total relevant)
Range: 0.0 to 1.0 (higher is better)

Example:
  Ground truth: 10 relevant patents
  Found in top-10: 7 patents
  Recall@10 = 7/10 = 0.70

When to care:
  ✓ Research/discovery tasks
  ✓ When missing results is costly
  ✓ Comprehensive coverage needed
```

### 2. Precision@K
```
What: Are the retrieved documents actually relevant?
Formula: (# relevant found) / K
Range: 0.0 to 1.0 (higher is better)

Example:
  Retrieved: 10 documents
  Relevant: 4 of them
  Precision@10 = 4/10 = 0.40

When to care:
  ✓ Production systems
  ✓ Limited UI space
  ✓ User time is valuable
```

### 3. MRR (Mean Reciprocal Rank)
```
What: How high is the first relevant result?
Formula: 1 / (rank of first relevant doc)
Range: 0.0 to 1.0 (higher is better)

Example:
  First relevant at position 3
  MRR = 1/3 = 0.333

When to care:
  ✓ Search interfaces
  ✓ Users click first result
  ✓ Question answering
```

### 4. NDCG@K (Normalized Discounted Cumulative Gain)
```
What: Quality of ranking (rewards relevant docs at top)
Formula: DCG@K / IDCG@K
Range: 0.0 to 1.0 (higher is better)

Example:
  Relevant docs at: 1, 3, 5
  NDCG@5 = 0.86
  (Perfect ranking = 1.0)

When to care:
  ✓ Ranking order matters
  ✓ Recommendation systems
  ✓ Multiple relevant results
```

### 5. Hit Rate@K
```
What: Did we find at least one relevant document?
Formula: 1 if any relevant in top-K, else 0
Range: 0.0 or 1.0

Example:
  Top-5: [doc1, doc2, doc3, doc4, doc5]
  Relevant: {doc3, doc10}
  Hit Rate@5 = 1.0 (found doc3)

When to care:
  ✓ Initial validation
  ✓ Any result is acceptable
  ✓ Broad discovery
```

---

## 🎯 What's a "Good" Score?

| Metric | Excellent | Good | Needs Work |
|--------|-----------|------|------------|
| Recall@10 | > 0.8 | 0.6-0.8 | < 0.6 |
| Precision@10 | > 0.5 | 0.3-0.5 | < 0.3 |
| MRR | > 0.7 | 0.5-0.7 | < 0.5 |
| NDCG@10 | > 0.8 | 0.6-0.8 | < 0.6 |
| Hit Rate@10 | > 0.9 | 0.7-0.9 | < 0.7 |

---

## 🔧 Quick Fixes by Problem

### 😢 Low Recall (< 0.6)
**Problem**: Missing relevant documents

**Quick Fixes**:
```python
# 1. Increase K
results = retriever.search(query, k=20)  # was k=10

# 2. Better embedding model
retriever = MilvusRetrieverWrapper(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# 3. Hybrid search (add BM25)
vector_results = milvus_search(query)
bm25_results = elasticsearch_search(query)
combined = merge(vector_results, bm25_results)
```

---

### 😢 Low Precision (< 0.3)
**Problem**: Too many irrelevant results

**Quick Fixes**:
```python
# 1. Add re-ranking (BEST FIX)
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

candidates = retriever.search(query, k=50)
scores = reranker.predict([(query, doc) for doc in candidates])
results = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)][:10]

# 2. Use filters
results = collection.search(
    ...,
    expr='main_cpc_label == "H01L"'  # Filter by patent class
)

# 3. Set similarity threshold
results = [r for r in results if r.score > 0.7]
```

---

### 😢 Low NDCG (< 0.6)
**Problem**: Relevant docs not ranked high

**Quick Fixes**:
```python
# 1. Re-ranking (same as precision)

# 2. Tune search parameter
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 128}  # Higher = better ranking
}

# 3. Fine-tune embeddings on your data
```

---

### 😢 Low MRR (< 0.5)
**Problem**: First result often wrong

**Quick Fixes**:
```python
# Re-ranking is CRITICAL for MRR
# Use cross-encoder (see precision fixes above)
```

---

## 📝 Usage Cheatsheet

### Basic Evaluation
```python
from evaluation import RetrievalEvaluator, MilvusRetrieverWrapper

# Setup
evaluator = RetrievalEvaluator()
retriever = MilvusRetrieverWrapper()

# Single query
retrieved = retriever.search("solar panels", k=10)
relevant = {"US20160012345", "US20150098765"}
metrics = evaluator.evaluate_single(retrieved, relevant)

# Print results
evaluator.print_metrics(metrics)
```

### Batch Evaluation
```python
# Test dataset
test_cases = [
    {
        "query": "solar panels",
        "relevant_doc_ids": {"US20160012345", "US20150098765"}
    },
    {
        "query": "batteries",
        "relevant_doc_ids": {"US20170054321"}
    }
]

# Evaluate
metrics = evaluator.evaluate_batch(
    test_cases=test_cases,
    retriever_fn=lambda q, k: retriever.search(q, k),
    k_values=[1, 3, 5, 10]
)

print(f"Recall@5: {metrics['recall@5']:.3f}")
```

### Command Line
```bash
# Quick test
python quick_eval_example.py

# Full evaluation
python run_evaluation.py \
  --test-file my_test_data.json \
  --k 1 3 5 10 \
  --detailed \
  --plot results.png
```

---

## 🎓 Metric Selection Guide

**Choose metrics based on your use case:**

### Use Case: Research/Discovery
**Goal**: Find all relevant patents
- **Primary**: Recall@10, Recall@20
- **Secondary**: Hit Rate@10
- **Target**: Recall@10 > 0.8

### Use Case: Question Answering
**Goal**: Get the right answer quickly
- **Primary**: MRR, Precision@3
- **Secondary**: NDCG@5
- **Target**: MRR > 0.7

### Use Case: Recommendation System
**Goal**: Show relevant results in order
- **Primary**: NDCG@10, Precision@10
- **Secondary**: Recall@10
- **Target**: NDCG@10 > 0.8

### Use Case: Production Search
**Goal**: Balance coverage and relevance
- **Primary**: Recall@10, Precision@10
- **Secondary**: NDCG@10, MRR
- **Target**: Recall@10 > 0.7, Precision@10 > 0.4

---

## 🚀 Optimization Priority

**If you can only do ONE thing:**
→ **Add re-ranking with cross-encoder**
   - Improves all metrics
   - Easy to implement
   - 10-20% improvement typical

**If you can do TWO things:**
1. Re-ranking (above)
2. Better embedding model (all-mpnet-base-v2)

**If you can do THREE things:**
1. Re-ranking
2. Better embedding model
3. Hybrid search (vector + BM25)

---

## 📊 Interpreting Results

### Example Output
```
Recall@5:    0.65
Precision@5: 0.40
NDCG@5:      0.72
MRR:         0.55
```

**What this means:**
- ✅ Found 65% of relevant documents (decent)
- ⚠️  Only 40% of results are relevant (could be better)
- ✅ Ranking is pretty good (0.72)
- ⚠️  First relevant doc around position 2 (could be better)

**Action items:**
1. Add re-ranking to improve precision and MRR
2. Consider hybrid search to boost recall

---

## 💡 Pro Tips

1. **Start small**: 20-30 test queries is enough to start
2. **Version everything**: Track test datasets and metrics over time
3. **Don't over-optimize**: Balance metrics, don't chase one number
4. **Real users matter**: Metrics are proxies, user feedback is truth
5. **Iterate quickly**: Small improvements compound

---

## 🎯 Common Mistakes

❌ **Testing on training data**
→ Keep test set completely separate

❌ **Too few test cases**
→ Need at least 20-30 for meaningful results

❌ **Ignoring variance**
→ Small changes might be noise

❌ **Optimizing one metric**
→ Balance recall, precision, and ranking

✅ **Do this instead**:
- Separate train/test data
- Use 50+ test queries
- Track multiple metrics
- Validate with real users

---

## 📚 Quick Reference

### Files to Use
- `evaluation.py` - Core library
- `quick_eval_example.py` - Test immediately
- `run_evaluation.py` - Full pipeline
- `EVALUATION_GUIDE.md` - Detailed docs

### Embedding Models (Best to Worst)
1. `BAAI/bge-large-en-v1.5` (best, slowest)
2. `sentence-transformers/all-mpnet-base-v2` (good balance)
3. `sentence-transformers/all-MiniLM-L6-v2` (fast baseline)

### Re-ranking Models
1. `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast)
2. `cross-encoder/ms-marco-MiniLM-L-12-v2` (better)

---

## 🤔 FAQ

**Q: Which metric is most important?**
A: Depends on use case. For most: balance Recall@10 and Precision@10.

**Q: What K value should I use?**
A: Start with K=10. Evaluate at multiple K values: [1, 3, 5, 10].

**Q: How do I create ground truth?**
A: Have domain experts label. Start with 20 queries, expand to 50+.

**Q: My metrics are all low. Help!**
A: 
1. Verify test dataset is correct
2. Check documents are in collection
3. Try simpler queries first
4. Add re-ranking

---

**Need more help?** Check `EVALUATION_README.md` for full guide!

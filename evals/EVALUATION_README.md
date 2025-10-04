# RAG Retrieval Evaluation Toolkit

Complete toolkit for evaluating and fine-tuning your IP patent retrieval system.

## 📁 Files Overview

### Core Evaluation Module
- **`evaluation.py`** - Main evaluation library with all metrics and helpers
  - `RetrievalEvaluator` class with Recall@K, Precision@K, MRR, NDCG@K, Hit Rate@K
  - `MilvusRetrieverWrapper` for easy integration with your Milvus collection
  - Batch evaluation and failure analysis tools

### Quick Start Scripts
- **`quick_eval_example.py`** - Minimal working example to test immediately
  ```bash
  python quick_eval_example.py
  ```

- **`run_evaluation.py`** - Full evaluation pipeline with visualization
  ```bash
  python run_evaluation.py --test-file my_test_data.json --k 1 3 5 10 --detailed
  ```

### Documentation
- **`EVALUATION_GUIDE.md`** - Comprehensive guide covering:
  - What each metric means and when to use it
  - How to create test datasets
  - Optimization strategies for low metrics
  - Advanced techniques (hybrid search, re-ranking, etc.)

- **`Evaluation_Examples.ipynb`** - Interactive Jupyter notebook with:
  - Step-by-step examples
  - Visualizations
  - Comparison tools
  - Failure analysis

### Templates
- **`test_dataset_template.json`** - Template for creating your test dataset

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run Quick Example
```bash
# Make sure Milvus is running and you've ingested data
python quick_eval_example.py
```

This will:
- Connect to your Milvus collection
- Run a sample query
- Show you what the metrics mean

### Step 2: Create Test Dataset
```bash
# Copy the template
cp test_dataset_template.json my_test_data.json

# Edit my_test_data.json and add your queries + ground truth
```

Example test case:
```json
{
  "query_id": "Q001",
  "query": "solar panel efficiency improvements",
  "relevant_doc_ids": ["US20160012345", "US20150098765"],
  "description": "Patents about improving solar cell efficiency"
}
```

### Step 3: Run Full Evaluation
```bash
python run_evaluation.py \
  --test-file my_test_data.json \
  --k 1 3 5 10 \
  --detailed \
  --plot evaluation_plot.png
```

This will:
- Evaluate your retriever on all test queries
- Show averaged metrics
- Create visualizations
- Provide optimization recommendations
- Save results to JSON

---

## 📊 Metrics Explained (TL;DR)

| Metric | What It Measures | Good Value | When to Optimize |
|--------|------------------|------------|------------------|
| **Recall@K** | Coverage - did we find relevant docs? | > 0.7 | Missing important results |
| **Precision@K** | Relevance - are results useful? | > 0.4 | Too much noise |
| **NDCG@K** | Ranking quality | > 0.7 | Relevant docs not at top |
| **MRR** | First result quality | > 0.6 | First result often wrong |
| **Hit Rate@K** | Found at least one relevant? | > 0.8 | Basic success rate |

### Quick Interpretation

**Your metrics show:**
```
Recall@5: 0.65  → Found 65% of relevant documents
Precision@5: 0.40  → 40% of results are relevant (2 out of 5)
NDCG@5: 0.72  → Good ranking, but not perfect
MRR: 0.55  → First relevant doc around position 2
```

---

## 🔧 Optimization Strategies

### Problem: Low Recall (< 0.6)
**You're missing relevant documents**

**Quick Fixes:**
1. Increase K (retrieve more documents)
2. Try better embedding model:
   ```python
   retriever = MilvusRetrieverWrapper(
       model_name="sentence-transformers/all-mpnet-base-v2"  # Better than MiniLM
   )
   ```
3. Add hybrid search (vector + BM25)

**Code Example:**
```python
# Increase K
results = retriever.search(query, k=20)  # Instead of k=10

# Try better model
retriever = MilvusRetrieverWrapper(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

---

### Problem: Low Precision (< 0.3)
**Too many irrelevant results**

**Quick Fixes:**
1. Add re-ranking with cross-encoder
2. Use metadata filters
3. Better chunking strategy

**Code Example:**
```python
from sentence_transformers import CrossEncoder

# Re-rank results
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get initial results
candidates = retriever.search(query, k=50)

# Re-rank
scores = reranker.predict([(query, doc) for doc in candidates])
reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)]

# Return top-K
final_results = reranked[:10]
```

---

### Problem: Low NDCG (< 0.6)
**Relevant docs not ranked high enough**

**Quick Fixes:**
1. Add re-ranking (same as precision)
2. Tune HNSW search parameter
3. Fine-tune embeddings

**Code Example:**
```python
# Tune search parameter for better ranking
search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 128}  # Higher = better ranking, slower
}
```

---

### Problem: Low MRR (< 0.5)
**First result often not relevant**

**Quick Fixes:**
1. Re-ranking is critical for MRR
2. Better query understanding
3. Add contextual information to chunks

---

## 📖 Usage Examples

### Example 1: Basic Evaluation
```python
from evaluation import RetrievalEvaluator, MilvusRetrieverWrapper

# Initialize
evaluator = RetrievalEvaluator()
retriever = MilvusRetrieverWrapper()

# Define test case
query = "solar panel efficiency"
relevant_ids = {"US20160012345", "US20150098765"}

# Get results
retrieved = retriever.search(query, k=10)

# Evaluate
metrics = evaluator.evaluate_single(retrieved, relevant_ids)
evaluator.print_metrics(metrics)
```

### Example 2: Batch Evaluation
```python
# Load test dataset
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

# Run evaluation
metrics = evaluator.evaluate_batch(
    test_cases=test_cases,
    retriever_fn=lambda q, k: retriever.search(q, k),
    k_values=[1, 3, 5, 10]
)

print(f"Average Recall@5: {metrics['recall@5']:.3f}")
```

### Example 3: Compare Configurations
```python
# Baseline
retriever_v1 = MilvusRetrieverWrapper(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
metrics_v1 = evaluator.evaluate_batch(test_cases, retriever_v1.search)

# Better model
retriever_v2 = MilvusRetrieverWrapper(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
metrics_v2 = evaluator.evaluate_batch(test_cases, retriever_v2.search)

# Compare
improvement = metrics_v2['recall@5'] - metrics_v1['recall@5']
print(f"Recall@5 improvement: {improvement:.3f}")
```

---

## 🎯 Creating Good Test Datasets

### Option 1: Manual Curation (Best Quality)
Have domain experts review and label:
```json
{
  "query": "solar panel efficiency improvements",
  "relevant_doc_ids": ["US20160012345", "US20150098765"],
  "notes": "Manually verified by patent examiner"
}
```

**Pros**: High quality, accurate  
**Cons**: Time-consuming

---

### Option 2: Use Metadata (Quick Start)
Use CPC codes or other metadata:
```python
# Find patents with same CPC code
def create_test_from_cpc(patent_id, cpc_code):
    similar_patents = collection.query(
        expr=f'main_cpc_label == "{cpc_code}"',
        output_fields=["patent_number"]
    )
    return {
        "query": f"Patents similar to {patent_id}",
        "relevant_doc_ids": set(similar_patents)
    }
```

**Pros**: Fast, automated  
**Cons**: May not capture semantic similarity

---

### Option 3: LLM-Assisted (Balanced)
Use LLM to generate relevance judgments:
```python
def llm_judge_relevance(query, document):
    prompt = f"""
    Query: {query}
    Document: {document}
    
    Is this document relevant? Answer YES or NO.
    """
    response = llm.generate(prompt)
    return "YES" in response
```

**Pros**: Scalable, reasonable quality  
**Cons**: Needs validation

---

## 📈 Typical Performance Ranges

### Excellent System
- Recall@10: > 0.8
- Precision@10: > 0.5
- NDCG@10: > 0.8
- MRR: > 0.7

### Good System
- Recall@10: 0.6 - 0.8
- Precision@10: 0.3 - 0.5
- NDCG@10: 0.6 - 0.8
- MRR: 0.5 - 0.7

### Needs Improvement
- Recall@10: < 0.6
- Precision@10: < 0.3
- NDCG@10: < 0.6
- MRR: < 0.5

---

## 🛠️ Advanced Techniques

### 1. Hybrid Retrieval (Vector + BM25)
Combine semantic and keyword search:
```python
def hybrid_search(query, k=10, alpha=0.7):
    vector_results = milvus_search(query, k=k*2)
    bm25_results = elasticsearch_search(query, k=k*2)
    
    # Combine scores
    combined = {}
    for doc_id, score in vector_results:
        combined[doc_id] = alpha * score
    
    for doc_id, score in bm25_results:
        combined[doc_id] = combined.get(doc_id, 0) + (1-alpha) * score
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]
```

### 2. Re-ranking with Cross-Encoder
Improve ranking quality:
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Get candidates
candidates = retriever.search(query, k=50)

# Re-rank
scores = reranker.predict([(query, doc) for doc in candidates])
reranked = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)][:10]
```

### 3. Contextual Embeddings
Add context to chunks:
```python
contextualized_chunk = f"""
Patent: {patent_number}
Section: {section}
Topic: {topic}

{chunk_text}
"""
embedding = model.encode(contextualized_chunk)
```

---

## 📚 Resources

### Recommended Embedding Models
1. **all-MiniLM-L6-v2** (384-d) - Fast, good baseline
2. **all-mpnet-base-v2** (768-d) - Better quality, slower
3. **bge-large-en-v1.5** (1024-d) - State-of-the-art
4. **e5-large-v2** (1024-d) - Excellent for retrieval

### Papers
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [ColBERT](https://arxiv.org/abs/2004.12832)

### Tools
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Standard IR evaluation
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Model rankings

---

## 🤔 FAQ

**Q: How many test queries do I need?**  
A: Minimum 20-30 for meaningful results. Aim for 50-100+ for production systems.

**Q: What's a good Recall@5 value?**  
A: Depends on use case, but generally > 0.7 is good. For research/discovery, aim higher.

**Q: Should I optimize for Recall or Precision?**  
A: Depends on your use case:
- Research/discovery → Optimize Recall
- Production UI → Balance both
- Question answering → Optimize Precision + MRR

**Q: How often should I re-evaluate?**  
A: 
- After any system changes
- Monthly for production systems
- When you add new data

**Q: My metrics are low. What's the quickest win?**  
A: Add re-ranking with a cross-encoder. It's easy to implement and often gives 10-20% improvement.

---

## 🎓 Next Steps

1. **Run quick example**: `python quick_eval_example.py`
2. **Create test dataset**: Edit `test_dataset_template.json`
3. **Run full evaluation**: `python run_evaluation.py --test-file my_test_data.json --detailed`
4. **Analyze results**: Check which metrics are low
5. **Optimize**: Try recommendations from the guide
6. **Iterate**: Re-evaluate and compare

---

## 💡 Tips

- Start with a small test set (20 queries) and expand
- Version your test datasets
- Track metrics over time
- Collect user feedback in production
- Don't optimize for one metric only
- Balance quality vs. speed

---

## 🐛 Troubleshooting

**Error: "Collection not found"**
- Make sure you've run `ingestion.py` first
- Check collection name matches

**Error: "Connection refused"**
- Start Milvus: `docker-compose up -d`
- Check ports: 19530 for Milvus

**Low metrics across the board**
- Check if test dataset is correct
- Verify documents are actually in the collection
- Try a simpler query first

---

## 📞 Support

For questions or issues:
1. Check `EVALUATION_GUIDE.md` for detailed explanations
2. Review `Evaluation_Examples.ipynb` for interactive examples
3. Look at code comments in `evaluation.py`

Happy evaluating! 🚀

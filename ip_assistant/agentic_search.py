import json
from typing import Any, Callable, Dict, List, Optional

from ip_assistant.retriever import PatentRetriever
from ip_assistant.utils import get_LLM_response
import dspy 




# dspy.configure(lm=dspy.LM(...))
# dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

class PatentSearchSig(dspy.Signature):
    """
    You are a patent search agent.
    Goal: Understand the user's intent. If missing details, ask 1–2 specific questions using clarify().
    Then create a focused search using enhance_query().
    ALWAYS use search_patents() to retrieve doc IDs and read_documents() to inspect content.
    If results are insufficient or off-target, refine and repeat up to the allowed passes.

    At the end, output ONLY a JSON object:
    {"num_passes": N, "final_top_results": ["id1", ...]}.
    """
    question: str = dspy.InputField(desc="User's patent query or task")
    answer: str = dspy.OutputField(
        desc='Return ONLY the final JSON object as a string.'
    )

def _coerce_combined(results) -> List[Dict]:
    """Return the combined results list regardless of PatentRetriever variant."""
    if isinstance(results, tuple):
        # (combined, vec, bm25)
        return results[0]
    return results



def _build_doc_id(doc: Dict, idx: int) -> str:
    pub = (doc.get("publication_number") or "N/A").strip()
    sec = (doc.get("section") or "UNK").strip()
    return f"{pub}|{sec}|{idx}"


def _trim_result_for_log(doc: Dict) -> Dict:
    return {
        "doc_id": None,  # filled by caller
        "publication_number": doc.get("publication_number"),
        "section": doc.get("section"),
        "score": float(doc.get("score", 0.0) or 0.0),
        "preview": (doc.get("text", "") or "")[:160],
    }


def run_agentic_search(
    query: str,
    retriever: PatentRetriever,
    top_k: int = 10,
    max_passes: int = 4,
    min_score: float = 0.0,
    clarify_callback: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """
    Agentic multi-pass search using DSPy ReAct with the retriever exposed as a tool.

    Workflow
    - Determine intent; if unclear, call clarify tool (uses `clarify_callback` when provided)
    - Produce an enhanced query
    - Call the retriever via search tool; optionally read documents
    - If weak results, refine and repeat up to `max_passes`

    Returns a dict with:
      - final_results: List[str] (doc_ids)
      - search_passes: List[Dict] (per-pass logs)
      - clarifications: List[Dict] (questions/answers asked via clarify tool)
    """

    all_search_passes: List[Dict[str, Any]] = []
    clarifications: List[Dict[str, str]] = []
    corpus: Dict[str, str] = {}

    # Attempt DSPy ReAct with tools first; pass tools as plain functions for compatibility
    import dspy  # type: ignore

    # Tools as plain functions (names + docstrings are used by DSPy)
    def clarify(question: str) -> str:
        """Ask the user a short clarifying question; returns user's answer (may be empty)."""
        q = (question or "").strip()
        if not q:
            q = "Please provide any missing details (domain, constraints, key features)."
        if clarify_callback is not None:
            try:
                answer = (clarify_callback(q) or "").strip()
            except Exception:
                answer = ""
        else:
            try:
                prompt = (
                    "Provide a short note with likely missing details for a patent search based on this question. "
                    "Be conservative and avoid fabricating specifics. If unclear, return an empty string.\n\n"
                    f"User question: {query}\nClarification prompt: {q}\n"
                )
                answer = (get_LLM_response(prompt, max_tokens=256, temperature=0.2) or "").strip()
            except Exception:
                answer = ""
        clarifications.append({"question": q, "answer": answer})
        return answer or ""

    def enhance_query(base_query: str) -> str:
        """Refine a base search query using available clarifications; return only the refined string."""
        base = (base_query or query or "").strip()
        try:
            context_notes = "\n\n".join(
                [f"Q: {c['question']}\nA: {c['answer']}" for c in clarifications if c.get("answer")]
            )
            prompt = (
                "You improve patent prior-art search queries.\n"
                "Given a base query and optional clarifications, return a single refined search query string.\n"
                "Be specific, include key technical terms and synonyms, and avoid over-broad phrasing.\n"
                "Return ONLY the refined query string.\n\n"
                f"Base query: {base}\n\n"
                f"Clarifications (may be empty):\n{context_notes}\n"
            )
            refined = get_LLM_response(prompt, max_tokens=256, temperature=0.3)
            return (refined or "").strip().strip('"')
        except Exception:
            return base

    def search_patents(q: str) -> str:
        """Search patents via dense+BM25 hybrid. Input: query string. Returns JSON {doc_ids: [...]}"""
        qq = (q or "").strip()
        raw = _coerce_combined(retriever.search(qq, top_k=top_k, min_score=min_score))
        doc_ids: List[str] = []
        pass_log: List[Dict] = []
        for i, d in enumerate(raw):
            doc_id = _build_doc_id(d, i)
            corpus[doc_id] = d.get("text", "") or ""
            t = _trim_result_for_log(d)
            t["doc_id"] = doc_id
            pass_log.append(t)
            doc_ids.append(doc_id)
        all_search_passes.append({"query": qq, "results": pass_log})
        return json.dumps({"doc_ids": doc_ids}, ensure_ascii=False)

    def read_documents(raw_ids: str) -> str:
        """Read documents by IDs (JSON list or CSV) and return a JSON map of doc_id->text."""
        raw = (raw_ids or "").strip()
        ids: List[str] = []
        try:
            maybe = json.loads(raw)
            if isinstance(maybe, list):
                ids = [str(x) for x in maybe]
            else:
                ids = [s.strip() for s in raw.split(",") if s.strip()]
        except json.JSONDecodeError:
            ids = [s.strip() for s in raw.split(",") if s.strip()]
        docs = {doc_id: corpus.get(doc_id, "Not found") for doc_id in ids[: max(1, top_k)]}
        return json.dumps(docs, ensure_ascii=False)

    tools = [clarify, enhance_query, search_patents, read_documents]


    # Build ReAct agent with common constructor variants
    agent = dspy.ReAct(signature = PatentSearchSig, tools=tools, max_iters=max_passes)

    res = agent(question=query)

    # Parse final JSON from the agent's final output if present
    final_json: Dict[str, Any] = {}
    try:
        text = str(getattr(res, "answer", res))
        if "{" in text and "}" in text:
            js = text[text.find("{") : text.rfind("}") + 1]
            final_json = json.loads(js)
    except Exception:
        final_json = {}

    final_ids = list(final_json.get("final_top_results", []))
    if not final_ids and all_search_passes:
        latest = all_search_passes[-1].get("results", [])
        final_ids = [r.get("doc_id") for r in latest][:top_k]

    return {
        "final_results": final_ids,
        "search_passes": all_search_passes,
        "clarifications": clarifications,
    }


def agentic_search(
    query: str,
    retriever: PatentRetriever,
    top_k: int = 5,
    max_passes: int = 3,
    min_score: float = 0.0,
    clarify_callback: Optional[Callable[[str], str]] = None,
) -> List[Dict]:
    """
    Convenience wrapper that returns the final retrieved documents (combined list)
    instead of doc_ids, after running the agentic search workflow.
    """
    result = run_agentic_search(
        query=query,
        retriever=retriever,
        top_k=top_k,
        max_passes=max_passes,
        min_score=min_score,
        clarify_callback=clarify_callback,
    )

    # Collect the last pass documents directly from the retriever for the returned doc list
    # If no passes recorded, just execute once.
    last_query = None
    if result.get("search_passes"):
        last_query = result["search_passes"][-1].get("query")
    q = last_query or query
    docs = _coerce_combined(retriever.search(q, top_k=top_k, min_score=min_score))
    return docs


__all__ = [
    "run_agentic_search",
    "agentic_search",
]


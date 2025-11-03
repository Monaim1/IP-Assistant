import json
from typing import List, Dict, Optional, Tuple

from ip_assistant.retriever import PatentRetriever
from ip_assistant.utils import get_LLM_response


def _coerce_combined(results) -> List[Dict]:
    """Return the combined results list regardless of PatentRetriever variant."""
    if isinstance(results, tuple):
        return results[0]
    return results


def _summarize_results_for_tool(docs: List[Dict], max_chars_per_text: int = 500) -> str:
    """Return a compact JSON string suitable as a tool observation for ReAct."""
    summary = [
        {
            "publication_number": d.get("publication_number"),
            "section": d.get("section"),
            "score": float(d.get("score", 0.0) or 0.0),
            "text": (d.get("text", "") or "")[:max_chars_per_text],
        }
        for d in docs
    ]
    return json.dumps(summary, ensure_ascii=False)


def _is_sufficient(docs: List[Dict], required_min: int) -> bool:
    """Heuristic sufficiency check: enough hits and non-trivial text."""
    if len(docs) >= required_min:
        return True
    # If any doc has reasonably long text, consider sufficient
    for d in docs:
        if len((d.get("text") or "").strip()) > 400:
            return True
    return False


def agentic_search(
    query: str,
    retriever: PatentRetriever,
    top_k: int = 5,
    max_loops: int = 3,
    min_score: float = 0.0,
) -> List[Dict]:
    """
    Perform agentic, multi-hop retrieval using a DSPy ReAct agent with a search tool.

    Falls back to a simple LLM-guided refinement loop if DSPy isn't available or fails.

    Returns a flat list of retrieved documents (combined), suitable for building context.
    """

    required_min = max(1, min(3, top_k))
    last_docs: List[Dict] = []

    # 1) Try DSPy ReAct with a search tool backed by PatentRetriever
    try:
        import dspy  # type: ignore

        # Shared mutable capture to expose the latest retrieval back to caller
        state = {"last_docs": []}  # type: ignore[var-annotated]

        def patent_search_tool(q: str) -> str:
            docs = _coerce_combined(retriever.search(q, top_k=top_k, min_score=min_score))
            state["last_docs"] = docs
            return _summarize_results_for_tool(docs)

        # Build a Tool if available, else pass callable directly
        try:
            search_tool = dspy.Tool(
                name="patent_search",
                description=(
                    "Search patents via dense+BM25 hybrid. Return JSON array of {publication_number, section, score, text}."
                ),
                forward=patent_search_tool,
            )
            tools = [search_tool]
        except Exception:
            tools = [patent_search_tool]

        # Construct ReAct agent with bounded iterations
        try:
            agent = dspy.ReAct(
                tools=tools,
                max_iters=max_loops,
                instruction=(
                    "You are a patent search assistant. Use the patent_search tool to retrieve relevant snippets. "
                    "If results are insufficient or off-target, refine the query and search again. Stop within the given limit. "
                    "When done, output a concise synthesis and ensure you have gathered relevant snippets."
                ),
            )
        except TypeError:
            # Back-compat for alternative parameter naming
            agent = dspy.ReAct(tools=tools, max_turns=max_loops)

        # Drive the agent; it internally decides how many tool calls to make
        _ = agent(question=query)
        last_docs = list(state.get("last_docs") or [])

        if last_docs:
            return last_docs
        # If nothing came back, fall through to fallback loop
    except Exception:
        # DSPy not present or failed to run; fallback below
        pass

    # 2) Fallback: simple refinement loop driven by the existing LLM client
    cur_query = query
    for i in range(max_loops):
        docs = _coerce_combined(retriever.search(cur_query, top_k=top_k, min_score=min_score))
        if _is_sufficient(docs, required_min):
            return docs

        # Ask the LLM to refine the query for better patent retrieval
        refine_prompt = (
            "You are improving a patent prior-art search query. "
            "Given the original query and an empty or weak result set, produce a single refined query string that is more specific, uses synonyms, and targets core technical terms.\n\n"
            f"Original query: {cur_query}\n"
            "Return only the refined query without commentary."
        )
        try:
            refined = get_LLM_response(refine_prompt, max_tokens=256, temperature=0.3)
            refined = (refined or "").strip().strip('"')
            if refined and refined.lower() != cur_query.lower():
                cur_query = refined
            else:
                # No useful refinement; stop early
                break
        except Exception:
            break

    # If all else fails, return the last attempt (which may be empty)
    if last_docs:
        return last_docs
    return _coerce_combined(retriever.search(cur_query, top_k=top_k, min_score=min_score))

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


def run_search_agent(
    query: str,
    retriever: PatentRetriever,
    top_k: int = 10,
    max_passes: int = 4,
) -> Dict:
    """
    A DSPy ReAct search agent mirroring the Claude SDK example, using tools:
      - search: runs PatentRetriever and returns JSON with doc_ids
      - read_documents: returns JSON mapping doc_id -> text

    Returns {"final_results": [doc_id,...], "search_passes": [...]}
    Falls back to a simple multi-pass loop if DSPy isn't available.
    """

    all_search_passes: List[Dict] = []
    corpus: Dict[str, str] = {}

    # Attempt DSPy agent first
    try:
        import dspy  # type: ignore

        state = {"last_query": None}

        def search_tool(input_text: str) -> str:
            q = (input_text or "").strip()
            state["last_query"] = q
            raw = _coerce_combined(retriever.search(q, top_k=top_k))
            doc_ids: List[str] = []
            pass_log: List[Dict] = []
            for i, d in enumerate(raw):
                doc_id = _build_doc_id(d, i)
                corpus[doc_id] = d.get("text", "") or ""
                t = _trim_result_for_log(d)
                t["doc_id"] = doc_id
                pass_log.append(t)
                doc_ids.append(doc_id)
            all_search_passes.append({
                "query": q,
                "results": pass_log,
            })
            return json.dumps({"doc_ids": doc_ids}, ensure_ascii=False)

        def read_documents_tool(input_text: str) -> str:
            raw_ids = (input_text or "").strip()
            ids: List[str] = []
            try:
                maybe = json.loads(raw_ids)
                if isinstance(maybe, list):
                    ids = [str(x) for x in maybe]
                else:
                    ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
            except json.JSONDecodeError:
                ids = [s.strip() for s in raw_ids.split(",") if s.strip()]
            docs = {doc_id: corpus.get(doc_id, "Not found") for doc_id in ids[: max(1, top_k)]}
            return json.dumps(docs, ensure_ascii=False)

        # Wrap as DSPy Tools if available
        try:
            tool_search = dspy.Tool(
                name="search",
                description=f"Search patents. Input: query string. Returns JSON: {{doc_ids: [..]}}. top_k fixed at {top_k}.",
                forward=search_tool,
            )
            tool_read = dspy.Tool(
                name="read_documents",
                description="Read documents by IDs. Input: JSON list or CSV of doc_ids. Returns JSON map of doc_id->text.",
                forward=read_documents_tool,
            )
            tools = [tool_search, tool_read]
        except Exception:
            # Fall back to plain callables
            tools = [search_tool, read_documents_tool]

        instruction = (
            "You are a search agent whose only way to find documents is through tool calls.\n"
            "ALWAYS use search() to retrieve document IDs and read_documents() to inspect them.\n"
            "Never guess without using the tools.\n\n"
            f"WORKFLOW ({max_passes} passes):\n"
            f"1. PASS 1: Search with original query using search(query)\n"
            f"2. Read results with read_documents(doc_ids=[...])\n"
            f"3. PASS 2–{max_passes-1}: Refine query based on what was read\n"
            f"4. PASS {max_passes}: Final comprehensive search\n"
            f"5. Output JSON with top {top_k} document IDs\n\n"
            f"OUTPUT FORMAT:\n{{\"num_passes\": {max_passes}, \"final_top_results\": [\"id1\", \"id2\", ...]}}\n"
            "Only output the JSON object on the final message."
        )

        try:
            agent = dspy.ReAct(
                tools=tools,
                max_iters=max_passes,
                instruction=instruction,
            )
        except TypeError:
            agent = dspy.ReAct(tools=tools, max_turns=max_passes)

        _ = agent(question=query)

        # Try to parse a JSON object from the final scratchpad/output
        final_json = {}
        try:
            text = str(_.answer if hasattr(_, "answer") else _)
            if "{" in text and "}" in text:
                js = text[text.find("{") : text.rfind("}") + 1]
                final_json = json.loads(js)
        except Exception:
            final_json = {}

        final_results = list(final_json.get("final_top_results", []))
        if not final_results:
            # Fallback to latest doc_ids if agent didn't follow format
            if all_search_passes:
                latest = all_search_passes[-1].get("results", [])
                final_results = [r.get("doc_id") for r in latest][:top_k]

        return {
            "final_results": final_results,
            "search_passes": all_search_passes,
        }

    except Exception:
        # Fallback: simple multi-pass without DSPy
        cur_q = query
        for _i in range(max_passes):
            raw = _coerce_combined(retriever.search(cur_q, top_k=top_k))
            pass_log: List[Dict] = []
            doc_ids: List[str] = []
            for idx, d in enumerate(raw):
                doc_id = _build_doc_id(d, idx)
                corpus[doc_id] = d.get("text", "") or ""
                t = _trim_result_for_log(d)
                t["doc_id"] = doc_id
                pass_log.append(t)
                doc_ids.append(doc_id)
            all_search_passes.append({"query": cur_q, "results": pass_log})

            # Simple sufficiency check
            if _is_sufficient(raw, max(1, min(3, top_k))):
                break

            # Refine query
            refine_prompt = (
                "Improve a patent search query. Return only the refined query string.\n\n"
                f"Original query: {cur_q}\n"
            )
            try:
                refined = get_LLM_response(refine_prompt, max_tokens=128, temperature=0.3)
                refined = (refined or "").strip().strip('"')
                if refined and refined.lower() != cur_q.lower():
                    cur_q = refined
                else:
                    break
            except Exception:
                break

        final_ids: List[str] = []
        if all_search_passes:
            final_ids = [r.get("doc_id") for r in all_search_passes[-1].get("results", [])][:top_k]

        return {
            "final_results": final_ids,
            "search_passes": all_search_passes,
        }

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from ip_assistant.utils import get_LLM_response, stream_LLM_response, DEFAULT_SYSTEM_PROMPT
from ip_assistant.observability import RunLogger, summarize_chunks
from ip_assistant.retriever import PatentRetriever
from ip_assistant.agentic_search import agentic_search

load_dotenv()

app = FastAPI(
    title="IP Assistant API",
    description="API for IP Assistant - A Retrieval-Augmented Generation system for patent analysis",
    version="1.0.0"
)

retriever = None

@app.on_event("startup")
async def startup_event():
    """Initialize the retriever on startup."""
    global retriever
    try:
        milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
        milvus_port = os.getenv("MILVUS_PORT", "19530")
        retriever = PatentRetriever(host=milvus_host, port=milvus_port)
        print(f"✓ Connected to Milvus at {milvus_host}:{milvus_port}")
    except Exception as e:
        print(f"⚠ Warning: Could not initialize retriever: {str(e)}")
        print("  API will run in LLM-only mode without RAG capabilities")
        retriever = None

class QueryRequest(BaseModel):
    query: str
    # Default to Gemini; not using Ollama anymore
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    max_tokens: int = 2000
    temperature: float = 0.7
    top_k: int = 5
    use_rag: bool = True

class RetrievalResult(BaseModel):
    publication_number: Optional[str]
    text: str
    score: float
    section: Optional[str]
    decision: Optional[str]

class QueryResponse(BaseModel):
    response: str
    model: str
    tokens_used: int
    retrieved_documents: Optional[List[RetrievalResult]] = None

@app.get("/")
async def root():
    return {
        "message": "IP Assistant API is running",
        "endpoints": {
            "/query": "POST - Query the RAG system",
            "/search": "POST - Search patents without LLM",
            "/health": "GET - Health check"
        },
        "rag_enabled": retriever is not None
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "milvus_connected": retriever is not None}


@app.post("/search")
async def search_patents(query: str, top_k: int = 5):
    """Search for relevant patents without LLM generation."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized. Ensure Milvus is healthy and collection exists.")
    
    try:
        import time
        t0 = time.perf_counter()
        with RunLogger(run_name="search", tags={"endpoint": "/search"}) as run:
            run.log_params({"top_k": top_k})
            results = None
            with run.timeit("retrieval_latency"):
                results = agentic_search(query, retriever, top_k=top_k, max_passes=3)

            # agentic_search returns a flat combined list of docs
            combined = results
            run.log_json("retrieved.json", summarize_chunks(combined))
            run.log_text("query.txt", query)
            run.log_metrics({
                "chunks_retrieved": float(len(combined or [])),
                "request_latency_ms": (time.perf_counter() - t0) * 1000.0,
            })

        return {"query": query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_patents(request: QueryRequest):
    """Query the RAG system with optional retrieval."""
    try:
        import time
        t0 = time.perf_counter()
        with RunLogger(run_name="query", tags={"endpoint": "/query"}) as run:
            run.log_params({
                "model": request.model,
                "use_rag": request.use_rag,
                "top_k": request.top_k,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            })

            retrieved_docs = None
            context = None

            if request.use_rag and retriever is not None:
                try:
                    with run.timeit("retrieval_latency"):
                        results = agentic_search(request.query, retriever, top_k=request.top_k, max_passes=3)
                    combined = results
                    retrieved_docs = combined
                    run.log_metrics({"chunks_retrieved": float(len(combined or []))})
                    run.log_json("retrieved.json", summarize_chunks(combined))

                    if combined:
                        context_parts = []
                        for i, doc in enumerate(combined, 1):
                            context_parts.append(
                                f"Document {i} (Patent: {doc.get('publication_number', 'N/A')}, Score: {doc.get('score', 0):.3f}):\n{doc.get('text', '')}"
                            )
                        context = "\n\n".join(context_parts)
                except Exception as e:
                    print(f"Retrieval warning: {str(e)}")

            if context:
                prompt = (
                    "Using the retrieved patent documents below, assess the novelty and patentability of the user's idea.\n\n"
                    f"Retrieved Documents:\n{context}\n\n"
                    f"User Idea / Question: {request.query}\n\n"
                    "Provide a concise, structured assessment and cite any relevant patents."
                )
            else:
                prompt = f"Assess the novelty and patentability of the following idea:\n\n{request.query}"

            run.log_text("prompt.txt", prompt)
            run.log_text("system_prompt.txt", DEFAULT_SYSTEM_PROMPT)
            if context:
                run.log_text("context.txt", context)
            run.log_text("query.txt", request.query)

            with run.timeit("llm_latency"):
                response = get_LLM_response(
                    prompt=prompt,
                    model=request.model,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

            tokens_used = len(response.split())
            run.log_metrics({"tokens_used": float(tokens_used)})
            run.log_text("output.txt", response)

            out = {
                "response": response,
                "model": request.model,
                "tokens_used": tokens_used,
                "retrieved_documents": retrieved_docs if request.use_rag else None
            }
            # Log total request latency right before returning
            run.log_metrics({"request_latency_ms": (time.perf_counter() - t0) * 1000.0})
            return out
    except Exception as e:
        error_msg = str(e)
        if "Error getting AI response" in error_msg:
            error_msg = error_msg.split("Error getting AI response: ")[-1]
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/query/stream")
async def query_patents_stream(request: QueryRequest):
    """Stream the LLM answer tokens as they are generated."""
    try:
        import time
        t0 = time.perf_counter()
        with RunLogger(run_name="query_stream", tags={"endpoint": "/query/stream"}) as run:
            run.log_params({
                "model": request.model,
                "use_rag": request.use_rag,
                "top_k": request.top_k,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            })

            retrieved_docs = None
            context = None

            if request.use_rag and retriever is not None:
                try:
                    with run.timeit("retrieval_latency"):
                        results = agentic_search(request.query, retriever, top_k=request.top_k, max_passes=3)
                    combined = results
                    retrieved_docs = combined
                    run.log_metrics({"chunks_retrieved": float(len(combined or []))})
                    run.log_json("retrieved.json", summarize_chunks(combined))
                    if combined:
                        context_parts = []
                        for i, doc in enumerate(combined, 1):
                            context_parts.append(
                                f"Document {i} (Patent: {doc.get('publication_number', 'N/A')}, Score: {doc.get('score', 0):.3f}):\n{doc.get('text', '')}"
                            )
                        context = "\n\n".join(context_parts)
                except Exception:
                    pass

            if context:
                prompt = (
                    "Using the retrieved patent documents below, assess the novelty and patentability of the user's idea.\n\n"
                    f"Retrieved Documents:\n{context}\n\n"
                    f"User Idea / Question: {request.query}\n\n"
                    "Provide a concise, structured assessment and cite any relevant patents."
                )
            else:
                prompt = f"Assess the novelty and patentability of the following idea:\n\n{request.query}"

            run.log_text("prompt.txt", prompt)
            run.log_text("system_prompt.txt", DEFAULT_SYSTEM_PROMPT)
            if context:
                run.log_text("context.txt", context)
            run.log_text("query.txt", request.query)

            def token_generator():
                # Not true streaming under the hood; we still time LLM once
                chunks = []
                with run.timeit("llm_latency"):
                    for piece in stream_LLM_response(
                        prompt=prompt,
                        model=request.model,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                    ):
                        chunks.append(piece)
                        yield piece
                final_text = "".join(chunks)
                run.log_metrics({"tokens_used": float(len(final_text.split()))})
                run.log_text("output.txt", final_text)
                run.log_metrics({"request_latency_ms": (time.perf_counter() - t0) * 1000.0})

            return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
    except Exception as e:
        error_msg = str(e)
        if "Error getting AI response" in error_msg:
            error_msg = error_msg.split("Error getting AI response: ")[-1]
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

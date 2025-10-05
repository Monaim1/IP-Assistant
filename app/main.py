from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from utils import get_LLM_response

# Load environment variables
load_dotenv()

app = FastAPI(
    title="IP Assistant API",
    description="API for IP Assistant - A Retrieval-Augmented Generation system for patent analysis",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str
    model: str = "moonshotai/kimi-k2"
    max_tokens: int = 2000
    temperature: float = 0.7
    context: str = None

class QueryResponse(BaseModel):
    response: str
    model: str
    tokens_used: int

@app.get("/")
async def root():
    return {"message": "IP Assistant API is running. Use /query to interact with the model."}

@app.post("/query", response_model=QueryResponse)
async def query_patents(request: QueryRequest):
    try:
        response = get_LLM_response(
            prompt=request.query,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            context=request.context
        )
        return {
            "response": response,
            "model": request.model,
            "tokens_used": len(response.split())  # Approximate token count
        }
    except Exception as e:
        # Extract error message from the exception
        error_msg = str(e)
        if "Error getting AI response" in error_msg:
            error_msg = error_msg.split("Error getting AI response: ")[-1]
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

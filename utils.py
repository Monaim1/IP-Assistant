from openai import OpenAI
import os
import requests
from dotenv import load_dotenv

def get_LLM_client():
    """Initialize and return the OpenAI client with Ollama settings."""
    load_dotenv()
    return OpenAI(
        base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        api_key="ollama",  # API key is not used by Ollama
    )

def get_LLM_response(
    prompt: str,
    model: str = "qwen2.5:1.5b",  # Using qwen2.5:1.5b as the default model
    max_tokens: int = 32000,        # Reduced from 10000 as it's too high for most models
    temperature: float = 0.7,
    **kwargs
) -> str:
    try:
        client = get_LLM_client()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting AI response: {str(e)}")

if __name__ == "__main__":
    try:
        response = get_LLM_response("What is the meaning of life?")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

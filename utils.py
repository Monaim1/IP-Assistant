from openai import OpenAI
import os
from typing import Optional, Iterator
from dotenv import load_dotenv

def get_LLM_client():
    """Initialize and return the OpenAI client with Ollama settings."""
    load_dotenv()
    return OpenAI(
        base_url=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        api_key="ollama",  # API key is not used by Ollama
    )

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior patent analyst. Given a user-described idea and any "
    "retrieved patent snippets, provide a concise novelty and patentability "
    "assessment under US (35 U.S.C. §§102/103) and EPO (Art. 54/56) standards. "
    "Respond in clear sections:\n"
    "1) Idea understanding (bullets)\n"
    "2) Key technical features and differentiators\n"
    "3) Prior art signals from provided context (cite as [Patent: <pubno>])\n"
    "4) Novelty analysis (Section 102 / Art. 54)\n"
    "5) Non-obviousness / Inventive step (Section 103 / Art. 56)\n"
    "6) Risks, gaps, or enablement concerns\n"
    "7) Recommendations (claim angles, search next steps).\n"
    "Be specific and avoid generic statements. If context is insufficient, ask 3–5 focused clarifying questions."
)

def get_LLM_response(
    prompt: str,
    model: str = "qwen2.5:1.5b",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    try:
        client = get_LLM_client()
        system_msg = system_prompt or os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting AI response: {str(e)}")


def stream_LLM_response(
    prompt: str,
    model: str = "qwen2.5:1.5b",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Iterator[str]:
    """Yield model tokens incrementally using OpenAI-compatible streaming."""
    client = get_LLM_client()
    system_msg = system_prompt or os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content
            except Exception:
                # Be resilient to shape differences
                pass
    except Exception as e:
        # Propagate as a final message to the stream consumer
        yield f"\n[stream error] {str(e)}\n"

if __name__ == "__main__":
    try:
        response = get_LLM_response("What is the meaning of life?")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

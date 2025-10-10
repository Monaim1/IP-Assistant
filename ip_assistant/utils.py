import json
import os
from typing import Optional, Iterator
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from dotenv import load_dotenv

def get_LLM_client():

    load_dotenv()
    base_url = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment")
    return {"base_url": base_url, "api_key": api_key}

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
    "## Be specific and avoid generic statements."
    "## If no context is provided, state that no relevant prior art was found."
    "## IF user idea is missing or unclear, state that clearly. withougt making assumptions and going off on tangents, and ask for clarification."
    "## Use a formal tone appropriate for legal/technical analysis."
)

def get_LLM_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 30000,
    temperature: float = 0.95,
) -> str:
    try:
        cfg = get_LLM_client()

        if not str(model).startswith("gemini"):
            model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        url = f"{cfg['base_url']}/models/{model}:generateContent?{urlencode({'key': cfg['api_key']})}"

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": int(max_tokens),
            },
            "systemInstruction": {
                "parts": [{"text": DEFAULT_SYSTEM_PROMPT}],
            },
        }

        req = urlrequest.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlrequest.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except HTTPError as he:
            try:
                err_body = he.read().decode("utf-8")
            except Exception:
                err_body = str(he)
            raise Exception(f"Gemini HTTPError {he.code}: {err_body}")
        except URLError as ue:
            raise Exception(f"Gemini URLError: {ue.reason}")

        # Extract text from the first candidate
        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise Exception(f"No candidates in response: {data}")
            parts = candidates[0].get("content", {}).get("parts", [])
            text_segments = [p.get("text", "") for p in parts if isinstance(p, dict)]
            return "".join(text_segments).strip()
        except Exception as parse_err:
            raise Exception(f"Failed to parse Gemini response: {parse_err}")
    except Exception as e:
        raise Exception(f"Error getting AI response: {str(e)}")


def stream_LLM_response(
    prompt: str,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    system_prompt: Optional[str] = None,
    **kwargs
) -> Iterator[str]:

    try:
        full_text = get_LLM_response(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            **kwargs,
        )
        # Yield in reasonably sized chunks
        chunk_size = int(os.getenv("STREAM_CHUNK_SIZE", "256"))
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size]
    except Exception as e:
        yield f"\n[stream error] {str(e)}\n"

if __name__ == "__main__":
    try:
        response = get_LLM_response("What is the meaning of life?")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

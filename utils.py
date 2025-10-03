from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def get_LLM_response(
    prompt: str,
    model: str = "moonshotai/kimi-k2",
    max_tokens: int = 2000,
    temperature: float = 0.7,
    **kwargs
) -> str:

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error getting AI response: {str(e)}")


if __name__ == "__main__":
    # Example usage
    try:
        response = get_LLM_response("What is the meaning of life?")
        print(response)
    except Exception as e:
        print(f"Error: {e}")

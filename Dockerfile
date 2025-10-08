FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv & streamlit
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN pip install streamlit

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .


EXPOSE 8000

# Ensure system 'python' and 'pip' use the project venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Run the app
CMD ["uv", "run", "uvicorn", "ip_assistant.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

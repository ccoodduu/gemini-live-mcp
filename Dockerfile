FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    portaudio19-dev \
    libasound2-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml .

RUN uv pip install --system \
    "exceptiongroup>=1.2.2" \
    "google-genai>=1.8.0" \
    "mcp[cli]>=1.6.0" \
    "pillow>=11.1.0" \
    "pyaudio>=0.2.14" \
    "python-dotenv>=1.1.0" \
    "taskgroup>=0.2.2"

COPY main.py .
COPY mcp_handler.py .
COPY mcp_config.json .

CMD ["python", "main.py", "--mode", "none", "--text-only"]

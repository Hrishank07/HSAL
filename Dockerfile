FROM python:3.14-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hsal/ hsal/
COPY app.py .

# Chroma persistence lives on a volume (see docker-compose.yml)
ENV CHROMA_PATH=/data/chroma_db
ENV OLLAMA_HOST=http://ollama:11434

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

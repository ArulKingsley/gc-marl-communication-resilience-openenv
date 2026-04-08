FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source
COPY models.py   .
COPY env.py      .
COPY openenv.yaml .
COPY server/     ./server/

EXPOSE 8000

# Healthcheck — Docker will mark container unhealthy if this fails
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

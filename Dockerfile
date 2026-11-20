FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    SEED=7

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY airwatch ./airwatch
COPY static ./static

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request, os; urllib.request.urlopen('http://localhost:' + os.getenv('PORT', '8000') + '/healthz')" || exit 1

CMD ["sh", "-c", "uvicorn airwatch.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

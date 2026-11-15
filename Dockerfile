# Production API image for Render / Railway / any Docker host.
# Use a single uvicorn worker: SQLite is the default persistence layer.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    DATA_DIR=/var/data \
    MARKET_DATA_DB=/var/data/market.sqlite \
    PAPER_TRADING_DB=/var/data/paper.sqlite \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /var/data

EXPOSE 8000

# Render/Railway inject PORT. Keep workers=1 for SQLite safety.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]

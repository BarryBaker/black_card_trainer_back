FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tells Railway which port to route to when it does not inject PORT itself.
# Without this the edge has to guess the target port, and a wrong guess shows up
# as a 502 "Application failed to respond" even though the container is healthy.
EXPOSE 8080

# Railway injects PORT; 8080 is only the local fallback.
# One worker: each holds ~180 MB once pandas/numpy are warm.
CMD gunicorn run:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --access-logfile -

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    GUNICORN_CMD_ARGS="--bind=0.0.0.0:8001"

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY db.py run.py ./
COPY api/ api/
COPY features/ features/
COPY trees/ trees/

USER 10001:10001
EXPOSE 8001

CMD ["gunicorn", "--workers=1", "--threads=2", "--timeout=120", "--access-logfile=-", "run:app"]

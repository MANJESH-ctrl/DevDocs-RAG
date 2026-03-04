# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.in .
RUN pip install --no-cache-dir -r requirements.in

# NLTK after pip install — nltk package now exists
RUN python -c "\
import nltk; \
nltk.download('stopwords'); \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('averaged_perceptron_tagger')"

COPY warmup_models.py .
RUN python warmup_models.py

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache /root/.cache
COPY --from=builder /root/nltk_data /root/nltk_data

COPY . .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

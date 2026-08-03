FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/hybrid_model_best.zip \
    DEVICE=auto \
    OCR_LANGUAGES=en \
    EAGER_LOAD_MODEL=false \
    HF_HOME=/home/appuser/.cache/huggingface \
    EASYOCR_MODULE_PATH=/home/appuser/.EasyOCR

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements-demo.txt .
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
      --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-demo.txt && \
    useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser hybrid_model_best.zip ./hybrid_model_best.zip

EXPOSE 8000
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ReceiptGraph Explorer

The research dashboard exposes CORD samples, real-receipt upload, Hybrid graph
inspection and structured export:

```text
CORD image -> oracle words/boxes -> graph -> Hybrid inference
Uploaded image -> EasyOCR -> words/boxes -> graph -> Hybrid KIE
```

## Run locally

Python 3.12 is recommended. Keep `hybrid_model_best.zip` at the repository root.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-demo.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://127.0.0.1:8000>. The web UI starts immediately. The first model
run extracts and loads the checkpoint; uploaded images additionally lazy-load
EasyOCR. Later requests reuse the loaded components.

Three annotated CORD Dev receipts are bundled for presentation and Docker. If
the full `CORD1000/CORD/CORD` tree is present, the app automatically uses it.
Set `CORD_ROOT` to override discovery.

To warm the service before accepting traffic:

```powershell
$env:EAGER_LOAD_MODEL="true"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@receipt.png"
```

Operational endpoints:

- `GET /health/live`: process liveness.
- `GET /health/ready`: model/OCR readiness and load errors.
- `POST /api/v1/load`: explicitly warm the model.
- `POST /api/v1/extract`: JPEG, PNG or WebP receipt extraction.
- `POST /api/v1/analyze`: real-receipt analysis with one or more model modes.
- `GET /api/v1/samples`: discover local CORD research samples.
- `POST /api/v1/samples/{split}/{id}/analyze`: oracle-OCR Hybrid inference.
- `GET /api/v1/research-results`: frozen project metrics for the dashboard.
- `GET /docs`: OpenAPI documentation.

## Docker

```bash
docker build -t receiptgraph-kie .
docker run --rm -p 8000:8000 receiptgraph-kie
```

The provided image is a portable CPU build. For better throughput, deploy one process per GPU and install a CUDA-compatible PyTorch build in the image. Do not use multiple Uvicorn workers on one GPU because every worker loads another ~500 MB checkpoint and its own OCR/model state.

## Product constraints

- The checkpoint was evaluated with ground-truth CORD words and boxes. End-to-end accuracy with EasyOCR must be measured separately before production rollout.
- LayoutLMv3 accepts at most 512 tokens; responses expose `truncated=true` when OCR content is cut.
- `confidence` is the softmax score of the selected emission, not a calibrated CRF sequence probability. Calibrate thresholds on representative production receipts before automated decisions.
- The model predicts CORD fields, not merchant name, address or date. Extending the output schema requires labeled data and retraining.
- Pin and cache Hugging Face/EasyOCR assets for an offline or autoscaled production environment.
- Put authentication, TLS, request limits and rate limiting at an API gateway before exposing this service publicly.

## Research presentation flow

1. Select a Dev/Test CORD receipt and run the Original Hybrid graph.
2. Select a predicted word and inspect neighbors, relations, geometry and attention.
3. Inspect predicted labels, confidence, relation geometry and final-layer attention.
4. Export structured JSON/CSV and open **Research Results** for Hybrid metrics.

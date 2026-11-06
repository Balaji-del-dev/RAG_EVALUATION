# Enterprise RAG Evaluation Platform

Builds a RAG pipeline, then measures it. Upload documents, ask questions, and run test datasets and attack
suites through any configuration. Results across 37 metrics land in PostgreSQL, and a React dashboard compares
configurations side by side.

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md): what the system must do
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): how it fits together, and why
- [docs/METRICS.md](docs/METRICS.md): the 37 metrics and their formulas

## Stack

| Layer | Technology |
|-------|------------|
| Backend | Django 5.2, Django REST Framework |
| Frontend | React, Vite (built files are served by Django) |
| Database | PostgreSQL 17 |
| Vectors | Qdrant |
| BM25 | OpenSearch, or a built-in fallback for small corpora |
| Files | MinIO |
| Jobs and cache | Celery, Redis, Flower |
| Orchestration | LangGraph |
| LLM | LiteLLM to Ollama (`qwen2.5:1.5b` by default) |
| Embeddings, reranking, NLI | Sentence Transformers, cross-encoders (CPU) |
| Guards | Microsoft Presidio, rule-based injection and toxicity checks |
| Observability | OpenTelemetry, Langfuse (optional) |
| Testing | Pytest, Locust, Promptfoo, Garak |
| Evaluation | Built-in metrics; Ragas and DeepEval as optional second opinions |

## Prerequisites

Python 3.12, Node 20, Docker, and these services running locally:

| Service | Address | Notes |
|---------|---------|-------|
| Redis | 127.0.0.1:6379 | This project uses databases 8, 9 and 10 |
| Qdrant | 127.0.0.1:6333 | Collections are prefixed `rageval_` |
| Ollama | 127.0.0.1:11434 | Run `ollama pull qwen2.5:1.5b` and `ollama pull all-minilm` |

PostgreSQL and MinIO come from this project's `docker-compose.yml`, on ports 5434 and 9002.

## Setup

**1. Environment and secrets**

```bash
cp .env.example .env     # then replace every change_this value
```

**2. Python packages**

```bash
uv venv venv --python 3.12
uv pip install --python venv/bin/python --index-strategy unsafe-best-match -r requirements/lock.txt
# Or pip: python3 -m venv venv && venv/bin/pip install -r requirements/lock.txt
```

**3. Infrastructure and database**

```bash
docker compose up -d postgres minio
source venv/bin/activate
python manage.py migrate
```

**4. Sample data** (first run downloads about 400 MB of models)

```bash
python manage.py seed_demo
```

This creates the `admin`, `analyst` and `viewer` users, with passwords from `.env`. It also loads 7 sample
documents, 3 RAG configurations, a 20-question dataset and 24 security tests.

**5. Dashboard**

```bash
cd frontend && npm install && npm run build && cd ..
python manage.py collectstatic --no-input
```

**6. Run it**

```bash
scripts/start.sh         # server and worker, in the background. Logs are in logs/
scripts/stop.sh          # stops them
```

Open http://127.0.0.1:8010. They keep running after the terminal closes, until you stop them or reboot. To see
the logs in your terminal instead, use `scripts/dev.sh`, and press Ctrl+C to stop.

To work on the UI with live reload, also run `cd frontend && npm run dev`, then open http://127.0.0.1:5174.

After the first run has downloaded the models, set `HF_HUB_OFFLINE=1` in `.env`, so that loading a model never
waits on the network.

## Using it

1. **Documents:** upload files, or use the samples.
2. **Chat:** ask questions, continue with follow-ups, reopen past conversations, and inspect each retrieval stage.
3. **Runs:** pick a dataset and a configuration, then start a run.
4. **Results:** read Retrieval, Context, Answer, Hallucination, Security and Performance.
5. **Compare:** choose two or three runs to see what changed and which did better.
6. **Reports:** export results and see exactly what produced them.

## Roles

| Role | Can do |
|------|--------|
| Viewer | Ask questions, see documents they may read, see scores |
| Analyst | Also upload documents, create datasets, start runs, read per-question answers, export |
| Admin | Also read all documents and the audit log |

Analysts can read text from restricted documents through evaluation results, because runs read the whole corpus.

## Optional setup

### OpenSearch

The default `BM25_BACKEND=local` needs no service, but it reads every chunk per query. To use the OpenSearch
already running on this machine, create a login limited to `rageval_*` indexes:

```bash
OPENSEARCH_ADMIN_PASSWORD='...' python manage.py setup_opensearch
```

Put the two printed values in `.env`, restart, and new documents will index into OpenSearch. To rebuild the
existing index, use **Reprocess** on each document.

Alternatively, run a separate OpenSearch for this project, using about 600 MB more RAM:

```bash
docker compose --profile opensearch up -d opensearch
# In .env: BM25_BACKEND=opensearch, OPENSEARCH_URL=http://127.0.0.1:9201, OPENSEARCH_PASSWORD= (empty)
```

### Ragas and DeepEval

They run in their own environment, because they conflict with current `transformers`.

```bash
uv venv venv-frameworks --python 3.12
uv pip install --python venv-frameworks/bin/python -r requirements/evaluation.txt
```

Then tick them when starting a run. They judge with the LLM, at about a minute per question on a CPU.

### Langfuse

Create a project at http://127.0.0.1:3000, put its keys in `.env` as `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY`, and restart. Tracing is off until both are set.

## Commands

```bash
pytest -m "not integration"       # fast tests
pytest -m integration             # real models and Qdrant, no LLM
ruff check .

# End-to-end, against the running app, with the real LLM. Removes what it creates.
venv/bin/python tests/e2e/e2e_check.py       # every feature over HTTP (--quick skips Ragas and DeepEval)
/usr/bin/python3 tests/e2e/browser_check.py  # every page in a real browser (needs Playwright, outside the venv)

python manage.py seed_demo               # load the sample data
python manage.py setup_opensearch        # create the OpenSearch login
python manage.py export_metrics_doc      # regenerate docs/METRICS.md

celery -A config flower --port=5556      # task monitor

# Load test
LOCUST_USERNAME=viewer LOCUST_PASSWORD=... locust -f tests/load/locustfile.py \
  --host http://127.0.0.1:8010 --headless -u 5 -r 1 -t 60s

# External attack tools
eval "$(venv/bin/python scripts/api_session.py viewer)"
(cd security_tools/promptfoo && npx promptfoo@latest eval --no-cache --max-concurrency 1)
security_tools/garak/run_garak.sh
```

## If memory runs out

This was built on a 7.5 GB machine. A run needs about 2 GB free: about 1.1 GB for the LLM and 700 MB for the
worker's models.

`scripts/worker.sh` guards the machine. Below 200 MB free it stops the worker, and when memory recovers to 1000 MB
it starts it again, and the run continues from its last saved answer. The dashboard shows a warning while the worker
is stopped. Change the limits with `WORKER_MIN_FREE_MB` and `WORKER_RESUME_FREE_MB`. The server and worker are also
marked as first to be killed if memory runs out, so the kernel takes them before your editor or browser.

If the worker keeps stopping:

- Avoid the Chat during a run. The server loads its own 650 MB copy of the models on the first question, and
  keeps it until `scripts/stop.sh` and `scripts/start.sh`.
- Close browser tabs, or stop other services while the run finishes.
- Lower `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE` or `ML_THREADS` in `.env`.

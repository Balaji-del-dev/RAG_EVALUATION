"""Validated environment configuration. A missing or malformed value fails at startup, not mid-request."""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Env(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    DEBUG: bool = False
    SECRET_KEY: str
    ALLOWED_HOSTS: str = "localhost,127.0.0.1"
    CORS_ALLOWED_ORIGINS: str = ""
    CSRF_TRUSTED_ORIGINS: str = ""

    DATABASE_NAME: str
    DATABASE_USER: str
    DATABASE_PASSWORD: str
    DATABASE_HOST: str = "127.0.0.1"
    DATABASE_PORT: int = 5434

    CELERY_BROKER_URL: str = "redis://127.0.0.1:6379/8"
    CELERY_RESULT_BACKEND: str = "redis://127.0.0.1:6379/9"
    CACHE_URL: str = "redis://127.0.0.1:6379/10"

    MINIO_ENDPOINT: str = "127.0.0.1:9002"
    MINIO_ACCESS_KEY: str = ""
    MINIO_SECRET_KEY: str = ""
    MINIO_BUCKET: str = "rag-eval-documents"

    QDRANT_URL: str = "http://127.0.0.1:6333"

    BM25_BACKEND: str = "local"
    OPENSEARCH_URL: str = "https://127.0.0.1:9200"
    OPENSEARCH_USER: str = ""
    OPENSEARCH_PASSWORD: str = ""
    OPENSEARCH_VERIFY_CERTS: bool = False

    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_KEEP_ALIVE: str = "30s"
    LLM_TIMEOUT_SECONDS: int = 300

    # Size of the LLM's working memory, and how many CPU threads it and the local models may use.
    OLLAMA_NUM_CTX: int = 2048
    OLLAMA_NUM_THREAD: int = 4
    ML_THREADS: int = 4

    NLI_MODEL: str = "cross-encoder/nli-MiniLM2-L6-H768"
    # Set to 1 once the models are downloaded, so loading one never waits on the network.
    HF_HUB_OFFLINE: int = 0

    LANGFUSE_HOST: str = "http://127.0.0.1:3000"
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    OTEL_EXPORTER_OTLP_ENDPOINT: str = ""

    DEMO_ADMIN_PASSWORD: str = ""
    DEMO_USER_PASSWORD: str = ""


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


env = Env()

# These must be in the process environment before torch and huggingface_hub are imported.
os.environ.setdefault("HF_HUB_OFFLINE", str(env.HF_HUB_OFFLINE))
os.environ.setdefault("OMP_NUM_THREADS", str(env.ML_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

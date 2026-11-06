# Requirements – Enterprise RAG Evaluation Platform

The goal is not just a RAG chatbot. It is a platform that builds a RAG pipeline and then
measures it: quality, reliability, security and performance, with every result stored in SQL so
that configurations can be compared.

## Functional requirements

| ID | Module | Requirement | Implemented in |
|----|--------|-------------|----------------|
| FR01 | Documents | Upload PDF, DOCX, CSV (also TXT, MD) | `apps/documents/views.py`, `services/storage.py` |
| FR02 | Documents | Parse, clean and chunk documents | `services/parser.py`, `services/chunking.py` |
| FR03 | Embeddings | Generate vector embeddings | `services/embeddings.py` |
| FR04 | Retrieval | Vector and BM25 retrieval | `apps/retrieval/services/vector_search.py`, `bm25_search.py` |
| FR05 | Retrieval | Hybrid search, RRF, reranking, MMR | `hybrid_search.py`, `rrf.py`, `reranking.py`, `mmr.py` |
| FR06 | Chatbot | Generate answers using an LLM, with citations | `apps/chatbot/services/` |
| FR07 | Evaluation | Calculate the 37 evaluation metrics | `apps/evaluation/services/` |
| FR08 | Security | Detect PII and prompt injection | `apps/security/services/` |
| FR09 | Security | Enforce document-level authorization | `services/access_control.py`, filters in both retrievers |
| FR10 | Monitoring | Track latency, tokens and cost | `apps/monitoring/services/` |
| FR11 | Dashboard | Display evaluation scores | `frontend/` |
| FR12 | Reports | Compare evaluation experiments, export results | `apps/reports/services/` |
| FR13 | Background jobs | Run ingestion and evaluation asynchronously | Celery tasks in `documents` and `evaluation` |

## Non-functional requirements

| Requirement | How it is met |
|-------------|---------------|
| Scalability | Background jobs; retrieval and generation are separate services |
| Maintainability | Modular monolith: one Django app per responsibility, thin views, logic in `services/` |
| Security | Backend authorization before retrieval, secrets only in `.env`, guards on input and output, audit log |
| Performance | Redis cache for query embeddings, indexes on foreign keys and status columns, models loaded on first use |
| Reliability | Per-case error capture (one failed question never fails a whole run), task retries |
| Observability | Per-stage timings in SQL, OpenTelemetry spans, optional Langfuse traces |
| Reproducibility | Each run stores a snapshot: model, embedding, prompt, dataset and framework versions, retrieval parameters |

## The 37 metrics

| Stage | Metrics |
|-------|---------|
| Retrieval (6) | Precision@K, Recall@K, F1@K, Hit Rate@K, MRR, NDCG@K |
| Context (4) | Context Precision, Context Recall, Context Relevancy, Context Coverage |
| Answer (8) | Faithfulness, Answer Relevancy, Answer Correctness, Completeness, Semantic Similarity, Exact Match, BLEU, ROUGE |
| Reliability (5) | Hallucination Rate, Abstention Accuracy, Contradiction Rate, Citation Accuracy, Citation Completeness |
| Security (8) | PII Leakage, Sensitive Data Leakage, Access Control Violation, Prompt Injection Success, Jailbreak Success, Toxicity, Guard False Positive Rate, Guard False Negative Rate |
| Performance (6) | Retrieval Latency, End-to-End Latency, Token Consumption, Cost per Query, Throughput, Error Rate |

Formulas are stored in the `metric_definitions` table and listed in [METRICS.md](METRICS.md).

## Out of scope for this version

- Kubernetes and multi-node deployment.
- Multi-tenant isolation (document permissions are per user and group).
- Streaming answers.

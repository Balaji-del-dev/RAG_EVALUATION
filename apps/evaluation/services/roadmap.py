"""The 37 metrics in their recommended learning order. This is the source of truth for each metric's name,
position, priority and plain-language question. The formulas live with the code that computes them."""

from apps.common.constants import MetricCategory

STAGES = [
    (MetricCategory.RETRIEVAL, "Stage 1: Retrieval", "Does the retriever find the correct documents?"),
    (MetricCategory.CONTEXT, "Stage 2: Context", "Is the retrieved context relevant and sufficient for the LLM?"),
    (MetricCategory.ANSWER, "Stage 3: Answer generation", "Does the LLM generate a correct, relevant and grounded answer?"),
    (MetricCategory.RELIABILITY, "Stage 4: Hallucination and reliability", "Which answers are unsupported or incorrect?"),
    (MetricCategory.SECURITY, "Stage 5: Security and guardrails", "Does it expose sensitive information or follow malicious instructions?"),
    (MetricCategory.PERFORMANCE, "Stage 6: Performance and cost", "How efficient is the pipeline in production?"),
]

HIGHLY_CRITICAL = "Highly Critical"
CRITICAL = "Critical"
HIGH = "High"
LOW = "Low"

# Name in the code, display name, priority, what you need to understand.
ROADMAP = [
    ("precision_at_k", "Precision@K", CRITICAL, "Out of retrieved chunks, how many are relevant?"),
    ("recall_at_k", "Recall@K", HIGHLY_CRITICAL, "How many relevant chunks did retrieval find?"),
    ("f1_at_k", "F1 Score", HIGH, "Balance between precision and recall."),
    ("hit_rate_at_k", "Hit Rate@K", HIGH, "Did retrieval find at least one relevant chunk?"),
    ("mrr", "MRR (Mean Reciprocal Rank)", CRITICAL, "How early does the first relevant chunk appear?"),
    ("ndcg_at_k", "NDCG@K", CRITICAL, "Are the most relevant chunks ranked higher?"),
    ("context_precision", "Context Precision", HIGHLY_CRITICAL, "Are relevant chunks ranked at the top?"),
    ("context_recall", "Context Recall", HIGHLY_CRITICAL, "Does retrieved context contain all required information?"),
    ("context_relevancy", "Context Relevancy", CRITICAL, "Is the retrieved context relevant to the question?"),
    ("context_coverage", "Context Coverage", HIGH, "Does the context cover all important aspects of the query?"),
    ("faithfulness", "Faithfulness", HIGHLY_CRITICAL, "Is the answer supported by retrieved context?"),
    ("answer_relevancy", "Answer Relevancy", HIGHLY_CRITICAL, "Does the answer address the user's question?"),
    ("answer_correctness", "Answer Correctness", HIGHLY_CRITICAL, "Is the answer factually correct compared with the reference?"),
    ("completeness", "Answer Completeness", HIGH, "Does the answer include all required information?"),
    ("semantic_similarity", "Semantic Similarity", HIGH, "Does the answer have a similar meaning to the reference?"),
    ("exact_match", "Exact Match (EM)", LOW, "Does the answer exactly match the expected answer?"),
    ("bleu", "BLEU", LOW, "How much n-gram precision overlaps with reference text?"),
    ("rouge", "ROUGE", LOW, "How much reference text is covered by generated text?"),
    ("hallucination_rate", "Hallucination Rate", HIGHLY_CRITICAL, "How frequently does the LLM generate unsupported information?"),
    ("abstention_accuracy", "Abstention Accuracy", CRITICAL, "Does the system refuse when sufficient evidence is unavailable?"),
    ("contradiction_rate", "Contradiction Detection", CRITICAL, "Does the answer contradict the retrieved documents?"),
    ("citation_accuracy", "Citation Accuracy", HIGH, "Do cited sources actually support the answer?"),
    ("citation_completeness", "Citation Completeness", HIGH, "Are all important claims properly supported by citations?"),
    ("pii_leakage_rate", "PII Leakage Rate", HIGHLY_CRITICAL, "Does the response expose personal information?"),
    ("sensitive_data_leakage_rate", "Sensitive Data Leakage Rate", HIGHLY_CRITICAL, "Does it expose confidential enterprise data?"),
    ("access_control_violation_rate", "Access Control Violation Rate", HIGHLY_CRITICAL, "Can users retrieve unauthorized documents?"),
    ("prompt_injection_success_rate", "Prompt Injection Success Rate", HIGHLY_CRITICAL, "Can malicious instructions manipulate the system?"),
    ("toxicity_rate", "Toxicity Rate", CRITICAL, "Does the response contain harmful content?"),
    ("jailbreak_success_rate", "Jailbreak Success Rate", CRITICAL, "Can users bypass model safety restrictions?"),
    ("guard_false_positive_rate", "False Positive Rate", HIGH, "Are legitimate questions incorrectly blocked?"),
    ("guard_false_negative_rate", "False Negative Rate", HIGH, "Are unsafe requests incorrectly allowed?"),
    ("retrieval_latency", "Retrieval Latency", CRITICAL, "How long does retrieval take?"),
    ("e2e_latency", "End-to-End Latency", CRITICAL, "How long does the complete RAG pipeline take?"),
    ("token_consumption", "Token Consumption", HIGH, "How many tokens are consumed per request?"),
    ("cost_per_query", "Cost per Query", HIGH, "How much does each query cost?"),
    ("throughput", "Throughput", HIGH, "How many requests can the system handle?"),
    ("error_rate", "Error Rate", HIGH, "How frequently do pipeline requests fail?"),
]

from django.conf import settings
from django.db import models

from apps.common.models import TimeStampedModel


class ChatSession(TimeStampedModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chat_sessions")
    title = models.CharField(max_length=255, blank=True)
    rag_config = models.ForeignKey("retrieval.RagConfig", null=True, on_delete=models.SET_NULL)

    class Meta:
        db_table = "chat_sessions"
        ordering = ["-created_at"]


class ChatMessage(TimeStampedModel):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    question = models.TextField()
    standalone_question = models.TextField(blank=True, help_text="A follow-up rewritten as a complete question")
    answer = models.TextField(blank=True)
    citations = models.JSONField(default=list, blank=True)
    abstained = models.BooleanField(default=False)
    decline_reason = models.TextField(blank=True)
    quoted = models.BooleanField(default=False, help_text="Quoted from the sources, because the model declined")
    blocked = models.BooleanField(default=False)
    block_reason = models.CharField(max_length=255, blank=True)
    guard_findings = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "chat_messages"
        ordering = ["created_at"]


class LLMRun(models.Model):
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name="llm_runs")
    model_name = models.CharField(max_length=255)
    prompt_version = models.CharField(max_length=16)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    latency_ms = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        db_table = "llm_runs"
        ordering = ["-created_at"]

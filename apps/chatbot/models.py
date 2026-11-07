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
    answer_format = models.CharField(max_length=40, blank=True, help_text="The layout asked for: a key in formats.py")
    answer = models.TextField(blank=True)
    citations = models.JSONField(default=list, blank=True)
    abstained = models.BooleanField(default=False)
    decline_reason = models.TextField(blank=True)
    quoted = models.BooleanField(default=False, help_text="Quoted from the sources, because the model declined")
    blocked = models.BooleanField(default=False)
    block_reason = models.CharField(max_length=255, blank=True)
    guard_findings = models.JSONField(default=dict, blank=True)
    request_id = models.CharField(max_length=64, blank=True, db_index=True)
    # What the reader should double-check: figures not in the sources, citations to sources it was not given.
    checks = models.JSONField(default=dict, blank=True)
    cached = models.BooleanField(default=False, help_text="Answered from the answer cache")
    # search, library, conversation, clarify, reply or blocked: see routing.py.
    route = models.CharField(max_length=16, blank=True, default="search")

    class Meta:
        db_table = "chat_messages"
        ordering = ["created_at"]


class AnswerFeedback(TimeStampedModel):
    """A thumbs up or down from the person who asked. Low-rated answers are the place to look for new test cases."""

    class Rating(models.TextChoices):
        UP = "up"
        DOWN = "down"

    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name="feedback")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="answer_feedback")
    rating = models.CharField(max_length=4, choices=Rating.choices)
    comment = models.TextField(blank=True)

    class Meta:
        db_table = "answer_feedback"
        constraints = [models.UniqueConstraint(fields=["message", "user"], name="one_rating_per_user_per_answer")]


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

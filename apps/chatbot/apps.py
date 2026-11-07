from django.apps import AppConfig


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.chatbot"

    def ready(self):
        from django.db.models.signals import post_delete, post_save

        from apps.chatbot.services import answer_cache
        from apps.documents.models import Document
        from apps.retrieval.models import RagConfig

        # Any change to a document or configuration may change answers, so cached ones stop matching.
        for model in (Document, RagConfig):
            post_save.connect(answer_cache.invalidate, sender=model, dispatch_uid=f"answer-cache-{model.__name__}-save")
            post_delete.connect(
                answer_cache.invalidate, sender=model, dispatch_uid=f"answer-cache-{model.__name__}-delete"
            )

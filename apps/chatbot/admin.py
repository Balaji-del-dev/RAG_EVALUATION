from django.contrib import admin

from apps.chatbot.models import ChatMessage, ChatSession, LLMRun

admin.site.register(ChatSession)
admin.site.register(LLMRun)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ["id", "question", "abstained", "blocked", "created_at"]
    list_filter = ["abstained", "blocked"]

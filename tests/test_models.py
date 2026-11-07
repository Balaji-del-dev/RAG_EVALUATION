"""Choosing a local model in the chat: what is offered, what is refused, and that only the one question changes."""

from types import SimpleNamespace

import pytest
from rest_framework.test import APIClient

from apps.chatbot.services import local_models
from apps.retrieval.models import RagConfig

GB = 1000**3
TAGS = [
    {"name": "qwen2.5:1.5b", "size": int(0.99 * GB), "details": {"family": "qwen2", "parameter_size": "1.5B"}},
    {"name": "qwen2.5:3b", "size": int(1.93 * GB), "details": {"family": "qwen2", "parameter_size": "3.1B"}},
    {"name": "llama3:latest", "size": int(4.66 * GB), "details": {"family": "llama", "parameter_size": "8.0B"}},
    {"name": "all-minilm:latest", "size": int(0.05 * GB), "details": {"family": "bert", "parameter_size": "23M"}},
]


@pytest.fixture
def ollama(monkeypatch):
    """Ollama with four models, qwen2.5:1.5b loaded, and 2.2 GB of memory free besides it."""
    state = {"running": [{"name": "qwen2.5:1.5b", "size": int(1.2 * GB)}], "free": int(2.2 * GB)}

    def get(url, timeout):
        body = {"models": TAGS} if url.endswith("/api/tags") else {"models": state["running"]}
        return SimpleNamespace(json=lambda: body)

    monkeypatch.setattr(local_models.httpx, "get", get)
    monkeypatch.setattr(local_models, "available_bytes", lambda: state["free"])
    return state


def test_models_are_listed_with_what_fits(ollama):
    models, free = local_models.list_models()
    by_name = {model.name: model for model in models}
    assert free == int(2.2 * GB) + int(1.2 * GB)  # the loaded model counts: Ollama unloads it to load another
    assert by_name["qwen2.5:1.5b"].loaded and by_name["qwen2.5:1.5b"].fits
    assert by_name["qwen2.5:3b"].fits  # about 2.5 GB needed
    assert not by_name["llama3:latest"].fits  # about 5.5 GB needed
    assert not by_name["all-minilm:latest"].can_chat
    assert [model.name for model in models][-1] == "all-minilm:latest"  # chat models first


def test_a_model_that_does_not_fit_is_refused_with_the_numbers(ollama):
    ollama["free"], ollama["running"] = int(2.0 * GB), []
    assert "qwen2.5:3b needs about 2.6 GB of memory and 2.0 GB is free" in local_models.why_not(
        "ollama_chat/qwen2.5:3b"
    )
    assert local_models.why_not("ollama_chat/qwen2.5:1.5b") == ""


def test_only_installed_chat_models_are_accepted(ollama):
    assert "not installed" in local_models.why_not("openai/gpt-4o")
    assert "cannot answer questions" in local_models.why_not("ollama_chat/all-minilm:latest")


@pytest.fixture
def viewer(django_user_model):
    user = django_user_model.objects.create_user("v", password="pw-v-12345", role="viewer")
    client = APIClient()
    client.force_authenticate(user)
    return client


@pytest.mark.django_db
def test_the_models_endpoint(ollama, viewer):
    RagConfig.objects.create(name="Default", is_default=True, llm_model="ollama_chat/qwen2.5:1.5b")
    data = viewer.get("/api/rag/models/").json()
    assert data["default"] == "ollama_chat/qwen2.5:1.5b"
    assert [model["name"] for model in data["models"] if model["fits"] and model["can_chat"]] == [
        "qwen2.5:1.5b",
        "qwen2.5:3b",
    ]


@pytest.mark.django_db
def test_a_chosen_model_answers_this_question_only(ollama, viewer, monkeypatch):
    from apps.chatbot import views

    config = RagConfig.objects.create(name="Default", is_default=True, llm_model="ollama_chat/qwen2.5:1.5b")
    used = {}

    def fake_ask(user, question, chosen, session, **kwargs):
        used["model"] = chosen.llm_model
        raise views.ServiceUnavailable("stop here")

    monkeypatch.setattr(views, "ask", fake_ask)
    viewer.post("/api/rag/query/", {"question": "Q?", "llm_model": "ollama_chat/qwen2.5:3b"}, format="json")
    assert used["model"] == "ollama_chat/qwen2.5:3b"
    config.refresh_from_db()
    assert config.llm_model == "ollama_chat/qwen2.5:1.5b"  # the configuration is unchanged


@pytest.mark.django_db
def test_a_model_too_big_for_memory_is_refused_before_anything_runs(ollama, viewer):
    RagConfig.objects.create(name="Default", is_default=True)
    response = viewer.post(
        "/api/rag/query/", {"question": "Q?", "llm_model": "ollama_chat/llama3:latest"}, format="json"
    )
    assert response.status_code == 400
    assert "llama3:latest needs about" in response.json()["error"]

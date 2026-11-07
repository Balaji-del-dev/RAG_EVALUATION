"""Batch 4 of the production pipeline: reviewing chat answers and turning them into test cases."""

from types import SimpleNamespace

import pytest
from rest_framework.test import APIClient

from apps.chatbot.models import AnswerFeedback, ChatMessage, ChatSession
from apps.common.constants import Role
from apps.evaluation.models import EvaluationDataset
from apps.monitoring.models import PerformanceLog
from apps.retrieval.models import RagConfig
from apps.security.models import AuditLog

pytestmark = pytest.mark.django_db


@pytest.fixture
def chats(django_user_model):
    make = django_user_model.objects.create_user
    people = SimpleNamespace(
        viewer=make("viewer", password="pw-v-12345", role=Role.VIEWER),
        analyst=make("analyst", password="pw-a-12345", role=Role.ANALYST),
    )
    session = ChatSession.objects.create(user=people.viewer, title="t", rag_config=RagConfig.objects.create(name="c"))

    def message(question, **fields):
        succeeded = fields.pop("succeeded", True)
        created = ChatMessage.objects.create(session=session, question=question, **fields)
        PerformanceLog.objects.create(message=created, succeeded=succeeded)
        return created

    people.good = message("How many earned leave days?", answer="18 days [S1].")
    people.down = message("sick leave?", answer="10 days.", standalone_question="How many sick leave days at VVDN?")
    AnswerFeedback.objects.create(message=people.down, user=people.viewer, rating="down", comment="Wrong figures: 12")
    people.flagged = message("annual leave?", answer="5 days [S1].", checks={"numbers": ["5"]})
    people.noted = message("notice period?", answer="30 days.", checks={"retrieval": ["widened to 24 candidates"]})
    people.declined = message(
        "CEO salary?", answer="I don't know.", abstained=True, decline_reason="Not in the documents"
    )
    people.failed = message("leave policy?", succeeded=False)
    return people


def _client(user):
    client = APIClient()
    client.force_authenticate(user)
    return client


def test_viewers_cannot_review_other_peoples_answers(chats):
    assert _client(chats.viewer).get("/api/rag/review/").status_code == 403
    assert _client(chats.viewer).post("/api/rag/review/dataset/", {}, format="json").status_code == 403


def test_the_queue_lists_answers_that_need_a_second_look(chats):
    body = _client(chats.analyst).get("/api/rag/review/").json()
    listed = {item["id"]: item for item in body["results"]}
    # A good answer, and one whose only note is how the search worked, do not need review.
    assert set(listed) == {chats.down.id, chats.flagged.id, chats.declined.id, chats.failed.id}
    assert body["counts"] == {"thumbs_down": 1, "flagged": 1, "declined": 1, "failed": 1}
    down = listed[chats.down.id]
    assert down["tags"] == ["thumbs_down"] and down["feedback"] == {"rating": "down", "comment": "Wrong figures: 12"}
    assert listed[chats.flagged.id]["checks"] == {"numbers": ["5"]}
    assert listed[chats.declined.id]["decline_reason"] == "Not in the documents"
    assert listed[chats.failed.id]["tags"] == ["failed"]
    assert "viewer" not in str(body)  # who asked is not shown


@pytest.mark.parametrize("kind,expected", [("thumbs_down", "down"), ("flagged", "flagged"), ("failed", "failed")])
def test_the_queue_filters_by_kind(chats, kind, expected):
    body = _client(chats.analyst).get(f"/api/rag/review/?kind={kind}").json()
    assert [item["id"] for item in body["results"]] == [getattr(chats, expected).id]


def test_reviewed_answers_become_a_new_dataset_version(chats):
    client = _client(chats.analyst)
    items = [
        {"message_id": chats.down.id, "reference_answer": "Sick leave is 12 days a year."},
        {"message_id": chats.declined.id, "answerable": False},
    ]
    first = client.post("/api/rag/review/dataset/", {"name": "Chat review", "items": items}, format="json")
    assert first.status_code == 201 and first.json()["version"] == "1" and first.json()["case_count"] == 2
    second = client.post("/api/rag/review/dataset/", {"name": "Chat review", "items": items[:1]}, format="json")
    assert second.json()["version"] == "2"  # datasets never change; a new version is made instead
    dataset = EvaluationDataset.objects.get(name="Chat review", version="1")
    cases = list(dataset.cases.values_list("question", "reference_answer", "answerable"))
    # The follow-up "sick leave?" is saved as the full question it stood for.
    assert cases == [
        ("How many sick leave days at VVDN?", "Sick leave is 12 days a year.", True),
        ("CEO salary?", "", False),
    ]
    assert dataset.created_by == chats.analyst and "2 reviewed chat answers" in dataset.description
    assert AuditLog.objects.filter(action="dataset_from_review", target=f"dataset:{dataset.id}").exists()


def test_a_case_needs_a_correct_answer_unless_it_should_decline(chats):
    client = _client(chats.analyst)
    missing = client.post(
        "/api/rag/review/dataset/", {"name": "R", "items": [{"message_id": chats.flagged.id}]}, format="json"
    )
    assert missing.status_code == 400 and "reference_answer" in str(missing.json())
    unknown = client.post(
        "/api/rag/review/dataset/",
        {"name": "R", "items": [{"message_id": 999999, "reference_answer": "x"}]},
        format="json",
    )
    assert unknown.status_code == 400 and "999999" in str(unknown.json())
    assert not EvaluationDataset.objects.filter(name="R").exists()

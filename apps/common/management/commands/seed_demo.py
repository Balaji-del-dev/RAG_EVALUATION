"""Load the sample corpus, users, configurations, dataset and security tests. Safe to run repeatedly."""

import json
import mimetypes
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from apps.common.constants import Role
from apps.documents.models import Document
from apps.documents.services.ingestion import create_document, process_document
from apps.evaluation.models import EvaluationDataset
from apps.evaluation.services import dataset_manager
from apps.evaluation.services.run_executor import sync_metric_definitions
from apps.retrieval.models import RagConfig
from apps.security.models import SecurityTest
from config.env import env

SAMPLE_DIR = Path(settings.BASE_DIR) / "sample_data"


class Command(BaseCommand):
    help = "Create demo users, sample documents, RAG configurations, an evaluation dataset and security tests."

    def add_arguments(self, parser):
        parser.add_argument("--no-documents", action="store_true", help="Skip document ingestion (no ML models load)")

    def handle(self, *args, **options):
        manifest = json.loads((SAMPLE_DIR / "manifest.json").read_text())
        users = self._users()
        self._configs(manifest["rag_configs"])
        sync_metric_definitions()
        self.stdout.write("Metric definitions synced")
        self._dataset(users["analyst"])
        self._security_tests()
        if not options["no_documents"]:
            self._documents(manifest["documents"], users["analyst"])
        self._pin_corpus(manifest["documents"])
        self.stdout.write(self.style.SUCCESS("Demo data ready"))

    def _pin_corpus(self, documents: list[dict]) -> None:
        """Tie the sample dataset to the sample documents, so that later uploads cannot change its scores."""
        raw = json.loads((SAMPLE_DIR / "eval_dataset.json").read_text())
        dataset = EvaluationDataset.objects.get(name=raw["name"], version=raw["version"])
        dataset.documents.set(Document.objects.filter(filename__in=[entry["file"] for entry in documents]))

    def _users(self) -> dict:
        if not (env.DEMO_ADMIN_PASSWORD and env.DEMO_USER_PASSWORD):
            raise CommandError("Set DEMO_ADMIN_PASSWORD and DEMO_USER_PASSWORD in .env")
        User = get_user_model()
        users = {}
        for username, role, password, staff in (
            ("admin", Role.ADMIN, env.DEMO_ADMIN_PASSWORD, True),
            ("analyst", Role.ANALYST, env.DEMO_USER_PASSWORD, False),
            ("viewer", Role.VIEWER, env.DEMO_USER_PASSWORD, False),
        ):
            user, created = User.objects.get_or_create(
                username=username, defaults={"role": role, "is_staff": staff, "is_superuser": staff}
            )
            if created:
                user.set_password(password)
                user.save()
                self.stdout.write(f"Created user {username} ({role})")
            users[username] = user
        return users

    def _configs(self, configs: list[dict]) -> None:
        for raw in configs:
            data = dict(raw)
            # The manifest is the source of truth. Past runs keep their own settings in their snapshots.
            _, created = RagConfig.objects.update_or_create(name=data.pop("name"), defaults=data)
            if created:
                self.stdout.write(f"Created RAG config {raw['name']}")

    def _dataset(self, user) -> None:
        raw = json.loads((SAMPLE_DIR / "eval_dataset.json").read_text())
        if not EvaluationDataset.objects.filter(name=raw["name"], version=raw["version"]).exists():
            dataset_manager.create_dataset(
                name=raw["name"], version=raw["version"], description=raw["description"], cases=raw["cases"], user=user
            )
            self.stdout.write(f"Created dataset with {len(raw['cases'])} cases")

    def _security_tests(self) -> None:
        tests = json.loads((SAMPLE_DIR / "security_tests.json").read_text())
        for raw in tests:
            data = dict(raw)
            SecurityTest.objects.update_or_create(test_name=data.pop("test_name"), defaults=data)
        self.stdout.write(f"Synced {len(tests)} security tests")

    def _documents(self, documents: list[dict], owner) -> None:
        for entry in documents:
            path = SAMPLE_DIR / "documents" / entry["file"]
            existing = Document.objects.filter(filename=entry["file"]).first()
            if existing and existing.status == Document.Status.READY:
                continue
            document = existing or create_document(
                data=path.read_bytes(),
                filename=entry["file"],
                content_type=mimetypes.guess_type(entry["file"])[0] or "text/plain",
                owner=owner,
                visibility=entry["visibility"],
            )
            process_document(document.id)
            self.stdout.write(f"Ingested {entry['file']} ({document.chunks.count()} chunks, {entry['visibility']})")

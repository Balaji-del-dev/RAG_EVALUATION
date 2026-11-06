"""Create a least-privilege OpenSearch login for this project, limited to its own rageval_* indexes.

Usage: OPENSEARCH_ADMIN_PASSWORD=... python manage.py setup_opensearch
It prints the new password once. Copy it into .env as OPENSEARCH_PASSWORD and set BM25_BACKEND=opensearch.
"""

import json
import os
import secrets
import ssl
import urllib.error
import urllib.request

from django.core.management.base import BaseCommand, CommandError

from config.env import env

ROLE = "rag_eval_role"
ROLE_BODY = {
    "cluster_permissions": ["cluster_monitor", "indices:data/write/bulk"],
    "index_permissions": [{"index_patterns": ["rageval_*"], "allowed_actions": ["indices_all"]}],
}


class Command(BaseCommand):
    help = "Create the rag_eval OpenSearch user and role. Needs OPENSEARCH_ADMIN_PASSWORD in the environment."

    def handle(self, *args, **options):
        admin_password = os.environ.get("OPENSEARCH_ADMIN_PASSWORD")
        if not admin_password:
            raise CommandError("Set OPENSEARCH_ADMIN_PASSWORD in the environment for this one command")
        password = secrets.token_urlsafe(24)
        self._put(f"roles/{ROLE}", ROLE_BODY, admin_password)
        self._put(f"internalusers/{env.OPENSEARCH_USER}", {"password": password, "backend_roles": []}, admin_password)
        self._put(f"rolesmapping/{ROLE}", {"users": [env.OPENSEARCH_USER]}, admin_password)
        self.stdout.write(self.style.SUCCESS(f"Created OpenSearch user {env.OPENSEARCH_USER}."))
        self.stdout.write("Add these to .env, then restart the server and worker:")
        self.stdout.write(f"OPENSEARCH_PASSWORD={password}\nBM25_BACKEND=opensearch")

    def _put(self, path: str, body: dict, admin_password: str) -> None:
        import base64

        request = urllib.request.Request(
            f"{env.OPENSEARCH_URL}/_plugins/_security/api/{path}",
            data=json.dumps(body).encode(),
            method="PUT",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Basic " + base64.b64encode(f"admin:{admin_password}".encode()).decode(),
            },
        )
        context = ssl.create_default_context()
        if not env.OPENSEARCH_VERIFY_CERTS:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(request, context=context, timeout=15):
                pass
        except urllib.error.HTTPError as exc:
            raise CommandError(f"OpenSearch rejected {path}: {exc.code} {exc.read().decode()[:300]}") from exc
        except urllib.error.URLError as exc:
            raise CommandError(f"Could not reach OpenSearch at {env.OPENSEARCH_URL}: {exc.reason}") from exc

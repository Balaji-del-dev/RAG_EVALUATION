"""Load test for throughput and error rate under concurrent users.

    LOCUST_USERNAME=viewer LOCUST_PASSWORD=... locust -f tests/load/locustfile.py \
        --host http://127.0.0.1:8010 --headless -u 5 -r 1 -t 60s

Targets retrieval by default, which is what can actually be loaded on a CPU-only machine. Set
LOCUST_INCLUDE_LLM=1 to add full questions: the LLM handles one at a time, so these queue up quickly.
"""

import os
import random

from locust import HttpUser, between, task

QUESTIONS = [
    "How many days of annual leave do employees get?",
    "What is the minimum password length?",
    "How long are backups retained?",
    "What is the rate limit of the management API?",
    "What is the warranty on the router?",
]


class RagUser(HttpUser):
    wait_time = between(0.2, 1.0)

    def on_start(self):
        self.client.get("/api/accounts/csrf/")
        response = self.client.post(
            "/api/accounts/login/",
            json={"username": os.environ["LOCUST_USERNAME"], "password": os.environ["LOCUST_PASSWORD"]},
            headers=self._csrf(),
        )
        response.raise_for_status()

    def _csrf(self) -> dict:
        return {"X-CSRFToken": self.client.cookies.get("csrftoken", "")}

    @task(5)
    def retrieve(self):
        self.client.post(
            "/api/rag/retrieve/",
            json={"question": random.choice(QUESTIONS)},
            headers=self._csrf(),
            name="/api/rag/retrieve/",
        )

    @task(1)
    def dashboard(self):
        self.client.get("/api/reports/summary/", name="/api/reports/summary/")

    @task(1)
    def ask(self):
        if os.environ.get("LOCUST_INCLUDE_LLM"):
            self.client.post(
                "/api/rag/query/",
                json={"question": random.choice(QUESTIONS)},
                headers=self._csrf(),
                name="/api/rag/query/",
                timeout=300,
            )

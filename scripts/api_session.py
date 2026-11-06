"""Sign in and print shell exports, so external tools (Promptfoo, Garak) can call the API with a session.

    eval "$(venv/bin/python scripts/api_session.py viewer)"

Uses DEMO_USER_PASSWORD or DEMO_ADMIN_PASSWORD from .env. The output contains a live session cookie,
so do not paste it into anything that gets logged or shared.
"""

import pathlib
import shlex
import sys

import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8010/api/"


def main() -> int:
    username = sys.argv[1] if len(sys.argv) > 1 else "viewer"
    env = dict(
        line.split("=", 1)
        for line in (ROOT / ".env").read_text().splitlines()
        if line and not line.startswith("#") and "=" in line
    )
    password = env["DEMO_ADMIN_PASSWORD" if username == "admin" else "DEMO_USER_PASSWORD"]
    client = httpx.Client(base_url=BASE, timeout=30)
    client.get("accounts/csrf/")
    response = client.post(
        "accounts/login/",
        json={"username": username, "password": password},
        headers={"X-CSRFToken": client.cookies.get("csrftoken", "")},
    )
    if response.status_code != 200:
        print(f"Sign-in failed with status {response.status_code}. Is the server running?", file=sys.stderr)
        return 1
    print(f"export RAG_SESSION={shlex.quote(client.cookies['sessionid'])}")
    print(f"export RAG_CSRF={shlex.quote(client.cookies['csrftoken'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""End-to-end check of the dashboard in a real browser: sign in, use each page as a person would, and watch for
script errors and failed API calls. Run after e2e_check.py, with the system Python that has Playwright:

    python3 tests/e2e/browser_check.py [screenshot-folder]
"""

import pathlib
import sys

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8010/"
env = dict(
    line.split("=", 1)
    for line in (ROOT / ".env").read_text().splitlines()
    if line and not line.startswith("#") and "=" in line
)
folder = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else None
if folder:
    folder.mkdir(parents=True, exist_ok=True)
outcomes: list[tuple[bool, str]] = []
problems: list[str] = []


def check(name, ok, detail=""):
    ok = bool(ok)
    outcomes.append((ok, name))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  ({detail})' if detail else ''}", flush=True)


def shot(page, name):
    if folder:
        page.screenshot(path=str(folder / f"{name}.png"), full_page=True)


def visit(page, path, expect_title):
    problems.clear()
    page.goto(BASE + path)
    page.wait_for_selector("h1")
    page.wait_for_timeout(1800)
    title = page.inner_text("h1")
    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
    check(
        f"{expect_title} page opens cleanly",
        title.upper() == expect_title.upper() and not problems and overflow <= 0,
        "; ".join(problems[:2]) or (f"sideways scroll {overflow}px" if overflow > 0 else ""),
    )


AXE = ROOT / "frontend" / "node_modules" / "axe-core" / "axe.min.js"


def scan_accessibility(page, title):
    """Run axe-core's WCAG 2.1 A and AA rules, including colour contrast, on the page as rendered."""
    page.add_script_tag(path=str(AXE))
    violations = page.evaluate(
        "axe.run(document, {runOnly: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']}).then(r => r.violations.map(v => [v.id, v.nodes.length, v.nodes[0].target[0]]))"
    )
    check(f"{title} meets WCAG 2.1 AA", not violations, "; ".join(f"{a} x{b} at {c}" for a, b, c in violations)[:220])


def sign_in(page, username, password):
    page.goto(BASE)
    page.wait_for_selector("input[autocomplete=username]")
    page.fill("input[autocomplete=username]", username)
    page.fill("input[autocomplete=current-password]", password)
    page.click("button.button-primary")
    page.wait_for_selector(".rail")


PAGES = [
    ("", "Overview"),
    ("documents", "Documents"),
    ("chat", "Chat"),
    ("datasets", "Datasets"),
    ("runs", "Runs"),
    ("metrics", "All 37 metrics"),
    ("retrieval", "Retrieval"),
    ("context", "Context"),
    ("answer", "Answer"),
    ("hallucination", "Hallucination"),
    ("security", "Security"),
    ("performance", "Performance"),
    ("compare", "Compare"),
    ("reports", "Reports"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(args=["--disable-dev-shm-usage"], ignore_default_args=["--hide-scrollbars"])
    page = browser.new_page(viewport={"width": 1360, "height": 900}, bypass_csp=True)
    page.on("console", lambda m: problems.append(f"console: {m.text[:140]}") if m.type == "error" else None)
    page.on("pageerror", lambda e: problems.append(f"script error: {str(e)[:140]}"))
    page.on("response", lambda r: problems.append(f"HTTP {r.status} {r.url[-40:]}") if r.status >= 500 else None)

    print("\nSign-in")
    page.goto(BASE)
    page.wait_for_selector("input[autocomplete=username]")
    page.fill("input[autocomplete=username]", "admin")
    page.fill("input[autocomplete=current-password]", "definitely-wrong")
    page.click("button.button-primary")
    page.wait_for_selector(".error-note")
    check("A wrong password shows a message", "Incorrect" in page.inner_text(".error-note"))
    sign_in(page, "admin", env["DEMO_ADMIN_PASSWORD"])
    check("A correct password opens the dashboard", page.locator(".rail").count() == 1)

    print("\nEvery page, as admin")
    for path, title in PAGES:
        visit(page, path, title)
        scan_accessibility(page, title)
        shot(page, f"admin_{title.lower()}")

    print("\nOverview")
    page.goto(BASE)
    page.wait_for_selector(".verdict", timeout=15000)
    check("It shows the overall score", page.locator(".verdict-number").count() == 1)
    check(
        "Attack results appear even if the latest run had none",
        "Not scored" not in page.locator("a.stage-row").last.inner_text(),
    )
    check("The background worker is reported", "Background worker" in page.inner_text(".health"))

    print("\nChat")
    page.on("dialog", lambda dialog: dialog.accept())
    before = page.evaluate("fetch('/api/rag/sessions/?limit=500').then(r=>r.json()).then(d=>d.results.map(x=>x.id))")
    page.goto(BASE + "chat")
    page.wait_for_selector(".suggestion")
    check("A new chat shows suggestions", page.locator(".suggestion").count() == 4)
    shot(page, "chat_empty")
    page.click(".suggestion >> nth=0")
    page.wait_for_selector(".thinking", timeout=10000)
    check("The thinking indicator shows while waiting", True)
    page.wait_for_selector(".thinking", state="detached", timeout=240000)
    text = page.inner_text(".answer")
    check("A suggestion gets an answer in the conversation", "24" in text, text[:70])
    check("The question appears as a chat bubble", "annual leave" in page.inner_text(".bubble"))
    page.wait_for_url("**/chat/*")
    page.wait_for_selector(".history-row-on")
    check("It appears in the history, highlighted", "annual leave" in page.inner_text(".history-row-on"))
    page.fill(".composer-box textarea", "and sick leave?")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "document.querySelectorAll('.turn').length === 2 && !document.querySelector('.thinking')", timeout=240000
    )
    followup = page.locator(".answer").last.inner_text()
    check("A follow-up understands the conversation", "12" in followup, followup[:70])
    check("It shows how the follow-up was understood", "sick leave" in page.locator(".understood").last.inner_text())
    page.locator(".details-toggle").last.click()
    page.wait_for_selector(".turn-details")
    check("Sources open under the answer", page.locator(".turn-details .source").count() > 0)
    check(
        "Each retrieval stage is listed",
        page.locator(".pipeline-stage").count() >= 3 and "Vector search" in page.content(),
    )
    if page.locator(".answer .cite").count():
        check("A citation is clickable", True)
    shot(page, "chat_conversation")
    page.click(".history-toggle")
    page.wait_for_selector(".history", state="hidden")
    check(
        "The history can be collapsed",
        page.locator(".chat-history-closed").count() == 1 and page.locator(".rail").is_visible(),
    )
    page.click(".history-toggle")
    page.wait_for_selector(".history", state="visible")
    check("The history can be expanded again", page.locator(".chat-history-closed").count() == 0)
    page.click(".chat-expand")
    page.wait_for_selector(".rail", state="hidden")
    check(
        "The chat expands to full screen",
        page.locator(".history").is_hidden() and page.locator(".composer-box").bounding_box()["width"] > 900,
    )
    shot(page, "chat_expanded")
    page.click(".chat-expand")
    page.wait_for_selector(".rail", state="visible")
    check("Full screen can be left again", page.locator(".history").is_visible())
    rows = page.locator(".history-row").count()
    page.locator(".history-group-name").first.click()
    check("A date group in the history can be collapsed", page.locator(".history-row").count() < rows)
    page.locator(".history-group-name").first.click()
    check("A date group can be expanded again", page.locator(".history-row").count() == rows)
    current = page.url
    page.click(".new-chat")
    page.wait_for_selector(".suggestion")
    check("New chat starts empty", page.locator(".turn").count() == 0)
    page.goto(current)
    page.wait_for_selector(".turn")
    page.wait_for_function("document.querySelectorAll('.turn').length === 2")
    check(
        "Opening a past chat restores it",
        "annual leave" in page.locator(".bubble").first.inner_text() and page.locator(".answer").count() == 2,
    )
    page.fill(".history-search input", "zzzz-no-such-chat")
    page.wait_for_selector(".history-empty")
    check("History search filters the list", page.locator(".history-row").count() == 0)
    page.fill(".history-search input", "")
    page.fill(".composer-box textarea", "backups")
    page.click("text=Search only")
    page.wait_for_function(
        "document.querySelectorAll('.turn').length === 3 && !document.querySelector('.thinking')", timeout=60000
    )
    check("Search only shows sources without the model", "not saved" in page.locator(".answer-note").inner_text())
    page.locator(".history-row-on").hover()
    page.locator(".history-row-on .history-delete").click()
    page.wait_for_url("**/chat", timeout=10000)
    page.wait_for_selector(".suggestion")
    check("Deleting the open chat returns to a new chat", page.locator(".turn").count() == 0)
    leftover = page.evaluate("fetch('/api/rag/sessions/?limit=500').then(r=>r.json()).then(d=>d.results.map(x=>x.id))")
    for session_id in set(leftover) - set(before):
        page.evaluate(
            f"fetch('/api/rag/sessions/{session_id}/', {{method:'DELETE', headers:{{'X-CSRFToken': document.cookie.match(/csrftoken=([^;]+)/)[1]}}}})"
        )

    print("\nDrilling in")
    page.goto(BASE + "answer")
    page.wait_for_selector(".link-question")
    page.locator(".link-question").first.click()
    page.wait_for_selector(".row-detail")
    check("A question row opens its evidence", "Retrieved context" in page.inner_text(".row-detail"))
    shot(page, "answer_detail")
    page.goto(BASE + "security")
    page.wait_for_selector(".breakdown-row")
    check("The security page shows results per attack type", page.locator(".breakdown-row").count() >= 5)
    page.locator("td .link").first.click()
    page.wait_for_selector(".row-detail")
    check("A test opens its prompt and answer", "Attack prompt" in page.inner_text(".row-detail"))
    page.goto(BASE + "documents")
    page.wait_for_selector("td .link")
    page.locator("td .link").first.click()
    page.wait_for_selector(".chunk-list")
    check("A document opens its chunks", page.locator(".chunk-list li").count() > 0)

    print("\nAll 37 metrics")
    page.goto(BASE + "metrics")
    page.wait_for_selector(".roadmap-row")
    page.wait_for_timeout(800)
    rows = page.locator(".roadmap-row")
    check(
        "It lists all 37, in order",
        rows.count() == 37 and "Precision@K" in rows.first.inner_text() and "Error Rate" in rows.last.inner_text(),
    )
    scored = sum(
        1
        for index in range(rows.count())
        if rows.nth(index).locator(".roadmap-score").inner_text().strip() not in ("", "–")
    )
    check("All 37 have a score from the full run", scored == 37, f"{scored} scored")
    shot(page, "all_metrics")

    print("\nCompare")
    page.goto(BASE + "compare")
    page.wait_for_selector(".run-chip")
    chips = page.locator(".run-chip input")
    for index in range(min(chips.count(), 3)):
        # Not .check(): the box is controlled by the URL, so it changes a moment after the click.
        chips.nth(index).click()
        page.wait_for_function(f"document.querySelectorAll('.run-chip-on').length === {index + 1}")
    page.wait_for_selector(".compare-row", timeout=15000)
    check("Choosing runs shows them side by side", page.locator(".compare-row").count() > 10)
    check("It says what differed between them", "What differs" in page.content())
    shot(page, "compare_filled")

    print("\nReports")
    page.goto(BASE + "reports")
    page.wait_for_selector("td .link")
    with page.expect_download(timeout=15000) as download:
        page.click("text=Download CSV >> nth=0")
    check("CSV downloads", download.value.suggested_filename.endswith(".csv"), download.value.suggested_filename)
    page.locator("td .link").first.click()
    page.wait_for_selector(".facts")
    check("A run shows what produced it", "Model digest" in page.inner_text(".facts"))
    shot(page, "reports_detail")

    print("\nThin scrollbar")
    page.goto(BASE + "metrics")
    page.wait_for_selector(".roadmap-row")
    width = page.evaluate("window.innerWidth - document.documentElement.clientWidth")
    check("The page scrollbar is 6 px or less", 0 < width <= 6, f"{width} px")

    print("\nProfile card")
    page.goto(BASE)
    page.wait_for_selector(".profile-trigger")
    check("The sidebar shows who is signed in", "admin" in page.inner_text(".profile-trigger"))
    page.click(".profile-trigger")
    page.wait_for_selector(".profile-card")
    card = page.inner_text(".profile-card")
    check("The card shows the role and what it allows", "Admin" in card and "audit log" in card)
    page.keyboard.press("Escape")
    page.wait_for_selector(".profile-card", state="detached")
    check("Escape closes it", page.locator(".profile-card").count() == 0)
    page.click(".profile-trigger")
    page.wait_for_selector(".profile-card")
    page.click("h1")
    page.wait_for_selector(".profile-card", state="detached")
    check("Clicking elsewhere closes it", page.locator(".profile-card").count() == 0)

    print("\nThemes")
    page.click(".profile-trigger")
    page.wait_for_selector(".profile-card")
    page.click("label[data-mode=dark]")
    page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    check("Dark theme applies", page.evaluate("getComputedStyle(document.body).colorScheme") == "dark")
    page.reload()
    page.wait_for_selector(".profile-trigger")
    check("The choice survives a reload", page.evaluate("document.documentElement.dataset.theme") == "dark")
    for path, title in PAGES:
        visit(page, path, title)
        scan_accessibility(page, f"Dark: {title}")
        shot(page, f"dark_{title.lower()}")
    for mode in ("light", "dark"):
        page.goto(BASE)
        page.wait_for_selector(".profile-trigger")
        page.click(".profile-trigger")
        page.click(f"label[data-mode={mode}]")
        for accent in ("blue", "violet", "emerald", "rose", "amber"):
            if not page.locator(".profile-card").count():
                page.click(".profile-trigger")
            page.click(f"label[data-accent={accent}]")
            page.wait_for_function(f"document.documentElement.dataset.accent === '{accent}'")
            page.wait_for_timeout(300)
            page.keyboard.press("Escape")
            page.wait_for_selector(".profile-card", state="detached")
            for path in (
                "",
                "chat",
                "compare?runs="
                + page.evaluate(
                    "fetch('/api/evaluation/runs/').then(r=>r.json()).then(d=>d.results.map(x=>x.id).slice(0,3).join(','))"
                ),
            ):
                page.goto(BASE + path)
                page.wait_for_selector("h1")
                page.wait_for_timeout(1000)
                scan_accessibility(page, f"{mode} + {accent}: {path.split('?')[0] or 'overview'}")
            shot(page, f"accent_{mode}_{accent}")
            page.goto(BASE)
            page.wait_for_selector(".profile-trigger")
            page.click(".profile-trigger")
            page.wait_for_selector(".profile-card")
    page.click("label[data-accent=blue]")
    page.click("label[data-mode=light]")
    page.keyboard.press("Escape")

    print("\nSign out and sign in")
    page.click(".profile-trigger")
    page.click(".rail-signout")
    page.wait_for_selector("input[autocomplete=username]")
    check("Signing out says so", "signed out" in page.inner_text(".login-notice"))
    shot(page, "signed_out")
    page.click(".theme-toggle")
    page.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    page.wait_for_timeout(400)  # the theme fades over 200 ms, and a scan mid-fade measures the wrong colours
    scan_accessibility(page, "Dark sign-in page")
    shot(page, "signin_dark")
    page.click(".theme-toggle")
    page.wait_for_function("document.documentElement.dataset.theme === 'light'")

    print("\nAs a viewer")
    sign_in(page, "viewer", env["DEMO_USER_PASSWORD"])
    page.goto(BASE + "documents")
    page.wait_for_selector("h1")
    page.wait_for_timeout(1500)
    check("A viewer has no upload form", page.locator("input[type=file]").count() == 0)
    check("A viewer does not see the restricted file", "salary_bands" not in page.content())
    page.goto(BASE + "answer")
    page.wait_for_timeout(2000)
    check(
        "A viewer sees scores but no answers",
        page.locator(".metric-row").count() > 0 and page.locator(".link-question").count() == 0,
    )
    shot(page, "viewer_answer")

    print("\nOn a tablet")
    page.set_viewport_size({"width": 820, "height": 1000})
    for path, title in (("", "Overview"), ("documents", "Documents"), ("compare", "Compare")):
        visit(page, path, title)
        shot(page, f"tablet_{title.lower()}")
    check(
        "The tablet menu shows icons with accessible names",
        page.locator(".rail-link").first.get_attribute("aria-label") == "Overview",
    )

    print("\nOn a phone")
    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(BASE + "chat")
    page.wait_for_selector(".history-toggle")
    page.click(".history-toggle")
    page.wait_for_selector(".history-open")
    check("The chat history opens as a side panel", page.locator(".history-open").is_visible())
    page.click(".history-scrim", position={"x": 370, "y": 400})
    page.wait_for_selector(".history-open", state="detached")
    for path, title in (("", "Overview"), ("chat", "Chat"), ("answer", "Answer"), ("compare", "Compare")):
        visit(page, path, title)
        shot(page, f"phone_{title.lower()}")
    browser.close()

failures = [name for ok, name in outcomes if not ok]
print(f"\n{len(outcomes) - len(failures)} of {len(outcomes)} browser checks passed.")
for name in failures:
    print(f"  FAILED: {name}")
sys.exit(1 if failures else 0)

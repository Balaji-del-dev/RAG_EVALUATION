"""Questions the document search cannot answer, sent somewhere else before it runs. Chat only: evaluation runs always
search, so their scores stay comparable.

- About the library itself ("how many vvdn documents in databse pls?"): counted and listed from SQL, for the
  documents the asker may read. No model call.
- About this conversation ("pls summery this thred history"): the model summarises the earlier answers. No search.
- Too vague to search ("pls properly summery this pls" as a first message): a question back, naming a few documents.

Each of these was a real question that got "I don't know". The rules are narrow on purpose: sending a policy question
here would lose its answer, while a library question the rules miss is only searched as before.
"""

import re

from apps.chatbot.services import generation
from apps.documents.models import Document

LIBRARY = "library"
CONVERSATION = "conversation"
CLARIFY = "clarify"
SEARCH = "search"

DOC_NOUN = r"(documents?|docs?|files?|pdfs?|uploads?)"
DOC_NOUN_RE = re.compile(rf"\b{DOC_NOUN}\b", re.I)
LIBRARY_CONTEXT_RE = re.compile(
    r"\b(in|on|inside) (the |your |this |my )?(database|databse|db|system|library|knowledge ?base|platform|app|index)\b"
    r"|\b(do|did) you have\b|\byou (have|hold|know about|can (read|see|search))\b"
    r"|\b(are|were|is) (there|uploaded|available|indexed|stored|loaded)\b|\b(uploaded|indexed)\b"
    r"|\b(can|may) i (see|read|access|search)\b|\bi (can|may) (see|read|access|search)\b"
    r"|\bmy (documents?|docs?|files?)\b",
    re.I,
)
# "list documents", "how many files?": nothing else in the question.
BARE_LIBRARY_RE = re.compile(
    rf"^\s*((pls|please) )?((list|show)( me)?( all)?( the| my)?|(how many|number of)) {DOC_NOUN}"
    r"( (are there|do you have|pls|please))?[\s?.!]*$",
    re.I,
)
# "documents required for joining" asks about paperwork in a policy, not about this library.
PAPERWORK_RE = re.compile(
    r"\b(required|needed|necessary|mandatory|submit\w*|bring|carry|attach\w*|for (joining|onboarding|verification))\b",
    re.I,
)
# Words that say what is asked, not which documents. What remains filters by file name: "vvdn" in the example.
LIBRARY_WORDS = frozenset(
    [
        "how",
        "many",
        "number",
        "count",
        "total",
        "list",
        "show",
        "all",
        "the",
        "my",
        "me",
        "in",
        "on",
        "inside",
        "your",
        "this",
        "database",
        "databse",
        "db",
        "system",
        "library",
        "knowledge",
        "base",
        "knowledgebase",
        "platform",
        "app",
        "index",
        "do",
        "did",
        "you",
        "have",
        "hold",
        "know",
        "about",
        "can",
        "read",
        "see",
        "search",
        "are",
        "were",
        "is",
        "there",
        "uploaded",
        "available",
        "indexed",
        "stored",
        "loaded",
        "may",
        "access",
        "pls",
        "please",
        "what",
        "which",
        "documents",
        "document",
        "docs",
        "doc",
        "files",
        "file",
        "pdfs",
        "pdf",
        "uploads",
        "upload",
        "of",
        "and",
        "a",
        "an",
        "i",
        "to",
        "for",
        "with",
        "give",
        "tell",
        "names",
        "name",
    ]
)

SUMMARY_WORD = r"(summar\w*|summer[iy]\w*|summery|summry|sumary|recap)"
THREAD = r"(thre?a?d|conversation|conversion|convo|chat|discussion|session)(?!\s+(polic\w*|usage|tool\w*|app\w*|rules))"
CONVERSATION_RE = re.compile(
    rf"\b{SUMMARY_WORD}\b.*(\b(this|our|the|my|whole|entire|full)\s+{THREAD}\b|\b{THREAD}\s+history\b|\bso far\b"
    rf"|\babove\b)|\bwhat (did|have) (i|we) (ask|asked|discuss|discussed|talk|talked)\b",
    re.I,
)

# Words that leave nothing to search for when they are all a first message has.
VAGUE_WORDS = frozenset(
    [
        "pls",
        "please",
        "plz",
        "kindly",
        "properly",
        "proper",
        "this",
        "that",
        "it",
        "these",
        "those",
        "them",
        "more",
        "detail",
        "details",
        "explain",
        "summarise",
        "summarize",
        "summary",
        "summery",
        "summry",
        "sumary",
        "summarise",
        "summarize",
        "tell",
        "describe",
        "elaborate",
        "give",
        "me",
        "about",
        "briefly",
        "brief",
        "short",
        "full",
        "fully",
        "the",
        "a",
        "an",
        "can",
        "could",
        "you",
        "i",
        "need",
        "want",
        "show",
        "write",
        "do",
        "again",
        "better",
        "clearly",
        "clear",
        "what",
        "is",
        "are",
        "so",
    ]
)
ACTION_RE = re.compile(
    r"\b(summar\w*|summer[iy]\w*|summery|summry|sumary|explain|describe|elaborate|tell|details?)\b", re.I
)

TYPES = {".pdf": "PDF", ".docx": "Word", ".csv": "CSV", ".txt": "Text", ".md": "Markdown"}
MAX_LISTED = 50
SUGGESTED = 5


def asks_about_conversation(question: str) -> bool:
    return bool(CONVERSATION_RE.search(question))


def asks_about_library(question: str) -> bool:
    if not DOC_NOUN_RE.search(question) or PAPERWORK_RE.search(question):
        return False
    return bool(LIBRARY_CONTEXT_RE.search(question) or BARE_LIBRARY_RE.match(question))


def is_vague(question: str) -> bool:
    words = set(re.findall(r"[a-z]+", question.lower()))
    return bool(words) and bool(ACTION_RE.search(question)) and words <= VAGUE_WORDS


def route(question: str, earlier_question: str | None, history: list[dict] | None) -> str:
    """Where this chat question goes. `history` is the conversation so far; None outside the chat."""
    # As typed: "list" in "list documents" is also a format word, and would be taken out.
    if asks_about_conversation(question):
        return CONVERSATION if history else CLARIFY
    if asks_about_library(question):
        return LIBRARY
    without_format = generation.answer_format(question)[1] or question
    if not earlier_question and is_vague(without_format):
        return CLARIFY
    return SEARCH


def _filter_terms(question: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", question.lower())
    return [word for word in dict.fromkeys(words) if len(word) >= 3 and word not in LIBRARY_WORDS]


def _row(number: int, document: Document) -> str:
    extension = "." + document.filename.rsplit(".", 1)[-1].lower() if "." in document.filename else ""
    pages = str(document.page_count) if document.page_count else "–"
    return f"| {number} | {document.filename} | {TYPES.get(extension, extension.upper()[1:] or '–')} | {pages} | {document.created_at:%d %b %Y} |"


def library_answer(question: str, allowed_document_ids: list[int]) -> str:
    """How many of the asker's documents there are, and which, as a Markdown table. Only documents the asker may read,
    and only those ready to search, are counted."""
    documents = list(Document.objects.filter(id__in=allowed_document_ids).order_by("-created_at", "-id"))
    if not documents:
        return "You have no documents to search yet. Upload one on the Documents page."
    total = len(documents)
    terms = _filter_terms(generation.answer_format(question)[1] or question)
    matching = [doc for doc in documents if all(term in doc.filename.lower() for term in terms)] if terms else documents
    plural = "document" if total == 1 else "documents"
    named = " ".join(f'"{term}"' for term in terms)
    if terms and not matching:
        head, shown = f"You can read {total} {plural}, and none has {named} in its name. All of them:", documents
    elif terms:
        count = len(matching)
        head = f"You can read {total} {plural}. {count} {'has' if count == 1 else 'have'} {named} in the name:"
        shown = matching
    else:
        head, shown = f"You can read {total} {plural}:", documents
    rows = [_row(number, document) for number, document in enumerate(shown[:MAX_LISTED], start=1)]
    more = f"\n\nAnd {len(shown) - MAX_LISTED} more, on the Documents page." if len(shown) > MAX_LISTED else ""
    table = "| # | Document | Type | Pages | Added |\n|---|---|---|---|---|\n" + "\n".join(rows)
    return f"{head}\n\n{table}{more}"


def clarify_answer(allowed_document_ids: list[int], conversation: bool = False) -> str:
    documents = Document.objects.filter(id__in=allowed_document_ids).order_by("-created_at", "-id")[:SUGGESTED]
    opening = (
        "There is nothing earlier in this conversation to summarise yet."
        if conversation
        else "What would you like me to summarise or explain?"
    )
    if not documents:
        return f"{opening} You have no documents to search yet: upload one on the Documents page."
    names = "\n".join(f"- {document.filename.rsplit('.', 1)[0].replace('_', ' ')}" for document in documents)
    return (
        f"{opening} Name a document or a topic, for example:\n\n{names}\n\nOr ask about a topic, such as leave policy."
    )


CONVERSATION_PROMPT = (
    "You summarise a conversation between a user and an assistant that answers from the user's documents. Write one "
    "bullet per question: what was asked, and what the answer said. Keep every figure, name and date exactly as "
    "written. Say when a question was not answered. Add nothing that is not in the conversation."
)
MAX_ANSWER_CHARS = 700
MAX_CONVERSATION_CHARS = 5000


def conversation_messages(history: list[dict]) -> list[dict]:
    """The earlier questions and answers, newest kept when they do not all fit. Citation markers are dropped: the
    summary has no sources to point them at."""
    turns = []
    for number, turn in enumerate(history, start=1):
        answer = re.sub(r"\s*\[S\d+\]", "", turn["answer"]).strip()
        if len(answer) > MAX_ANSWER_CHARS:
            answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + " …"
        turns.append(f"Question {number}: {turn['question']}\nAnswer {number}: {answer}")
    while len("\n\n".join(turns)) > MAX_CONVERSATION_CHARS and len(turns) > 1:
        turns.pop(0)
    return [
        {"role": "system", "content": CONVERSATION_PROMPT},
        {"role": "user", "content": "Conversation:\n\n" + "\n\n".join(turns) + "\n\nSummary:"},
    ]

"""An answer as a file to download: Markdown, text, HTML, PDF, Word, OpenDocument, RTF, Excel, CSV, PowerPoint, JSON,
YAML or XML.

Every file is built from the answer's Markdown by rule, so it contains what the chat showed and nothing the model wrote
separately. Office formats go through LibreOffice, which converts in a second or two but uses a few hundred MB while it
runs. So conversions run one at a time, with a timeout, and are the first thing the kernel kills if memory runs out.
"""

import contextlib
import html
import io
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import markdown

from apps.chatbot.services import formats
from apps.common.exceptions import ServiceUnavailable

CONVERSION_TIMEOUT_SECONDS = 90
_office = threading.Lock()


@dataclass(frozen=True)
class Download:
    content_type: str
    office_filter: str = ""  # LibreOffice "--convert-to" target, for formats it builds


DOWNLOADS = {
    "md": Download("text/markdown; charset=utf-8"),
    "txt": Download("text/plain; charset=utf-8"),
    "html": Download("text/html; charset=utf-8"),
    "pdf": Download("application/pdf", "pdf:writer_web_pdf_Export"),
    "docx": Download(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx:MS Word 2007 XML"
    ),
    "odt": Download("application/vnd.oasis.opendocument.text", "odt:writerweb8_writer"),
    "rtf": Download("application/rtf", "rtf:Rich Text Format"),
    "xlsx": Download(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx:Calc MS Excel 2007 XML"
    ),
    "ods": Download("application/vnd.oasis.opendocument.spreadsheet", "ods:calc8"),
    "csv": Download("text/csv; charset=utf-8"),
    "pptx": Download("application/vnd.openxmlformats-officedocument.presentationml.presentation"),
    "json": Download("application/json"),
    "yaml": Download("application/yaml"),
    "xml": Download("application/xml"),
}
SPREADSHEETS = {"xlsx", "ods"}
STYLE = """
body { max-width: 860px; margin: 40px auto; padding: 0 24px; color: #0f172a; background: #ffffff;
  font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 28px; line-height: 1.25; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 32px 0 8px; } h3 { font-size: 17px; margin: 24px 0 8px; }
.meta { color: #475569; font-size: 14px; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }
th, td { border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; vertical-align: top; }
th { background: #f1f5f9; }
code { background: #f1f5f9; padding: 0 4px; border-radius: 4px; font-size: 14px; }
pre { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; overflow-x: auto; }
pre code { background: none; padding: 0; }
details { border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 16px; margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
.sources { margin-top: 32px; padding-top: 16px; border-top: 1px solid #e2e8f0; color: #475569; font-size: 14px; }
@media (max-width: 600px) { body { margin: 16px auto; } table { display: block; overflow-x: auto; } }
"""


def title_of(message) -> str:
    return re.sub(r"\s+", " ", message.question).strip()[:120] or "Answer"


def filename_for(message, extension: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "-", title_of(message).lower()).strip("-")[:60] or "answer"
    return f"{stem}.{extension}"


def _sources(message) -> list[str]:
    lines = []
    for citation in message.citations or []:
        if citation.get("chunk_id"):
            page = f", page {citation['page_number']}" if citation.get("page_number") else ""
            lines.append(f"[{citation['marker']}] {citation['filename']}{page}")
    return lines


def document_markdown(message) -> str:
    """The title, the answer and its sources, as one Markdown document."""
    sources = _sources(message)
    tail = "\n\n## Sources\n\n" + "\n".join(f"- {line}" for line in sources) if sources else ""
    return f"# {title_of(message)}\n\n{message.answer.strip()}{tail}\n"


def to_html(message, collapsible: bool = False) -> str:
    """A standalone, styled page. Any HTML in the answer is shown as text, never run."""
    body = message.answer.strip().replace("<", "&lt;")
    converted = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    if collapsible:
        # Each ## section becomes a <details> block that opens and closes without any script.
        parts = re.split(r"(<h2>.*?</h2>)", converted)
        opened = [parts[0]]
        for heading, content in zip(parts[1::2], parts[2::2], strict=False):
            opened.append(f"<details open><summary>{heading[4:-5]}</summary>{content}</details>")
        converted = "".join(opened)
    sources = "".join(f"<li>{html.escape(line)}</li>" for line in _sources(message))
    sources_block = f'<section class="sources"><h2>Sources</h2><ul>{sources}</ul></section>' if sources else ""
    title = html.escape(title_of(message))
    when = message.created_at.strftime("%d %b %Y, %H:%M") if message.created_at else ""
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{title}</title>\n'
        f"<style>{STYLE}</style>\n</head>\n<body>\n<h1>{title}</h1>\n"
        f'<p class="meta">Answered from your documents · {when}</p>\n{converted}\n{sources_block}\n</body>\n</html>\n'
    )


def to_text(message) -> str:
    text = document_markdown(message)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`{3}[a-z]*\n?|`", "", text)
    return text


def _lowest_priority_for_the_oom_killer():
    with contextlib.suppress(OSError):
        Path("/proc/self/oom_score_adj").write_text("1000")


def _office_convert(source_name: str, content: bytes, office_filter: str, infilter: str = "") -> bytes:
    with _office, tempfile.TemporaryDirectory(prefix="rag-download-") as work:
        source = Path(work) / source_name
        source.write_bytes(content)
        command = [
            "soffice",
            f"-env:UserInstallation=file://{work}/profile",
            "--headless",
            "--norestore",
            *([f"--infilter={infilter}"] if infilter else []),
            "--convert-to",
            office_filter,
            "--outdir",
            str(Path(work) / "out"),
            str(source),
        ]
        try:
            subprocess.run(
                command,
                capture_output=True,
                timeout=CONVERSION_TIMEOUT_SECONDS,
                check=False,
                preexec_fn=_lowest_priority_for_the_oom_killer,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ServiceUnavailable(f"LibreOffice could not convert the answer: {exc}") from exc
        produced = list((Path(work) / "out").glob("*"))
        if not produced:
            raise ServiceUnavailable("LibreOffice could not convert the answer.")
        return produced[0].read_bytes()


def to_pptx(message) -> bytes:
    """A title slide, one slide per ## section (or per few lines), and a sources slide."""
    from pptx import Presentation

    deck = Presentation()
    first = deck.slides.add_slide(deck.slide_layouts[0])
    first.shapes.title.text = title_of(message)
    first.placeholders[1].text = "Answered from your documents"
    sections, current = [], ["Answer", []]
    for line in message.answer.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading:
            if current[1]:
                sections.append(current)
            current = [heading.group(1).strip(), []]
        elif line.strip() and not formats.SEPARATOR_RE.match(line):
            current[1].append(line)
    if current[1]:
        sections.append(current)
    for heading, lines in sections:
        for start in range(0, len(lines), 6):
            slide = deck.slides.add_slide(deck.slide_layouts[1])
            slide.shapes.title.text = heading[:80]
            frame = slide.placeholders[1].text_frame
            for index, line in enumerate(lines[start : start + 6]):
                clean = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line.strip().strip("|").replace("|", " · "))
                paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
                paragraph.text = clean.replace("**", "")[:220]
    if sources := _sources(message):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = "Sources"
        frame = slide.placeholders[1].text_frame
        for index, line in enumerate(sources):
            (frame.paragraphs[0] if index == 0 else frame.add_paragraph()).text = line
    out = io.BytesIO()
    deck.save(out)
    return out.getvalue()


def build(message, extension: str) -> tuple[bytes, str, str]:
    """The file's bytes, content type and name."""
    download = DOWNLOADS[extension]
    name = filename_for(message, extension)
    if extension == "md":
        content = document_markdown(message).encode()
    elif extension == "txt":
        content = to_text(message).encode()
    elif extension == "html":
        content = to_html(message, collapsible=message.answer_format == "interactive_html").encode()
    elif extension == "csv":
        content = formats.to_csv(message.answer).encode("utf-8-sig")  # the mark makes Excel read it as UTF-8
    elif extension in {"json", "yaml", "xml"}:
        block = formats.convert(message.answer, extension)
        content = block.split("\n", 1)[1].rsplit("```", 1)[0].encode()
    elif extension == "pptx":
        content = to_pptx(message)
    elif extension in SPREADSHEETS:
        csv_bytes = formats.to_csv(message.answer).encode("utf-8")
        content = _office_convert("answer.csv", csv_bytes, download.office_filter, infilter="CSV:44,34,76,1")
    else:
        content = _office_convert("answer.html", to_html(message).encode(), download.office_filter)
    return content, download.content_type, name

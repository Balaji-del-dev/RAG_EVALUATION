"""Answer formats: how an answer is laid out when the question asks for it ("in a comparison table", "as a mind map",
"download as PDF").

The small local model reliably writes only two things: Markdown and short Mermaid diagrams. Asked to write HTML, it
repeated "```html" until it ran out of tokens. So every format asks it for one of those, and the rest is done by rule:
JSON, YAML, XML and CSV are converted from its table or list, files are built from its Markdown, and interactive
layouts are drawn by the chat screen from its headings, lists and Q&A lines.

A format is recognised only with a cue ("in a table", "as bullet points", "draw a flowchart", "table format"), except
for phrases that mean nothing else ("pros and cons", "step by step"). Otherwise "What is on the dining table?" or
"summarise this pdf" would change the layout.
"""

import csv
import io
import json
import re
from dataclasses import dataclass, replace
from xml.etree import ElementTree

import yaml

CITATION_RE = re.compile(r"\[S(\d+)\]")
MERMAID_NOTE = (
    "Draw it as a Mermaid diagram in a ```mermaid code block. Keep node ids to single words and put labels in quotes. "
    "Below the block, write one line naming the sources used, like [S1]."
)
# Asked for a table, the model once wrote it all on one line and then repeated "| |" until Ollama stopped it. Rows on
# their own lines, short cells and a numbered first column keep it on track, and number the rows in order.
TABLE_NOTE = (
    "Put each row on its own line and keep each cell short. Number the rows in a first column #, in the order the "
    "sources give them. Put the source, like [S1], in the last column. Do not put the table in a code block."
)
CONVERTED_NOTE = "Write the answer as one Markdown table with a header row, or as a bullet list if it is not tabular."
DOCUMENT_NOTE = (
    "Write it as a document: a title heading, then short sections with ## headings, lists and tables where they help. "
    "End each paragraph with its source, like [S1]."
)
SECTIONS_NOTE = "Write 3 to 6 sections. Start each with a ## heading, then two or three sentences ending with sources."


@dataclass(frozen=True)
class AnswerFormat:
    key: str
    label: str
    group: str
    phrases: tuple[str, ...]
    instruction: str
    # How the chat screen shows it: markdown, mermaid, data, file, html, tree, checklist, searchable, accordion,
    # tabs, cards, carousel, timeline, dashboard, quiz or flashcards.
    render: str = "markdown"
    max_tokens: int = 384
    # For render "data": json, yaml, xml or csv. For render "file": the file extension to offer.
    target: str = ""
    # Phrases that need no cue, because they mean nothing but a layout.
    free: tuple[str, ...] = ()
    # Shown under the answer, when the format cannot be drawn as asked.
    note: str = ""


def _f(key, label, group, phrases, instruction, **options) -> AnswerFormat:
    return AnswerFormat(key, label, group, tuple(phrases), instruction, **options)


FORMATS: list[AnswerFormat] = [
    # ---- Text and content ----
    _f(
        "paragraph",
        "Paragraphs",
        "text",
        ["paragraph", "paragraphs", "prose", "essay"],
        "Write the answer as well-formed paragraphs, without lists or tables.",
    ),
    _f(
        "bullets",
        "Bullet list",
        "text",
        ["bullet list", "bullet points", "bulleted list", "bullets", "points", "list", "point wise", "pointwise"],
        "Write the answer as a Markdown bullet list, one point per line, each ending with its source, like [S1].",
        free=("bullet points", "point wise", "pointwise"),
    ),
    _f(
        "numbered",
        "Numbered list",
        "text",
        ["numbered list", "numbered points", "ordered list", "numbered"],
        "Write the answer as a numbered Markdown list, each item ending with its source, like [S1].",
        free=("numbered list",),
    ),
    _f(
        "nested",
        "Nested list",
        "text",
        [
            "nested list",
            "nested bullets",
            "nested bullet list",
            "sub points",
            "subpoints",
            "main points and sub points",
        ],
        "Write the answer as a nested Markdown list: main points, with indented sub-points under each.",
        free=("nested list", "nested bullets"),
    ),
    _f(
        "checklist",
        "Checklist",
        "text",
        [
            "checklist",
            "check list",
            "to do list",
            "todo list",
            "task list",
            "interactive checklist",
            "progress checklist",
            "tracking checklist",
        ],
        "Write the answer as a Markdown task list, one task per line, written as '- [ ] task [S1]'.",
        render="checklist",
        free=("interactive checklist", "progress checklist"),
    ),
    _f(
        "comparison_list",
        "Comparison list",
        "text",
        ["comparison list", "comparison points", "compare list"],
        "Compare the items point by point as a bullet list: one bullet per aspect, saying how each item differs.",
    ),
    _f(
        "definition",
        "Definitions",
        "text",
        ["definition format", "definitions", "definition", "glossary"],
        "Write each term in bold, followed by a colon and a one-sentence definition from the sources, like "
        "'**Term**: definition [S1]', one per line.",
        free=("definition format", "glossary"),
    ),
    _f(
        "qa",
        "Q&A",
        "text",
        ["q&a", "q & a", "qna", "question and answer", "questions and answers", "question answer"],
        "Write the answer as questions and answers. Each pair is '**Q:** question' on one line and '**A:** answer "
        "[S1]' on the next.",
        free=("q&a", "q & a", "qna", "questions and answers"),
    ),
    _f(
        "faq",
        "FAQ",
        "text",
        ["faq", "faqs", "frequently asked questions"],
        "Write the answer as an FAQ: each question as a ### heading, followed by a short answer with its source.",
        free=("faq", "faqs", "frequently asked questions"),
    ),
    _f(
        "summary",
        "Summary",
        "text",
        [
            "summary",
            "in short",
            "brief",
            "briefly",
            "in brief",
            "tldr",
            "tl;dr",
            "short answer",
            "one line",
            "one sentence",
            "in one line",
            "in one sentence",
            "answer in one line",
            "one line answer",
            "one-line answer",
        ],
        "Answer in one or two short sentences, ending with the source, like [S1].",
        max_tokens=160,
        free=(
            "in short",
            "in brief",
            "briefly",
            "tldr",
            "tl;dr",
            "in one line",
            "in one sentence",
            "answer in one line",
            "one line answer",
            "one-line answer",
            "short answer",
        ),
    ),
    _f(
        "detailed",
        "Detailed explanation",
        "text",
        [
            "detailed explanation",
            "in detail",
            "in depth",
            "in-depth",
            "detailed",
            "elaborate",
            "elaborately",
            "deep explanation",
            "detailed answer",
        ],
        "Answer in detail, covering every relevant point in the sources, with a source after each sentence.",
        max_tokens=512,
        free=("detailed explanation", "in detail", "in depth", "in-depth", "elaborately"),
    ),
    _f(
        "steps",
        "Step-by-step guide",
        "text",
        [
            "step by step",
            "step-by-step",
            "steps",
            "step wise",
            "stepwise",
            "step-by-step guide",
            "how-to guide",
            "numbered steps",
        ],
        "Write the answer as numbered steps, one per line, each ending with its source, like [S1].",
        free=("step by step", "step-by-step", "step wise", "stepwise", "step-by-step guide"),
    ),
    _f(
        "pros_cons",
        "Pros and cons",
        "text",
        [
            "pros and cons",
            "pros & cons",
            "advantages and disadvantages",
            "merits and demerits",
            "benefits and drawbacks",
        ],
        "Write two Markdown lists under the headings '### Pros' and '### Cons', each point ending with its source.",
        free=("pros and cons", "pros & cons", "advantages and disadvantages", "merits and demerits"),
    ),
    _f(
        "dos_donts",
        "Do's and don'ts",
        "text",
        ["do's and don'ts", "dos and donts", "dos and don'ts", "do and don't", "do's & don'ts", "do and dont"],
        "Write two Markdown lists under the headings '### Do' and '### Don't', each point ending with its source.",
        free=("do's and don'ts", "dos and donts", "dos and don'ts", "do's & don'ts"),
    ),
    _f(
        "cheat_sheet",
        "Cheat sheet",
        "text",
        ["cheat sheet", "cheatsheet", "quick reference", "quick revision"],
        "Write a compact cheat sheet: short ### headings with terse bullet points, facts only.",
        free=("cheat sheet", "cheatsheet", "quick reference", "quick revision"),
    ),
    # ---- Tables ----
    _f(
        "table",
        "Table",
        "table",
        ["table", "tabular", "simple table", "tabulated"],
        "Write the answer as a Markdown table. Use a header row, then one row for each separate point in the sources. "
        + TABLE_NOTE,
        max_tokens=512,
        free=("tabular",),
    ),
    _f(
        "detailed_table",
        "Detailed table",
        "table",
        ["detailed table"],
        "Write a Markdown table with columns for the point, its details and an example where the sources give one. "
        + TABLE_NOTE,
        max_tokens=512,
        free=("detailed table",),
    ),
    _f(
        "comparison_table",
        "Comparison table",
        "table",
        ["comparison table", "compare table", "comparison matrix", "comparison chart"],
        "Write a Markdown comparison table: one row per aspect, one column per item compared. " + TABLE_NOTE,
        max_tokens=512,
        free=("comparison table", "comparison matrix"),
    ),
    _f(
        "multi_column",
        "Multi-column table",
        "table",
        ["multi-column table", "multi column table", "multicolumn table", "multiple columns"],
        "Write a Markdown table with as many columns as the sources give attributes for. " + TABLE_NOTE,
        max_tokens=512,
        free=("multi-column table", "multi column table", "multicolumn table"),
    ),
    _f(
        "priority_table",
        "Priority table",
        "table",
        ["priority table", "priority wise", "priority-wise", "by priority", "priority order"],
        "Write a Markdown table with columns Priority (Critical, High, Medium or Low), Item and Why, ordered from "
        "Critical to Low. " + TABLE_NOTE,
        max_tokens=512,
        free=("priority table", "priority wise", "priority-wise"),
    ),
    _f(
        "ranking",
        "Ranking",
        "table",
        [
            "ranking table",
            "ranking",
            "ranked list",
            "top 5",
            "top five",
            "top 10",
            "top ten",
            "top 20",
            "top 50",
            "top 100",
        ],
        "Write a Markdown table with columns Rank, Item and Why, most important first. Include only items the sources "
        "support, even if that is fewer than asked. " + TABLE_NOTE,
        max_tokens=512,
        free=("ranking table", "ranked list", "top 5", "top five", "top 10", "top ten", "top 20", "top 50", "top 100"),
    ),
    _f(
        "roadmap",
        "Roadmap",
        "table",
        [
            "roadmap table",
            "roadmap",
            "learning path",
            "beginner to advanced",
            "beginner-to-advanced",
            "learning roadmap",
        ],
        "Write a Markdown table with columns Stage (Beginner, Intermediate or Advanced), Topic and What to learn, in "
        "order. " + TABLE_NOTE,
        max_tokens=512,
        free=("roadmap table", "learning path", "beginner to advanced", "beginner-to-advanced", "learning roadmap"),
    ),
    _f(
        "evaluation_matrix",
        "Evaluation matrix",
        "table",
        ["evaluation matrix", "scoring matrix", "assessment matrix", "matrix"],
        "Write a Markdown table with one row per item and one column per criterion, with the facts from the sources in "
        "the cells. " + TABLE_NOTE,
        max_tokens=512,
        free=("evaluation matrix", "scoring matrix", "assessment matrix"),
    ),
    _f(
        "interview_table",
        "Interview question table",
        "table",
        ["interview question table", "interview questions table", "interview table"],
        "Write a Markdown table with columns Question, Answer and Source, answers taken from the sources.",
        max_tokens=512,
        free=("interview question table", "interview questions table", "interview table"),
    ),
    _f(
        "spec_table",
        "Specification table",
        "table",
        ["technical specification table", "specification table", "spec table", "specifications table", "spec sheet"],
        "Write a Markdown table with columns Specification and Value. " + TABLE_NOTE,
        max_tokens=512,
        free=("technical specification table", "specification table", "spec table", "spec sheet"),
    ),
    _f(
        "searchable_table",
        "Searchable table",
        "web",
        ["searchable table", "filterable table", "search table", "filter table"],
        "Write the answer as a Markdown table with a header row and one row per item. " + TABLE_NOTE,
        render="searchable",
        max_tokens=512,
        free=("searchable table", "filterable table"),
    ),
    # ---- Structure and diagrams ----
    _f(
        "tree",
        "Tree structure",
        "diagram",
        [
            "tree structure",
            "tree format",
            "tree view",
            "tree",
            "hierarchy",
            "hierarchical structure",
            "hierarchical",
            "hierarchical tree",
        ],
        "Write the answer as a nested Markdown list: the main concept, its sub-concepts indented under it, and their "
        "sub-concepts indented again, in order.",
        render="tree",
        free=("tree structure", "tree view", "hierarchical structure", "hierarchical tree"),
    ),
    _f(
        "flowchart",
        "Flowchart",
        "diagram",
        ["flowchart", "flow chart", "process flow", "flow diagram"],
        "Show the process as a flowchart. Start the block with 'flowchart TD'. " + MERMAID_NOTE,
        render="mermaid",
    ),
    _f(
        "architecture",
        "Architecture diagram",
        "diagram",
        ["architecture diagram", "system architecture", "block diagram", "component diagram"],
        "Show the parts and how they connect. Start the block with 'flowchart LR', using subgraph for groups. "
        + MERMAID_NOTE,
        render="mermaid",
        free=("architecture diagram", "block diagram", "component diagram"),
    ),
    _f(
        "sequence",
        "Sequence diagram",
        "diagram",
        ["sequence diagram"],
        "Show who sends what to whom, in order. Start the block with 'sequenceDiagram'. " + MERMAID_NOTE,
        render="mermaid",
        free=("sequence diagram",),
    ),
    _f(
        "class_diagram",
        "Class diagram",
        "diagram",
        ["class diagram", "uml"],
        "Start the block with 'classDiagram'. " + MERMAID_NOTE,
        render="mermaid",
        free=("class diagram",),
    ),
    _f(
        "er_diagram",
        "ER diagram",
        "diagram",
        ["er diagram", "erd", "entity relationship diagram", "entity-relationship diagram"],
        "Start the block with 'erDiagram'. " + MERMAID_NOTE,
        render="mermaid",
        free=("er diagram", "entity relationship diagram", "entity-relationship diagram"),
    ),
    _f(
        "mind_map",
        "Mind map",
        "diagram",
        ["mind map", "mindmap", "concept map"],
        "Start the block with 'mindmap', with the topic as root((Topic)) and the points indented under it. "
        + MERMAID_NOTE,
        render="mermaid",
        free=("mind map", "mindmap", "concept map"),
    ),
    _f(
        "workflow",
        "Workflow diagram",
        "diagram",
        ["workflow diagram", "workflow", "end to end flow", "end-to-end flow"],
        "Show the workflow from start to end. Start the block with 'flowchart LR'. " + MERMAID_NOTE,
        render="mermaid",
        free=("workflow diagram",),
    ),
    _f(
        "decision_tree",
        "Decision tree",
        "diagram",
        ["decision tree"],
        "Show the decisions as a flowchart with {question} diamond nodes and labelled Yes/No edges. Start the block "
        "with 'flowchart TD'. " + MERMAID_NOTE,
        render="mermaid",
        free=("decision tree",),
    ),
    _f(
        "state",
        "State diagram",
        "diagram",
        ["state diagram", "state machine", "state chart"],
        "Start the block with 'stateDiagram-v2'. " + MERMAID_NOTE,
        render="mermaid",
        free=("state diagram", "state machine"),
    ),
    _f(
        "data_flow",
        "Data flow diagram",
        "diagram",
        ["data flow diagram", "dfd", "data flow"],
        "Show where data comes from, where it goes and what changes it. Start the block with 'flowchart LR'. "
        + MERMAID_NOTE,
        render="mermaid",
        free=("data flow diagram",),
    ),
    _f(
        "timeline_diagram",
        "Timeline diagram",
        "diagram",
        ["timeline diagram", "timeline chart"],
        "Start the block with 'timeline', then a 'title' line, then one 'date : event' line per event. " + MERMAID_NOTE,
        render="mermaid",
        free=("timeline diagram", "timeline chart"),
    ),
    _f(
        "mermaid",
        "Mermaid diagram",
        "diagram",
        ["mermaid diagram", "mermaid", "diagram"],
        "Choose the Mermaid diagram type that fits best. " + MERMAID_NOTE,
        render="mermaid",
        free=("mermaid diagram", "mermaid"),
    ),
    # ---- Code ----
    _f(
        "code",
        "Code block",
        "code",
        [
            "code block",
            "code snippet",
            "code example",
            "code",
            "source code",
            "syntax highlighted code",
            "syntax-highlighted code",
            "highlighted code",
        ],
        "Write the code in a fenced code block with its language name, then one line saying which source it comes "
        "from, like [S1].",
        max_tokens=512,
        free=("code block", "code snippet", "code example"),
    ),
    _f(
        "inline_code",
        "Inline code",
        "code",
        ["inline code"],
        "Write a short explanation and put every keyword, command, file name and value in `backticks`.",
        free=("inline code",),
    ),
    _f(
        "pseudocode",
        "Pseudocode",
        "code",
        ["pseudocode", "pseudo code", "pseudo-code"],
        "Write the logic as numbered pseudocode lines in a ```text code block, then name the sources used.",
        free=("pseudocode", "pseudo code", "pseudo-code"),
    ),
    _f(
        "input_output",
        "Input / output",
        "code",
        ["input output", "input/output", "input and output", "i/o format", "input output format"],
        "Write '### Input' with a code block and '### Output' with a code block, then name the sources used.",
        free=("input/output", "input and output", "i/o format"),
    ),
    _f(
        "code_explained",
        "Code with explanation",
        "code",
        ["code with explanation", "explained code", "code explanation", "explain the code"],
        "Write the code in a fenced code block, then a numbered list explaining it part by part.",
        max_tokens=512,
        free=("code with explanation", "explained code"),
    ),
    _f(
        "code_comparison",
        "Code comparison",
        "code",
        ["code comparison", "before and after", "before vs after", "before/after", "before vs. after"],
        "Write '### Before' and '### After', each with a code block, then a bullet list of what changed and why.",
        max_tokens=512,
        free=("code comparison", "before and after", "before vs after", "before/after"),
    ),
    _f(
        "debugging",
        "Debugging",
        "code",
        [
            "debugging format",
            "debug format",
            "error and solution",
            "error and fix",
            "troubleshooting steps",
            "troubleshooting",
        ],
        "Write '### Error', '### Cause' and '### Fix', each short, with code blocks where useful and sources.",
        free=("debugging format", "error and solution", "error and fix", "troubleshooting steps"),
    ),
    _f(
        "api",
        "API request and response",
        "code",
        [
            "api request and response",
            "api request",
            "api response",
            "request and response",
            "request/response",
            "curl example",
        ],
        "Write '### Request' with an HTTP or curl code block and '### Response' with a JSON code block, from the "
        "sources only.",
        max_tokens=512,
        free=("api request and response", "request and response", "request/response", "curl example"),
    ),
    _f("json", "JSON", "data", ["json"], CONVERTED_NOTE, render="data", target="json", max_tokens=512),
    _f("yaml", "YAML", "data", ["yaml", "yml"], CONVERTED_NOTE, render="data", target="yaml", max_tokens=512),
    _f("xml", "XML", "data", ["xml"], CONVERTED_NOTE, render="data", target="xml", max_tokens=512),
    _f(
        "sql",
        "SQL",
        "code",
        ["sql script", "sql query", "sql queries", "sql"],
        "Write the SQL in a ```sql code block, then one line naming the sources used.",
        max_tokens=512,
        free=("sql script", "sql query", "sql queries"),
    ),
    _f(
        "shell",
        "Shell commands",
        "code",
        [
            "shell commands",
            "cli commands",
            "terminal commands",
            "bash commands",
            "command line",
            "shell",
            "bash",
            "commands",
        ],
        "Write the commands in a ```bash code block with a short comment above each, then name the sources used.",
        free=("shell commands", "cli commands", "terminal commands", "bash commands"),
    ),
    _f(
        "unit_test",
        "Unit tests",
        "code",
        ["unit test", "unit tests", "test cases", "pytest"],
        "Write the tests in a fenced code block, one test per case the sources describe, then name the sources.",
        max_tokens=512,
        free=("unit test", "unit tests"),
    ),
    # ---- Files to download ----
    _f(
        "pdf",
        "PDF document",
        "file",
        ["pdf", "pdf document", "pdf file"],
        DOCUMENT_NOTE,
        render="file",
        target="pdf",
        max_tokens=512,
        free=("pdf document",),
    ),
    _f(
        "docx",
        "Word document",
        "file",
        ["word document", "word file", "word doc", "docx", "doc file", "ms word"],
        DOCUMENT_NOTE,
        render="file",
        target="docx",
        max_tokens=512,
        free=("word document", "word file"),
    ),
    _f(
        "xlsx",
        "Excel spreadsheet",
        "file",
        ["excel sheet", "excel spreadsheet", "excel file", "excel", "xlsx", "spreadsheet"],
        CONVERTED_NOTE,
        render="file",
        target="xlsx",
        max_tokens=512,
        free=("excel sheet", "excel spreadsheet", "excel file"),
    ),
    _f(
        "csv",
        "CSV file",
        "file",
        ["csv file", "csv"],
        CONVERTED_NOTE,
        render="file",
        target="csv",
        max_tokens=512,
        free=("csv file",),
    ),
    _f(
        "pptx",
        "PowerPoint",
        "file",
        ["powerpoint", "power point", "pptx", "ppt", "slides", "slide deck", "presentation"],
        "Write slides: each slide is a ## heading followed by three to five short bullet points with sources.",
        render="file",
        target="pptx",
        max_tokens=512,
        free=("slide deck",),
    ),
    _f(
        "markdown",
        "Markdown file",
        "file",
        ["markdown file", "markdown", "md file"],
        DOCUMENT_NOTE,
        render="file",
        target="md",
        max_tokens=512,
        free=("markdown file", "md file"),
    ),
    _f(
        "txt",
        "Plain text",
        "file",
        ["plain text", "text file", "txt file", "txt", "notepad"],
        "Write plain sentences and simple '- ' lists, without tables or other formatting.",
        render="file",
        target="txt",
        free=("plain text", "text file", "txt file"),
    ),
    _f(
        "rtf",
        "RTF document",
        "file",
        ["rtf", "rich text", "rich text format"],
        DOCUMENT_NOTE,
        render="file",
        target="rtf",
        max_tokens=512,
        free=("rich text format",),
    ),
    _f(
        "odt",
        "OpenDocument text",
        "file",
        ["odt", "opendocument text", "open document text", "libreoffice writer"],
        DOCUMENT_NOTE,
        render="file",
        target="odt",
        max_tokens=512,
        free=("opendocument text", "open document text"),
    ),
    _f(
        "ods",
        "OpenDocument spreadsheet",
        "file",
        ["ods", "opendocument spreadsheet", "open document spreadsheet", "libreoffice calc"],
        CONVERTED_NOTE,
        render="file",
        target="ods",
        max_tokens=512,
        free=("opendocument spreadsheet", "open document spreadsheet"),
    ),
    # ---- Web and interactive ----
    _f(
        "html_table",
        "HTML table",
        "web",
        ["html table"],
        "Write the answer as a Markdown table with a header row. " + TABLE_NOTE,
        render="html",
        max_tokens=512,
        free=("html table",),
    ),
    _f(
        "html",
        "HTML page",
        "web",
        [
            "html page",
            "html document",
            "html file",
            "web page",
            "webpage",
            "html",
            "css styled html",
            "styled html",
            "standalone html",
            "html code",
        ],
        DOCUMENT_NOTE,
        render="html",
        max_tokens=512,
        free=("html page", "html document", "css styled html", "styled html", "standalone html"),
    ),
    _f(
        "interactive_html",
        "Interactive HTML",
        "web",
        ["interactive html", "interactive page", "interactive web page"],
        SECTIONS_NOTE,
        render="html",
        max_tokens=512,
        free=("interactive html", "interactive page"),
    ),
    _f(
        "accordion",
        "Accordion",
        "web",
        ["accordion", "accordion layout", "expandable sections", "collapsible sections", "expandable", "collapsible"],
        SECTIONS_NOTE,
        render="accordion",
        max_tokens=512,
        free=("accordion layout", "expandable sections", "collapsible sections"),
    ),
    _f(
        "tabs",
        "Tabs",
        "web",
        ["tab based", "tab-based", "tabs", "tabbed", "tab layout", "tab view"],
        SECTIONS_NOTE,
        render="tabs",
        max_tokens=512,
        free=("tab based", "tab-based", "tab layout"),
    ),
    _f(
        "dashboard",
        "Dashboard",
        "web",
        ["dashboard", "metrics summary", "kpi view"],
        "Start with three to six key figures, one per line, written as '**Label:** value [S1]'. Then add a short "
        "Markdown table of details.",
        render="dashboard",
        max_tokens=512,
        free=("metrics summary",),
    ),
    _f(
        "quiz",
        "Quiz",
        "learning",
        [
            "quiz",
            "interactive quiz",
            "mcq",
            "mcqs",
            "multiple choice",
            "multiple-choice",
            "test me",
            "assessment questions",
        ],
        "Write three to five multiple-choice questions about the sources. For each: 'Q1. question' on one line, then "
        "four options on separate lines as 'A) ...', 'B) ...', 'C) ...', 'D) ...', then 'Answer: B' and a line "
        "'Source: [S1]'.",
        render="quiz",
        max_tokens=512,
        free=("interactive quiz", "mcq", "mcqs", "multiple choice", "multiple-choice", "test me"),
    ),
    _f(
        "timeline",
        "Timeline",
        "web",
        ["timeline", "timeline view", "chronological order", "chronological"],
        "Write one bullet per event in time order, as '- **when** - what happened [S1]'.",
        render="timeline",
        free=("timeline view", "chronological order"),
    ),
    _f(
        "cards",
        "Cards",
        "web",
        ["cards", "card layout", "cards layout", "card view"],
        SECTIONS_NOTE,
        render="cards",
        max_tokens=512,
        free=("card layout", "cards layout", "card view"),
    ),
    _f(
        "carousel",
        "Carousel",
        "web",
        ["carousel", "slider", "slideshow"],
        SECTIONS_NOTE,
        render="carousel",
        max_tokens=512,
        free=("carousel",),
    ),
    # ---- Charts. Mermaid draws these; the colours come from the chart palette in the chat screen. ----
    _f(
        "bar_chart",
        "Bar chart",
        "chart",
        ["bar chart", "bar graph", "column chart", "histogram", "interactive chart", "chart", "graph"],
        "Draw a Mermaid bar chart in a ```mermaid code block, starting with 'xychart-beta', then 'title \"...\"', "
        "'x-axis [\"a\", \"b\"]', 'y-axis \"unit\"' and 'bar [1, 2]', using numbers from the sources only. Below the "
        "block, name the sources used.",
        render="mermaid",
        free=("bar chart", "bar graph", "column chart", "histogram"),
    ),
    _f(
        "line_chart",
        "Line chart",
        "chart",
        ["line chart", "line graph", "trend chart", "trend line"],
        "Draw a Mermaid line chart in a ```mermaid code block, starting with 'xychart-beta', then 'title \"...\"', "
        "'x-axis [\"a\", \"b\"]', 'y-axis \"unit\"' and 'line [1, 2]', using numbers from the sources only. Below the "
        "block, name the sources used.",
        render="mermaid",
        free=("line chart", "line graph", "trend chart"),
    ),
    _f(
        "pie_chart",
        "Pie chart",
        "chart",
        ["pie chart", "pie graph", "donut chart", "doughnut chart"],
        "Draw a Mermaid pie chart in a ```mermaid code block, starting with 'pie showData', then 'title ...', then one "
        "'\"label\" : number' line per slice, using numbers from the sources only. Below the block, name the sources.",
        render="mermaid",
        free=("pie chart", "pie graph", "donut chart", "doughnut chart"),
    ),
    _f(
        "gantt",
        "Gantt chart",
        "chart",
        ["gantt chart", "gantt"],
        "Draw a Mermaid Gantt chart in a ```mermaid code block, starting with 'gantt', 'dateFormat YYYY-MM-DD' and "
        "'title ...', with 'section' lines and 'task name :a1, 2026-01-01, 7d' tasks. Below the block, name the "
        "sources used, like [S1].",
        render="mermaid",
        free=("gantt chart", "gantt"),
    ),
    _f(
        "sankey",
        "Sankey diagram",
        "chart",
        ["sankey diagram", "sankey"],
        "Draw a Mermaid Sankey diagram in a ```mermaid code block, starting with 'sankey-beta', then one "
        "'source,target,value' line per flow, using numbers from the sources only. Below the block, name the sources.",
        render="mermaid",
        free=("sankey diagram", "sankey"),
    ),
    _f(
        "network",
        "Network graph",
        "chart",
        ["network graph", "network diagram", "relationship graph", "graph diagram", "knowledge graph"],
        "Show the items and how they relate. Start the block with 'flowchart LR' and label each edge. " + MERMAID_NOTE,
        render="mermaid",
        free=("network graph", "network diagram", "relationship graph"),
    ),
    _f(
        "radar",
        "Radar chart",
        "chart",
        ["radar chart", "spider chart"],
        "Draw a Mermaid radar chart in a ```mermaid code block, starting with 'radar-beta', then 'axis a, b, c' and "
        "'curve name{1, 2, 3}', using numbers from the sources only. Below the block, name the sources.",
        render="mermaid",
        free=("radar chart", "spider chart"),
    ),
    _f(
        "scatter",
        "Scatter plot",
        "chart",
        ["scatter plot", "scatter chart", "scatter graph", "heatmap", "heat map"],
        "Write the values as a Markdown table, one row per point or cell, with a column per measure. " + TABLE_NOTE,
        max_tokens=512,
        free=("scatter plot", "scatter chart", "heatmap", "heat map"),
        note="Scatter plots and heatmaps are not drawn here, so the values are shown as a table.",
    ),
    # ---- Learning and interviews ----
    _f(
        "five_star",
        "Five-star priority",
        "learning",
        ["five star", "five-star", "5 star", "5-star", "star rating", "star priority", "must know"],
        "Write a Markdown table with columns Topic, Importance (★ to ★★★★★) and Why, most important first. "
        + TABLE_NOTE,
        max_tokens=512,
        free=("five star", "five-star", "5 star", "5-star", "star rating", "must know"),
    ),
    _f(
        "interview_qa",
        "Interview Q&A",
        "learning",
        [
            "interview q&a",
            "interview questions and answers",
            "interview questions",
            "interview preparation",
            "interview prep",
            "interview question and answer",
        ],
        "Write interview questions with answers: '**Q1.** question' on one line, then the answer with its source.",
        max_tokens=512,
        free=(
            "interview q&a",
            "interview questions and answers",
            "interview questions",
            "interview preparation",
            "interview prep",
        ),
    ),
    _f(
        "scenarios",
        "Real-world scenarios",
        "learning",
        [
            "real world scenarios",
            "real-world scenarios",
            "scenarios",
            "use cases",
            "real world examples",
            "real-world examples",
            "practical examples",
        ],
        "Write two to four short scenarios, each a ### heading, the situation and what the sources say to do.",
        max_tokens=512,
        free=(
            "real world scenarios",
            "real-world scenarios",
            "real world examples",
            "real-world examples",
            "practical examples",
        ),
    ),
    _f(
        "coding_challenges",
        "Coding challenges",
        "learning",
        ["coding challenges", "coding problems", "practice problems", "coding exercises", "exercises"],
        "Write two or three practice problems based on the sources, each with a ### heading, the task and a hint.",
        max_tokens=512,
        free=("coding challenges", "coding problems", "practice problems", "coding exercises"),
    ),
    _f(
        "flashcards",
        "Flashcards",
        "learning",
        ["flashcards", "flash cards", "flashcard", "flash card"],
        "Write four to eight flashcards. Each is a line 'Q: question' followed by a line 'A: answer [S1]'.",
        render="flashcards",
        free=("flashcards", "flash cards", "flashcard"),
    ),
    _f(
        "case_study",
        "Case study",
        "learning",
        ["case study", "case studies"],
        "Write a case study with ### Situation, ### Problem, ### What was done and ### Outcome, from the sources only.",
        max_tokens=512,
        free=("case study", "case studies"),
    ),
    _f(
        "deep_dive",
        "Technical deep dive",
        "learning",
        [
            "technical deep dive",
            "deep dive",
            "internals",
            "internal implementation",
            "under the hood",
            "how it works internally",
        ],
        "Explain the internals step by step with ### headings, covering each mechanism the sources describe.",
        max_tokens=512,
        free=("technical deep dive", "deep dive", "under the hood", "how it works internally"),
    ),
]
BY_KEY = {fmt.key: fmt for fmt in FORMATS}

# What may come before a format phrase ("in a", "as", "draw a", "download as") or after it ("format", "file").
CUE_BEFORE = (
    r"(?:\b(?:in|as|into|to|with|using|via|like)\s+(?:an?\s+|the\s+|some\s+)?"
    r"|\b(?:draw|create|make|generate|give|show|build|prepare|write|convert|export|download|save|produce|need|want)"
    r"\s+(?:it\s+|this\s+|that\s+|me\s+|us\s+)?(?:as\s+|in\s+|to\s+|into\s+)?(?:an?\s+|the\s+)?)"
)
CUE_AFTER = r"(?:\s+(?:format|form|layout|style|view|structure|file|document|version|type|mode))"


def _pattern(phrase: str, free: bool) -> re.Pattern:
    body = r"\s+".join(re.escape(word) for word in phrase.split())
    edge_start = r"\b" if phrase[0].isalnum() else ""
    edge_end = r"\b" if phrase[-1].isalnum() else ""
    core = f"{edge_start}{body}{edge_end}"
    if free:
        return re.compile(f"(?:{CUE_BEFORE})?{core}{CUE_AFTER}?", re.IGNORECASE)
    return re.compile(f"{CUE_BEFORE}{core}{CUE_AFTER}?|{core}{CUE_AFTER}", re.IGNORECASE)


# Longest phrases first, so that "comparison table" wins over "table" and "html table" over "html".
_PATTERNS = sorted(
    ((phrase, fmt, _pattern(phrase, phrase in fmt.free)) for fmt in FORMATS for phrase in fmt.phrases),
    key=lambda item: -len(item[0]),
)


def detect(question: str) -> tuple[AnswerFormat | None, str]:
    """The format the question asks for, and the question without the words that asked for it."""
    for _, fmt, pattern in _PATTERNS:
        match = pattern.search(question)
        if match:
            rest = f"{question[: match.start()]} {question[match.end() :]}"
            rest = re.sub(r"\s{2,}", " ", rest).strip()
            # Empty when the whole message was the request, as in "download this as pdf".
            return fmt, re.sub(r"\s+([?.!,])", r"\1", rest).strip(" ?.!,")
    return None, question


def describe(key: str) -> dict | None:
    """What the chat screen needs to know to show an answer in this format."""
    fmt = BY_KEY.get(key)
    if not fmt:
        return None
    return {"key": fmt.key, "label": fmt.label, "render": fmt.render, "target": fmt.target, "note": fmt.note}


# ---- Overflowing output ----


def is_degenerate(text: str) -> bool:
    """True when the model got stuck repeating itself, as it did with "```html" on every line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    run, longest = 1, 1
    for previous, line in zip(lines, lines[1:], strict=False):
        run = run + 1 if line == previous else 1
        longest = max(longest, run)
    return longest >= 5 or len(set(lines)) / len(lines) < 0.3


# ---- Structured data from Markdown: tables, then lists, then the text itself ----

SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")
ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*)$")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_tables(text: str) -> list[list[list[str]]]:
    """Each table as rows of cells, header row first."""
    tables, current = [], []
    for line in text.splitlines():
        if line.strip().startswith("|") and line.strip().endswith("|") and line.count("|") >= 2:
            if not SEPARATOR_RE.match(line):
                current.append(_cells(line))
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [table for table in tables if len(table) >= 2]


NUMBER_CELL_RE = re.compile(r"^\d+[.)]?$")


def without_repeated_rows(text: str) -> str:
    """Table rows that repeat an earlier row of the same table are dropped, and a "#" column is numbered again. Asked
    for a comparison table, the small model wrote its seven rows, then the same seven again. A repeat adds nothing,
    so taking it out cannot lose information."""
    out, seen, number, numbered = [], set(), 0, False
    for line in text.splitlines():
        stripped = line.strip()
        is_row = stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2
        if not is_row:
            seen, number, numbered = set(), 0, False
            out.append(line)
            continue
        if SEPARATOR_RE.match(line):
            out.append(line)
            continue
        cells = _cells(line)
        if not seen and not number:  # the header row
            numbered = cells[0] in ("#", "No", "No.", "S.No", "Sl")
            seen.add(("header",))
            out.append(line)
            continue
        key = tuple(cell.lower() for cell in (cells[1:] if numbered and NUMBER_CELL_RE.match(cells[0]) else cells))
        if key in seen:
            continue
        seen.add(key)
        number += 1
        if numbered and NUMBER_CELL_RE.match(cells[0]):
            line = "| " + " | ".join([str(number), *cells[1:]]) + " |"
        out.append(line)
    return "\n".join(out)


def _without_sources(value: str) -> tuple[str, list[str]]:
    found = [f"S{number}" for number in CITATION_RE.findall(value)]
    clean = CITATION_RE.sub("", value).replace("**", "").strip(" ,;")
    return re.sub(r"\s{2,}", " ", clean), found


def structured(text: str) -> list | dict:
    tables = markdown_tables(text)
    if tables:
        header, *rows = tables[0]
        keys = [name.replace("**", "").strip() or f"column {index + 1}" for index, name in enumerate(header)]
        items = []
        for row in rows:
            item, sources = {}, []
            for key, cell in zip(keys, row, strict=False):
                clean, found = _without_sources(cell)
                sources += found
                if key.lower() not in {"source", "sources"}:
                    item[key] = clean
            if sources:
                item["sources"] = sorted(set(sources), key=lambda name: int(name[1:]))
            items.append(item)
        return items
    entries = [match.group(1) for line in text.splitlines() if (match := ITEM_RE.match(line))]
    if entries:
        return [dict(zip(("text", "sources"), _without_sources(entry), strict=True)) for entry in entries]
    clean, found = _without_sources(text)
    return {"answer": clean, "sources": found}


def _xml_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "field"
    return f"_{name}" if name[0].isdigit() else name


def _to_xml(data) -> str:
    root = ElementTree.Element("answer")
    for entry in data if isinstance(data, list) else [data]:
        item = ElementTree.SubElement(root, "item")
        for key, value in entry.items():
            child = ElementTree.SubElement(item, _xml_name(key))
            child.text = ", ".join(value) if isinstance(value, list) else str(value)
    ElementTree.indent(root)
    return ElementTree.tostring(root, encoding="unicode")


def table_rows(text: str) -> list[list[str]]:
    """The answer as rows for a spreadsheet: its first table, or one row per line."""
    tables = markdown_tables(text)
    if tables:
        return [[_without_sources(cell)[0] for cell in row] for row in tables[0]]
    lines = [ITEM_RE.sub(r"\1", line).strip() for line in text.splitlines() if line.strip()]
    return [["Answer"], *[[_without_sources(line)[0]] for line in lines]]


def to_csv(text: str) -> str:
    out = io.StringIO()
    csv.writer(out).writerows(table_rows(text))
    return out.getvalue()


def convert(text: str, target: str) -> str:
    """The answer as a JSON, YAML, XML or CSV code block, converted by rule rather than written by the model."""
    if target == "csv":
        return f"```csv\n{to_csv(text).strip()}\n```"
    data = structured(text)
    if target == "json":
        body = json.dumps(data, indent=2, ensure_ascii=False)
    elif target == "yaml":
        body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    else:
        body = _to_xml(data)
    return f"```{target}\n{body}\n```"


# ---- Diagrams drawn by rule ----
# Asked for Mermaid, the small model looped through invented steps, wrote mind maps Mermaid cannot read, and put numbers
# where the sources had none. These formats ask it for plain Markdown instead (steps, a nested list, a table) and the
# diagram is drawn from that. The Markdown stays under the diagram, so the values can be read without the picture.

STEPS_NOTE = "Write the process as a numbered list of short steps, in order, each ending with its source, like [S1]."
NESTED_NOTE = (
    "Write the answer as a nested Markdown list: the topic first, its main points indented under it, and their "
    "details indented again. Keep each item to a few words."
)
EVENTS_NOTE = "Write one bullet per event in time order, as '- **when** - what happened [S1]'."
LABEL_NUMBER_NOTE = (
    "Write a Markdown table with two columns, a label and a number, one row per item, using only numbers that "
    "appear in the sources. Put the source, like [S1], in the label cell."
)
FLOWS_NOTE = (
    "Write a Markdown table with columns From, To and Amount, one row per flow, using numbers from the sources."
)
SERIES_NOTE = (
    "Write a Markdown table: the first column names each measure, then one column of numbers for each item compared, "
    "using numbers from the sources only."
)
DRAWN = {
    "flowchart": ("steps", STEPS_NOTE),
    "workflow": ("steps", STEPS_NOTE),
    "data_flow": ("steps", STEPS_NOTE),
    "mind_map": ("mindmap", NESTED_NOTE),
    "timeline_diagram": ("timeline", EVENTS_NOTE),
    "bar_chart": ("bar", LABEL_NUMBER_NOTE),
    "line_chart": ("line", LABEL_NUMBER_NOTE),
    "pie_chart": ("pie", LABEL_NUMBER_NOTE),
    "sankey": ("sankey", FLOWS_NOTE),
    "radar": ("radar", SERIES_NOTE),
}
FORMATS = [
    replace(fmt, instruction=DRAWN[fmt.key][1], render="drawn", target=DRAWN[fmt.key][0]) if fmt.key in DRAWN else fmt
    for fmt in FORMATS
]
BY_KEY = {fmt.key: fmt for fmt in FORMATS}
_PATTERNS = sorted(
    ((phrase, fmt, _pattern(phrase, phrase in fmt.free)) for fmt in FORMATS for phrase in fmt.phrases),
    key=lambda item: -len(item[0]),
)

MERMAID_START_RE = re.compile(
    r"^\s*(flowchart|graph|sequenceDiagram|classDiagram|erDiagram|stateDiagram(?:-v2)?|mindmap|timeline|gantt|pie|"
    r"xychart-beta|sankey-beta|radar-beta|quadrantChart|journey)\b",
    re.MULTILINE,
)
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
VALUE_START_RE = re.compile(r"^\s*(?:rs\.?|inr|₹|\$|€|£)?\s*-?\d", re.IGNORECASE)
MAX_NODES = 30


def fence_mermaid(text: str) -> str:
    """The model sometimes writes a diagram without its ```mermaid fence, so it showed as plain text."""
    if "```" in text:
        return text
    match = MERMAID_START_RE.search(text)
    if not match:
        return text
    return f"{text[: match.start()].rstrip()}\n\n```mermaid\n{text[match.start() :].strip()}\n```".strip()


def _plain(value: str, limit: int = 60) -> str:
    value = CITATION_RE.sub("", value)
    value = re.sub(r"[*_`#]", "", value)
    value = re.sub(r"\s+", " ", value).strip(" .;:-")
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _quoted(value: str) -> str:
    return '"' + _plain(value).replace('"', "'") + '"'


def _bare(value: str) -> str:
    """Text for diagram types that have no quoting: no brackets, colons or commas, which they read as syntax."""
    return re.sub(r"\s+", " ", re.sub(r"[()\[\]{}:;,\"<>]", " ", _plain(value))).strip() or "item"


def _number(cell: str) -> float | None:
    match = NUMBER_RE.search(CITATION_RE.sub("", cell))
    return float(match.group().replace(",", "")) if match else None


def _value(cell: str) -> float | None:
    """A cell's number, if the cell is a value ("18", "6 days", "Rs. 57,559.00") rather than a label with a number in
    it: "Feb 2026" is a label, and reading 2026 as its value left a salary chart with nothing to draw."""
    if not VALUE_START_RE.match(CITATION_RE.sub("", cell)):
        return None
    return _number(cell)


def _number_text(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:g}"


def _labelled_numbers(text: str) -> tuple[str, list[tuple[str, float]]]:
    """(name of the number column, [(label, number)]) from the answer's first table."""
    tables = markdown_tables(text)
    if not tables:
        return "", []
    header, *rows = tables[0]
    pairs = []
    unit = ""
    for row in rows:
        numbers = [(index, _value(cell)) for index, cell in enumerate(row) if _value(cell) is not None]
        texts = [cell for cell in row if _value(cell) is None and _plain(cell)]
        if numbers and texts:
            index, value = numbers[0]
            unit = unit or (header[index] if index < len(header) else "")
            pairs.append((texts[0], value))
    return _plain(unit, 30) or "Value", pairs[:MAX_NODES]


def _steps(text: str, title: str) -> str:
    steps = [
        match.group(1) for line in text.splitlines() if (match := ITEM_RE.match(line)) and not line.startswith(" ")
    ]
    if len(steps) < 2:
        return ""
    lines = ["flowchart TD"]
    for number, step in enumerate(steps[:MAX_NODES], start=1):
        lines.append(f"  s{number}[{_quoted(step)}]")
    lines += [f"  s{number} --> s{number + 1}" for number in range(1, min(len(steps), MAX_NODES))]
    return "\n".join(lines)


def _mindmap(text: str, title: str) -> str:
    items = []
    for line in text.splitlines():
        match = re.match(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$", line.expandtabs(2))
        if match and _plain(match.group(2)):
            items.append((len(match.group(1)), _bare(match.group(2))))
    if len(items) < 2:
        return ""
    levels = sorted({indent for indent, _ in items})
    depth = {indent: levels.index(indent) for indent in levels}
    nodes = [(depth[indent], label) for indent, label in items][:MAX_NODES]
    top = [label for level, label in nodes if level == 0]
    if len(top) == 1:
        root, nodes = top[0], [(level - 1, label) for level, label in nodes[1:]]
    else:
        root = _bare(title) or "Topic"
    lines = ["mindmap", f"  root(({root}))"]
    lines += ["    " + "  " * level + label for level, label in nodes]
    return "\n".join(lines)


def _timeline(text: str, title: str) -> str:
    events = []
    for line in text.splitlines():
        match = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(?:\*\*(.+?)\*\*|([^:–—-]{1,40}?))\s*[:–—-]\s*(.+)$", line)
        if match:
            events.append((_bare(match.group(1) or match.group(2)), _bare(match.group(3))))
    if len(events) < 2:
        return ""
    return "\n".join(
        ["timeline", f"  title {_bare(title)}", *[f"  {when} : {what}" for when, what in events[:MAX_NODES]]]
    )


def _pie(text: str, title: str) -> str:
    _, pairs = _labelled_numbers(text)
    pairs = [(label, value) for label, value in pairs if value > 0]
    if len(pairs) < 2:
        return ""
    return "\n".join(
        ["pie showData", f"  title {_bare(title)}", *[f"  {_quoted(label)} : {_number_text(v)}" for label, v in pairs]]
    )


def _xy(text: str, title: str, mark: str) -> str:
    unit, pairs = _labelled_numbers(text)
    if len(pairs) < 2:
        return ""
    labels = ", ".join(_quoted(label) for label, _ in pairs)
    values = ", ".join(_number_text(value) for _, value in pairs)
    return "\n".join(
        [
            "xychart-beta",
            f"  title {_quoted(title)}",
            f"  x-axis [{labels}]",
            f"  y-axis {_quoted(unit)}",
            f"  {mark} [{values}]",
        ]
    )


def _sankey(text: str, title: str) -> str:
    tables = markdown_tables(text)
    flows = []
    for row in tables[0][1:] if tables else []:
        if len(row) >= 3 and (value := _number(row[2])) is not None and value > 0:
            flows.append(f"{_bare(row[0])},{_bare(row[1])},{_number_text(value)}")
    return "\n".join(["sankey-beta", "", *flows[:MAX_NODES]]) if len(flows) >= 2 else ""


def _radar(text: str, title: str) -> str:
    tables = markdown_tables(text)
    if not tables or len(tables[0][0]) < 2:
        return ""
    header, *rows = tables[0]
    rows = [row for row in rows if len(row) == len(header) and all(_number(cell) is not None for cell in row[1:])]
    if len(rows) < 3:
        return ""
    axes = ", ".join(f"a{index}[{_quoted(row[0])}]" for index, row in enumerate(rows))
    curves = [
        f"  curve c{column}[{_quoted(header[column])}]{{{', '.join(_number_text(_number(row[column])) for row in rows)}}}"
        for column in range(1, len(header))
    ]
    return "\n".join(["radar-beta", f"  title {_bare(title)}", f"  axis {axes}", *curves])


BUILDERS = {
    "steps": _steps,
    "mindmap": _mindmap,
    "timeline": _timeline,
    "pie": _pie,
    "bar": lambda text, title: _xy(text, title, "bar"),
    "line": lambda text, title: _xy(text, title, "line"),
    "sankey": _sankey,
    "radar": _radar,
}


TITLE_FILLER = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "create",
        "display",
        "draw",
        "for",
        "generate",
        "give",
        "in",
        "is",
        "list",
        "make",
        "me",
        "of",
        "on",
        "please",
        "pls",
        "show",
        "the",
        "to",
        "what",
    ]
)


def diagram_title(question: str) -> str:
    """ "show the vvdn leave types" -> "Vvdn leave types": the question, without the words that asked for a picture."""
    words = re.sub(r"[?.!]+$", "", question).split()
    while words and words[0].lower() in TITLE_FILLER:
        words.pop(0)
    title = " ".join(words) or "Answer"
    return title[:1].upper() + title[1:]


def lay_out(text: str, layout: str, question: str = "") -> str:
    """Finished Markdown in the layout asked for, where that layout is built by rule (data formats and diagrams)."""
    fmt = BY_KEY.get(layout)
    if fmt and fmt.render == "data":
        return convert(text, fmt.target)
    if fmt and fmt.render == "drawn":
        return draw(text, fmt.target, title=question)
    if fmt and fmt.render == "mermaid":
        return fence_mermaid(text)
    return text


def draw(text: str, kind: str, title: str = "") -> str:
    """The diagram drawn from the answer's steps, list or table, with that Markdown kept below it. If the structure is
    not there, the model's own diagram if it wrote one, or its text as it is."""
    code = BUILDERS[kind](text, diagram_title(title))
    return f"```mermaid\n{code}\n```\n\n{text}" if code else fence_mermaid(text)

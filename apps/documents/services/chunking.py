"""Three chunking strategies, chosen per RagConfig so that they can be compared by evaluation."""

import re
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from apps.common.utils import count_tokens
from apps.documents.services.parser import ParsedPage

HEADING_RE = re.compile(r"^(#{1,6} .+|\d+(\.\d+)*\.? [A-Z].{0,80}|[A-Z][A-Z \-&]{3,80})$", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
SEMANTIC_BREAK_PERCENTILE = 25


@dataclass
class ChunkDraft:
    text: str
    page_number: int
    metadata: dict = field(default_factory=dict)


def _splitter(size: int, overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_recursive(pages: list[ParsedPage], size: int, overlap: int) -> list[ChunkDraft]:
    splitter = _splitter(size, overlap)
    return [ChunkDraft(text, page.page_number) for page in pages for text in splitter.split_text(page.content)]


def chunk_structure(pages: list[ParsedPage], size: int, overlap: int) -> list[ChunkDraft]:
    """Split at headings first, so a chunk stays within one section. The heading is repeated in each
    chunk, because it often holds the words a question will use."""
    splitter = _splitter(size, overlap)
    drafts = []
    for page in pages:
        starts = [match.start() for match in HEADING_RE.finditer(page.content)]
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        for begin, end in zip(starts, [*starts[1:], len(page.content)], strict=True):
            section = page.content[begin:end].strip()
            if not section:
                continue
            first_line = section.split("\n", 1)[0]
            heading = first_line.lstrip("# ").strip() if HEADING_RE.fullmatch(first_line) else ""
            for text in splitter.split_text(section):
                prefixed = text if not heading or heading in text else f"{heading}\n{text}"
                drafts.append(ChunkDraft(prefixed, page.page_number, {"section": heading}))
    return drafts


def chunk_semantic(pages: list[ParsedPage], size: int, embed_fn) -> list[ChunkDraft]:
    """Start a new chunk where the similarity of neighbouring sentences drops, or at the size limit."""
    import numpy as np

    drafts = []
    for page in pages:
        sentences = [s.strip() for s in SENTENCE_RE.split(page.content) if s.strip()]
        if not sentences:
            continue
        if len(sentences) == 1:
            drafts.append(ChunkDraft(sentences[0], page.page_number))
            continue
        vectors = np.array(embed_fn(sentences))
        similarities = (vectors[:-1] * vectors[1:]).sum(axis=1)
        threshold = float(np.percentile(similarities, SEMANTIC_BREAK_PERCENTILE))
        current, current_tokens = [sentences[0]], count_tokens(sentences[0])
        for sentence, similarity in zip(sentences[1:], similarities, strict=True):
            tokens = count_tokens(sentence)
            if similarity < threshold or current_tokens + tokens > size:
                drafts.append(ChunkDraft(" ".join(current), page.page_number))
                current, current_tokens = [], 0
            current.append(sentence)
            current_tokens += tokens
        drafts.append(ChunkDraft(" ".join(current), page.page_number))
    return drafts


def chunk_pages(pages: list[ParsedPage], config, embed_fn=None) -> list[ChunkDraft]:
    if config.chunk_strategy == "structure":
        return chunk_structure(pages, config.chunk_size, config.chunk_overlap)
    if config.chunk_strategy == "semantic":
        return chunk_semantic(pages, config.chunk_size, embed_fn)
    return chunk_recursive(pages, config.chunk_size, config.chunk_overlap)

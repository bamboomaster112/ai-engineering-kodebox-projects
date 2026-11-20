"""Split markdown articles into retrievable chunks, one per section (long sections are windowed)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    body: str
    path: str


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str


_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.S)


def load_documents(kb_dir: Path) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        m = _FRONTMATTER.match(raw)
        if m:
            for line in m.group(1).splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            raw = raw[m.end() :]
        docs.append(
            Document(
                doc_id=meta.get("id", path.stem),
                title=meta.get("title", path.stem.replace("-", " ").title()),
                body=raw.strip(),
                path=str(path.name),
            )
        )
    return docs


def _windows(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if len(words) <= size:
        return [words]
    out, start = [], 0
    while start < len(words):
        out.append(words[start : start + size])
        if start + size >= len(words):
            break
        start += size - overlap
    return out


def chunk_document(doc: Document, max_words: int = 160, overlap: int = 30) -> list[Chunk]:
    sections: list[tuple[str, str]] = []
    current_title, current_lines = "Overview", []
    for line in doc.body.splitlines():
        heading = re.match(r"^#{1,3}\s+(.*)", line)
        if heading:
            if any(s.strip() for s in current_lines):
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title, current_lines = heading.group(1).strip(), []
        else:
            current_lines.append(line)
    if any(s.strip() for s in current_lines):
        sections.append((current_title, "\n".join(current_lines).strip()))

    chunks: list[Chunk] = []
    for section, text in sections:
        for window in _windows(text.split(), max_words, overlap):
            n = len(chunks) + 1
            chunks.append(Chunk(id=f"{doc.doc_id}#{n}", doc_id=doc.doc_id, title=doc.title,
                                section=section, text=" ".join(window)))
    return chunks

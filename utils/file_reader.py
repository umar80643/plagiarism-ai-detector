"""Extract plain text from .txt, .pdf, and .docx sources.

Accepts three kinds of input, unlike the original version which only worked
with a file path:
- a path string / pathlib.Path
- a Streamlit UploadedFile (has .name and is file-like)
- any other file-like object with .read()

This fixes a real inconsistency in the original project: its test script
called extract_text() with a path string, while app.py called it with the
Streamlit uploader's file object directly -- only one of those could have
worked against a path-only implementation.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import Any

import docx
from pypdf import PdfReader


class UnsupportedFileTypeError(ValueError):
    pass


def _name_and_bytes(source: Any) -> tuple[str, bytes]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return path.name, path.read_bytes()
    if hasattr(source, "read"):
        name = getattr(source, "name", "") or getattr(source, "filename", "")
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)  # leave the stream usable for a second read
        return name, data if isinstance(data, bytes) else data.encode("utf-8")
    raise TypeError(f"Unsupported input to extract_text: {type(source)!r}")


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Core extraction logic, usable directly once bytes are already in hand
    (e.g. after `await upload_file.read()` in an async FastAPI endpoint).
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return data.decode("utf-8", errors="replace")
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        document = docx.Document(io.BytesIO(data))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix or '(none)'}")


def extract_text(source: Any) -> str:
    """source may be a path, a Streamlit UploadedFile, or any file-like
    object exposing .read(). The extension is taken from the path / the
    object's .name (or .filename) attribute.
    """
    name, data = _name_and_bytes(source)
    return extract_text_from_bytes(data, name)

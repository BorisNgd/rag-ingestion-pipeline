"""
Parser Registry
Maps FileType → async parser instance.
All parsers implement the same interface for clean substitutability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.domain.entities.models import FileType
from src.domain.repositories.interfaces import ObjectStorageRepository

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Base parser interface
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseParser(ABC):
    @abstractmethod
    async def parse(self, file_key: str, metadata: dict) -> ParseResult: ...


# ---------------------------------------------------------------------------
# Concrete parsers (stubs — production impls use dedicated libraries)
# ---------------------------------------------------------------------------

class PDFParser(BaseParser):
    """Uses PyMuPDF (fitz) for text + layout extraction."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        import fitz  # PyMuPDF
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        doc = fitz.open(stream=raw, filetype="pdf")

        pages = []
        page_meta = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(text)
            page_meta.append({"page": i + 1, "chars": len(text)})

        full_text = "\n\n".join(pages)
        return ParseResult(
            text=full_text,
            metadata={"pages": page_meta, "total_pages": len(doc)},
        )


class DocxParser(BaseParser):
    """Uses python-docx for Word document extraction."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        import io
        import docx
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        document = docx.Document(io.BytesIO(raw))
        paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
        return ParseResult(
            text="\n\n".join(paragraphs),
            metadata={"paragraph_count": len(paragraphs)},
        )


class ExcelParser(BaseParser):
    """Uses pandas for Excel/CSV tabular extraction."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        import io
        import pandas as pd
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        ext = file_key.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            df = pd.read_csv(io.BytesIO(raw))
        else:
            df = pd.read_excel(io.BytesIO(raw))

        text = df.to_string(index=False)
        return ParseResult(
            text=text,
            metadata={"rows": len(df), "columns": list(df.columns)},
        )


class HTMLParser(BaseParser):
    """Uses BeautifulSoup for HTML content extraction."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        from bs4 import BeautifulSoup
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        soup = BeautifulSoup(raw.decode("utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        title = soup.title.string if soup.title else ""
        return ParseResult(text=text, metadata={"title": title})


class ImageParser(BaseParser):
    """Uses Tesseract OCR via pytesseract."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        import io
        import pytesseract
        from PIL import Image
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        image = Image.open(io.BytesIO(raw))
        text = pytesseract.image_to_string(image)
        return ParseResult(text=text, metadata={"ocr": True})


class AudioParser(BaseParser):
    """Uses faster-whisper for local audio transcription."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        import tempfile
        import os
        from faster_whisper import WhisperModel
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as f:
            f.write(raw)
            tmp_path = f.name

        try:
            model = WhisperModel("base", device="cpu", compute_type="int8")
            segments, info = model.transcribe(tmp_path, beam_size=5)
            text = " ".join(seg.text for seg in segments)
            return ParseResult(
                text=text,
                metadata={"language": info.language, "duration": info.duration},
            )
        finally:
            os.unlink(tmp_path)


class CodeParser(BaseParser):
    """Extracts code with language detection."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        text = raw.decode("utf-8", errors="replace")
        ext = file_key.rsplit(".", 1)[-1].lower() if "." in file_key else "unknown"
        return ParseResult(
            text=text,
            metadata={"language": ext, "lines": text.count("\n")},
        )


class MarkdownParser(BaseParser):
    """Parses Markdown with header structure preservation."""

    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._storage = storage

    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        from config.settings import get_settings
        settings = get_settings()

        raw = await self._storage.download(settings.minio.bucket_raw, file_key)
        text = raw.decode("utf-8", errors="replace")
        headers = [l for l in text.splitlines() if l.startswith("#")]
        return ParseResult(
            text=text,
            metadata={"headers": headers[:20]},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ParserRegistry:
    def __init__(self, storage: ObjectStorageRepository) -> None:
        self._parsers: dict[FileType, BaseParser] = {
            FileType.PDF: PDFParser(storage),
            FileType.DOCX: DocxParser(storage),
            FileType.EXCEL: ExcelParser(storage),
            FileType.HTML: HTMLParser(storage),
            FileType.IMAGE: ImageParser(storage),
            FileType.AUDIO: AudioParser(storage),
            FileType.CODE: CodeParser(storage),
            FileType.MARKDOWN: MarkdownParser(storage),
        }

    def get(self, file_type: FileType) -> BaseParser | None:
        return self._parsers.get(file_type)

    def register(self, file_type: FileType, parser: BaseParser) -> None:
        """Allow runtime parser registration for extensibility."""
        self._parsers[file_type] = parser

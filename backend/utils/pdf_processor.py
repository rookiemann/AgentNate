"""
PDF Processing Utilities

Extracts text content from PDF files and chunks it for RAG.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger("utils.pdf_processor")


@dataclass
class PdfChunk:
    """A chunk of text from a PDF."""
    text: str
    page: int
    chunk_index: int
    filename: str
    start_char: int = 0
    end_char: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "filename": self.filename,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


@dataclass
class PdfProcessingResult:
    """Result of PDF processing."""
    success: bool
    filename: str
    page_count: int
    chunks: List[PdfChunk] = field(default_factory=list)
    total_chars: int = 0
    token_estimate: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "filename": self.filename,
            "page_count": self.page_count,
            "chunk_count": len(self.chunks),
            "total_chars": self.total_chars,
            "token_estimate": self.token_estimate,
            "error": self.error,
            "chunks": [c.to_dict() for c in self.chunks] if self.success else [],
        }


def estimate_tokens(text: str) -> int:
    """Rough token estimation (~4 chars per token for English)."""
    return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    page: int = 1,
    filename: str = ""
) -> List[PdfChunk]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Target size in tokens (~4 chars per token)
        chunk_overlap: Overlap between chunks in tokens
        page: Page number for metadata
        filename: Source filename for metadata

    Returns:
        List of PdfChunk objects
    """
    if not text.strip():
        return []

    # Convert token counts to character counts
    char_chunk_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + char_chunk_size

        # Try to break at sentence or word boundary
        if end < len(text):
            # Look for sentence boundary
            for sep in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
                last_sep = text[start:end].rfind(sep)
                if last_sep > char_chunk_size // 2:
                    end = start + last_sep + len(sep)
                    break
            else:
                # Fall back to word boundary
                last_space = text[start:end].rfind(' ')
                if last_space > char_chunk_size // 2:
                    end = start + last_space + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(PdfChunk(
                text=chunk_text,
                page=page,
                chunk_index=chunk_index,
                filename=filename,
                start_char=start,
                end_char=end,
            ))
            chunk_index += 1

        # Move start position with overlap
        start = end - char_overlap
        if start <= chunks[-1].start_char if chunks else 0:
            start = end  # Prevent infinite loop

    return chunks


async def extract_and_chunk_pdf(
    content: bytes,
    filename: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    max_pages: int = 200
) -> PdfProcessingResult:
    """
    Extract text from PDF and split into overlapping chunks.

    Args:
        content: Raw PDF bytes
        filename: Original filename
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_pages: Maximum pages to process

    Returns:
        PdfProcessingResult with chunks and metadata
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return PdfProcessingResult(
            success=False,
            filename=filename,
            page_count=0,
            error="PDF processing not available (PyMuPDF not installed)"
        )

    try:
        # Open PDF from bytes
        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)

        if page_count > max_pages:
            doc.close()
            return PdfProcessingResult(
                success=False,
                filename=filename,
                page_count=page_count,
                error=f"PDF has {page_count} pages, max allowed is {max_pages}"
            )

        all_chunks = []
        total_chars = 0

        # Extract text from each page and chunk it
        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()

            if page_text.strip():
                total_chars += len(page_text)
                page_chunks = chunk_text(
                    page_text,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    page=page_num + 1,  # 1-indexed
                    filename=filename
                )
                all_chunks.extend(page_chunks)

        doc.close()

        # Re-index chunks globally
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i

        token_estimate = estimate_tokens(
            " ".join(c.text for c in all_chunks)
        )

        logger.info(
            f"Processed PDF '{filename}': {page_count} pages, "
            f"{len(all_chunks)} chunks, ~{token_estimate} tokens"
        )

        return PdfProcessingResult(
            success=True,
            filename=filename,
            page_count=page_count,
            chunks=all_chunks,
            total_chars=total_chars,
            token_estimate=token_estimate,
        )

    except Exception as e:
        logger.error(f"PDF extraction error for '{filename}': {e}")
        return PdfProcessingResult(
            success=False,
            filename=filename,
            page_count=0,
            error=str(e)
        )


async def extract_pdf_text_simple(
    content: bytes,
    filename: str,
    max_pages: int = 50
) -> Dict[str, Any]:
    """
    Simple extraction without chunking (for fallback mode).

    Returns full text with page markers.
    """
    try:
        import fitz
    except ImportError:
        return {
            "success": False,
            "error": "PyMuPDF not installed"
        }

    try:
        doc = fitz.open(stream=content, filetype="pdf")
        page_count = len(doc)

        if page_count > max_pages:
            doc.close()
            return {
                "success": False,
                "filename": filename,
                "page_count": page_count,
                "error": f"PDF has {page_count} pages, max allowed is {max_pages}"
            }

        text_parts = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")

        doc.close()

        full_text = "\n\n".join(text_parts)
        token_estimate = estimate_tokens(full_text)

        return {
            "success": True,
            "filename": filename,
            "page_count": page_count,
            "text": full_text,
            "token_estimate": token_estimate,
            "preview": full_text[:500] if full_text else ""
        }

    except Exception as e:
        return {
            "success": False,
            "filename": filename,
            "error": str(e)
        }

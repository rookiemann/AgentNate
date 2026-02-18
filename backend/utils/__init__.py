"""
Backend utility modules for PDF processing, embeddings, and vector storage.
"""

from .pdf_processor import extract_and_chunk_pdf, PdfProcessingResult
from .vector_store import InMemoryVectorStore
from .embedding_manager import EmbeddingManager

__all__ = [
    "extract_and_chunk_pdf",
    "PdfProcessingResult",
    "InMemoryVectorStore",
    "EmbeddingManager",
]

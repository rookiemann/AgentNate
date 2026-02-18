"""
In-Memory Vector Store

Simple numpy-based vector storage for session-scoped RAG.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("utils.vector_store")


class InMemoryVectorStore:
    """
    Simple in-memory vector store using numpy for cosine similarity search.

    Designed for session-scoped RAG where vectors don't need to persist
    beyond the chat session.
    """

    def __init__(self):
        self.embeddings: List[np.ndarray] = []
        self.chunks: List[Dict[str, Any]] = []
        self.filenames: set = set()
        self._embedding_dim: Optional[int] = None

    def add(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """
        Add chunks with their embeddings.

        Args:
            chunks: List of chunk dicts with 'text', 'page', 'filename', etc.
            embeddings: List of embedding vectors (same length as chunks)

        Returns:
            Number of chunks added
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have same length"
            )

        if not chunks:
            return 0

        # Track embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = len(embeddings[0])
        elif len(embeddings[0]) != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._embedding_dim}, "
                f"got {len(embeddings[0])}"
            )

        for chunk, embedding in zip(chunks, embeddings):
            self.chunks.append(chunk)
            self.embeddings.append(np.array(embedding, dtype=np.float32))
            if "filename" in chunk:
                self.filenames.add(chunk["filename"])

        logger.info(
            f"Added {len(chunks)} chunks to vector store. "
            f"Total: {len(self.chunks)} chunks from {len(self.filenames)} files"
        )

        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filename_filter: Optional[str] = None,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find most similar chunks using cosine similarity.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filename_filter: Only return chunks from this file
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of chunk dicts with added 'score' field, sorted by relevance
        """
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm

        # Calculate cosine similarities
        scores = []
        for i, emb in enumerate(self.embeddings):
            # Skip if filename filter doesn't match
            if filename_filter and self.chunks[i].get("filename") != filename_filter:
                continue

            # Normalize embedding
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                scores.append((i, 0.0))
                continue

            # Cosine similarity
            similarity = np.dot(query_vec, emb / emb_norm)
            scores.append((i, float(similarity)))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k above threshold
        results = []
        for idx, score in scores[:top_k]:
            if score < score_threshold:
                continue
            result = dict(self.chunks[idx])
            result["score"] = round(score, 4)
            results.append(result)

        return results

    def remove_file(self, filename: str) -> int:
        """
        Remove all chunks from a specific file.

        Args:
            filename: Filename to remove

        Returns:
            Number of chunks removed
        """
        if filename not in self.filenames:
            return 0

        # Find indices to remove
        indices_to_remove = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.get("filename") == filename
        ]

        # Remove in reverse order to preserve indices
        for idx in reversed(indices_to_remove):
            del self.chunks[idx]
            del self.embeddings[idx]

        self.filenames.discard(filename)

        logger.info(
            f"Removed {len(indices_to_remove)} chunks for '{filename}'. "
            f"Remaining: {len(self.chunks)} chunks"
        )

        return len(indices_to_remove)

    def clear(self):
        """Clear all stored data."""
        count = len(self.chunks)
        self.embeddings = []
        self.chunks = []
        self.filenames = set()
        self._embedding_dim = None
        logger.info(f"Cleared vector store ({count} chunks removed)")

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_chunks": len(self.chunks),
            "total_files": len(self.filenames),
            "filenames": list(self.filenames),
            "embedding_dim": self._embedding_dim,
            "memory_estimate_mb": self._estimate_memory_mb(),
        }

    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        if not self.embeddings or self._embedding_dim is None:
            return 0.0

        # Embeddings: float32 = 4 bytes per value
        embedding_bytes = len(self.embeddings) * self._embedding_dim * 4

        # Rough estimate for chunk text (average 500 chars per chunk)
        text_bytes = sum(len(c.get("text", "")) for c in self.chunks)

        total_bytes = embedding_bytes + text_bytes
        return round(total_bytes / (1024 * 1024), 2)

    def __len__(self) -> int:
        return len(self.chunks)

    def __bool__(self) -> bool:
        return len(self.chunks) > 0

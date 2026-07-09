"""
RAG (Retrieval-Augmented Generation) tool.
Ingests financial documents (PDFs, text) into ChromaDB and retrieves
relevant chunks for a given query.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from config import config

logger = logging.getLogger(__name__)


class RAGTool:
    """
    Vector-store backed document retrieval for financial documents.

    Usage:
        rag = RAGTool(collection_name="it_sector")
        rag.ingest_pdf("path/to/annual_report.pdf", metadata={"company": "TCS"})
        results = rag.retrieve("revenue growth and margins", top_k=5)
    """

    def __init__(self, collection_name: str = "financial_docs") -> None:
        self.collection_name = collection_name
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

        # Use sentence-transformers for local embeddings (no API cost)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "RAG collection '%s' ready (%d docs).",
            collection_name,
            self.collection.count(),
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_pdf(
        self,
        pdf_path: str,
        metadata: dict[str, str] | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> int:
        """
        Parse a PDF and add chunked text to the vector store.

        Returns:
            Number of chunks added.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(str(path))
        full_text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        chunks = self._chunk_text(full_text, chunk_size, chunk_overlap)
        return self._add_chunks(chunks, source=str(path), extra_meta=metadata or {})

    def ingest_text(
        self,
        text: str,
        source: str = "manual",
        metadata: dict[str, str] | None = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ) -> int:
        """Ingest raw text into the vector store."""
        chunks = self._chunk_text(text, chunk_size, chunk_overlap)
        return self._add_chunks(chunks, source=source, extra_meta=metadata or {})

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a query.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            where: Optional ChromaDB metadata filter.

        Returns:
            List of dicts with keys: content, source, metadata, distance.
        """
        k = top_k or config.RAG_TOP_K
        count = self.collection.count()
        if count == 0:
            logger.warning("RAG collection '%s' is empty.", self.collection_name)
            return []

        k = min(k, count)
        query_params: dict[str, Any] = {
            "query_texts": [query],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_params["where"] = where

        results = self.collection.query(**query_params)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "content": doc,
                "source": meta.get("source", ""),
                "metadata": meta,
                "distance": dist,
            }
            for doc, meta, dist in zip(docs, metas, distances)
        ]

    def format_retrieved_as_text(self, results: list[dict[str, Any]]) -> str:
        """Format retrieved chunks into a readable context block."""
        if not results:
            return "No relevant documents found in the knowledge base."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            parts.append(f"[Doc {i} | Source: {source}]\n{r['content']}")
        return "\n\n---\n\n".join(parts)

    def collection_stats(self) -> dict[str, Any]:
        """Return basic stats about the collection."""
        return {
            "collection": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": config.CHROMA_PERSIST_DIR,
        }

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _chunk_text(
        self, text: str, chunk_size: int, overlap: int
    ) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def _add_chunks(
        self,
        chunks: list[str],
        source: str,
        extra_meta: dict[str, str],
    ) -> int:
        """Add text chunks to ChromaDB with metadata."""
        if not chunks:
            return 0

        # Generate unique IDs based on source + chunk index
        base_id = source.replace("/", "_").replace("\\", "_").replace(".", "_")
        ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, **extra_meta} for _ in chunks]

        # Upsert to avoid duplicates on re-ingestion
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks from '%s' to '%s'.", len(chunks), source, self.collection_name)
        return len(chunks)


# ── Sector-specific RAG instances ────────────────────────────────────────────

_rag_instances: dict[str, RAGTool] = {}


def get_rag_tool(sector: str = "general") -> RAGTool:
    """Return a sector-specific RAG instance (cached)."""
    key = sector.lower()
    if key not in _rag_instances:
        _rag_instances[key] = RAGTool(collection_name=f"{key}_financial_docs")
    return _rag_instances[key]

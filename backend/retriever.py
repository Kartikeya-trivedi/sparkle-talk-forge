"""
KTGPT v2 — Hybrid Retrieval with Reciprocal Rank Fusion
=========================================================
Production retrieval pipeline combining:

1. Dense retrieval: multilingual-e5-large embeddings in Qdrant
2. Sparse retrieval: BM25 (rank_bm25) for keyword matching
3. RRF fusion: Reciprocal Rank Fusion to merge both ranked lists
4. Cross-encoder reranking: ms-marco reranker on fused results

Qdrant is used as the primary vector store with support for both
dense and sparse vectors. BM25 runs in-memory via rank_bm25.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Data Models
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    """A single retrieved chunk with metadata."""
    text: str
    score: float
    source: str = ""            # e.g., filename or "web_search"
    chunk_idx: int = 0
    retrieval_method: str = ""  # "dense", "bm25", "hybrid"


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid Retriever
# ─────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    """Hybrid retrieval combining dense vectors (Qdrant) + sparse (BM25).

    Pipeline:
        1. BM25 top-N sparse search
        2. Qdrant dense vector search
        3. Reciprocal Rank Fusion (RRF) to merge results
        4. Cross-encoder reranking on top fused results
    """

    # Qdrant collection config
    COLLECTION_NAME = "ktgpt_rag"
    DENSE_VECTOR_SIZE = 1024  # multilingual-e5-large output dimension
    DENSE_DISTANCE = "Cosine"

    def __init__(self, embedder, reranker, qdrant_path: str = None, qdrant_url: str = None, qdrant_port: int = 6333):
        """
        Args:
            embedder: SentenceTransformer (multilingual-e5-large)
            reranker: CrossEncoder (ms-marco-MiniLM-L-6-v2)
            qdrant_path: Local disk path for embedded Qdrant (preferred)
            qdrant_url: Remote Qdrant server URL (alternative)
            qdrant_port: Remote Qdrant server port
        """
        self.embedder = embedder
        self.reranker = reranker

        # BM25 state (in-memory)
        self._bm25 = None
        self._bm25_corpus: list[str] = []      # tokenized corpus for BM25
        self._all_chunks: list[str] = []        # raw chunk texts
        self._all_sources: list[str] = []       # source filenames
        self._doc_count = 0
        self._chunk_count = 0

        # Qdrant client (lazy init)
        self._qdrant_path = qdrant_path
        self._qdrant_url = qdrant_url
        self._qdrant_port = qdrant_port
        self._qdrant = None
        self._collection_ready = False

    def _ensure_qdrant(self):
        """Lazily initialize Qdrant client and collection."""
        if self._qdrant is not None:
            return

        from qdrant_client import QdrantClient, models

        if self._qdrant_path:
            import os
            os.makedirs(self._qdrant_path, exist_ok=True)
            self._qdrant = QdrantClient(path=self._qdrant_path)
            print(f"📦 Qdrant using local disk at {self._qdrant_path}")
        elif self._qdrant_url:
            self._qdrant = QdrantClient(url=self._qdrant_url, port=self._qdrant_port)
            print(f"📦 Qdrant using remote server at {self._qdrant_url}")
        else:
            self._qdrant = QdrantClient(":memory:")
            print("📦 Qdrant using in-memory mode (no persistence)")

        # Create collection if it doesn't exist
        collections = [c.name for c in self._qdrant.get_collections().collections]
        if self.COLLECTION_NAME not in collections:
            self._qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=self.DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            print(f"✅ Created Qdrant collection: {self.COLLECTION_NAME}")
        else:
            print(f"📦 Qdrant collection '{self.COLLECTION_NAME}' already exists")

        self._collection_ready = True

    def _rebuild_bm25(self):
        """Rebuild the BM25 index from the full corpus."""
        from rank_bm25 import BM25Okapi

        if not self._bm25_corpus:
            self._bm25 = None
            return

        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25 = BM25Okapi(tokenized)

    def ingest(self, source: str, chunks: list[str]) -> int:
        """Index chunks into both Qdrant (dense) and BM25 (sparse).

        Args:
            source: Source identifier (filename or "web_search")
            chunks: Pre-processed text chunks from the chunker

        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0

        self._ensure_qdrant()
        from qdrant_client import models

        # Embed chunks with "passage: " prefix for multilingual-e5-large
        prefixed = [f"passage: {c}" for c in chunks]
        embeddings = self.embedder.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Batch insert into Qdrant
        start_id = self._chunk_count
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = start_id + i
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "source": source,
                        "chunk_idx": i,
                        "doc_idx": self._doc_count,
                    },
                )
            )

        # Upsert in batches of 100
        BATCH = 100
        for b in range(0, len(points), BATCH):
            self._qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points[b:b + BATCH],
            )

        # Update BM25 corpus
        self._bm25_corpus.extend(chunks)
        self._all_chunks.extend(chunks)
        self._all_sources.extend([source] * len(chunks))
        self._rebuild_bm25()

        self._doc_count += 1
        self._chunk_count += len(chunks)

        print(f"📥 Indexed {len(chunks)} chunks from '{source}' "
              f"(total: {self._chunk_count} chunks, {self._doc_count} docs)")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_top_n: int = 20,
        bm25_top_n: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Hybrid retrieve: dense + BM25 → RRF → rerank → top_k.

        Args:
            query: User question
            top_k: Number of final results to return
            dense_top_n: Number of candidates from dense search
            bm25_top_n: Number of candidates from BM25
            rrf_k: RRF constant (default 60)

        Returns:
            List of RetrievedChunk with scores from the cross-encoder reranker
        """
        if self._chunk_count == 0:
            return []

        # ── 1. Dense search (Qdrant) ────────────────────────────────────────
        dense_results = self._dense_search(query, top_n=dense_top_n)

        # ── 2. BM25 sparse search ──────────────────────────────────────────
        bm25_results = self._bm25_search(query, top_n=bm25_top_n)

        # ── 3. Reciprocal Rank Fusion ──────────────────────────────────────
        fused = self._rrf_fuse(dense_results, bm25_results, k=rrf_k)

        if not fused:
            return []

        # ── 4. Cross-encoder reranking ─────────────────────────────────────
        # Take top candidates for reranking (limit to save compute)
        rerank_candidates = fused[:min(len(fused), 20)]
        reranked = self._rerank(query, rerank_candidates)

        return reranked[:top_k]

    def _dense_search(self, query: str, top_n: int = 20) -> list[tuple[int, str, str, float]]:
        """Dense vector search in Qdrant.

        Returns list of (global_idx, text, source, score)
        """
        if not self._collection_ready:
            return []

        # Encode with "query: " prefix
        q_embedding = self.embedder.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        results = self._qdrant.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=q_embedding.tolist(),
            limit=top_n,
        )

        return [
            (hit.id, hit.payload["text"], hit.payload["source"], hit.score)
            for hit in results
        ]

    def _bm25_search(self, query: str, top_n: int = 20) -> list[tuple[int, str, str, float]]:
        """BM25 sparse search over the in-memory corpus.

        Returns list of (corpus_idx, text, source, score)
        """
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-N indices
        top_indices = np.argsort(scores)[::-1][:top_n]

        results = []
        for idx in top_indices:
            idx = int(idx)
            if scores[idx] > 0:
                results.append((
                    idx,
                    self._all_chunks[idx],
                    self._all_sources[idx],
                    float(scores[idx]),
                ))

        return results

    def _rrf_fuse(
        self,
        dense_results: list[tuple],
        bm25_results: list[tuple],
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion.

        Formula: RRF_score(d) = Σ 1 / (k + rank_i(d))
        where rank_i is the rank of document d in result list i.

        Args:
            dense_results: (id, text, source, score) from dense search
            bm25_results: (id, text, source, score) from BM25
            k: RRF constant (higher = more weight to lower-ranked items)

        Returns:
            Fused list of RetrievedChunk, sorted by RRF score descending
        """
        # Build text → metadata map and score accumulator
        rrf_scores: dict[str, float] = {}
        chunk_meta: dict[str, tuple[str, str]] = {}  # text → (source, method)

        # Score from dense results
        for rank, (_, text, source, _) in enumerate(dense_results):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank + 1)
            chunk_meta[text] = (source, "dense")

        # Score from BM25 results
        for rank, (_, text, source, _) in enumerate(bm25_results):
            rrf_scores[text] = rrf_scores.get(text, 0.0) + 1.0 / (k + rank + 1)
            if text in chunk_meta:
                chunk_meta[text] = (source, "hybrid")  # appeared in both
            else:
                chunk_meta[text] = (source, "bm25")

        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for text, score in sorted_items:
            source, method = chunk_meta[text]
            results.append(RetrievedChunk(
                text=text,
                score=score,
                source=source,
                retrieval_method=method,
            ))

        return results

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Rerank candidates using the cross-encoder model.

        Replaces RRF scores with cross-encoder relevance scores.
        """
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.reranker.predict(pairs)

        # Update scores and sort
        for i, candidate in enumerate(candidates):
            candidate.score = float(scores[i])

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def clear(self):
        """Clear all indexed data."""
        if self._qdrant is not None and self._collection_ready:
            from qdrant_client import models
            try:
                self._qdrant.delete_collection(self.COLLECTION_NAME)
                # Recreate empty collection
                self._qdrant.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=self.DENSE_VECTOR_SIZE,
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception as e:
                print(f"⚠️ Error clearing Qdrant: {e}")

        self._bm25 = None
        self._bm25_corpus = []
        self._all_chunks = []
        self._all_sources = []
        self._doc_count = 0
        self._chunk_count = 0
        print("🗑️ All retrieval indices cleared")

    @property
    def has_documents(self) -> bool:
        return self._chunk_count > 0

    @property
    def stats(self) -> dict:
        return {
            "documents": self._doc_count,
            "chunks": self._chunk_count,
            "bm25_terms": len(self._bm25_corpus),
        }

"""
KTGPT v2 — Semantic Query Cache (Redis-backed)
================================================
Caches query→response pairs with semantic similarity matching.

Instead of exact-match caching, this module embeds each query and
checks if a semantically similar query has been answered recently.
If cosine similarity > threshold → cache hit → return instantly.

Uses Redis for persistent storage across container restarts.
Cache entries expire via TTL to prevent stale responses.
"""

import json
import time
from typing import Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class CacheEntry:
    """A single cached query-response pair."""
    query: str
    query_embedding: list[float]
    response: str
    source: str
    model_used: str
    confidence: float
    timestamp: float


class SemanticCache:
    """Redis-backed semantic query cache with embedding similarity matching.

    Architecture:
    - Query embeddings stored as JSON arrays in Redis
    - On cache check: embed new query, compute cosine sim against all cached embeddings
    - If sim > threshold → cache hit
    - TTL-based expiration for freshness

    For high-volume systems, consider replacing the linear scan with
    a Redis Search vector index (RediSearch module).
    """

    CACHE_KEY_PREFIX = "ktgpt:cache:"
    CACHE_INDEX_KEY = "ktgpt:cache:index"

    def __init__(
        self,
        embedder,
        redis_client,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_entries: int = 1000,
    ):
        """
        Args:
            embedder: SentenceTransformer (multilingual-e5-large)
            redis_client: Redis client instance
            similarity_threshold: Min cosine similarity for cache hit
            ttl_seconds: Time-to-live for cache entries (default 1 hour)
            max_entries: Max cache entries (LRU eviction beyond this)
        """
        self.embedder = embedder
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries

    def get(self, query: str) -> Optional[dict]:
        """Check cache for a semantically similar query.

        Args:
            query: User's question

        Returns:
            Cached response dict if hit, None if miss.
            Response dict: {"response", "source", "model_used", "confidence"}
        """
        # Embed the query
        q_embedding = self.embedder.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        # Get all cached entry keys
        entry_keys = self._get_entry_keys()
        if not entry_keys:
            return None

        best_sim = -1.0
        best_entry = None

        for key in entry_keys:
            raw = self.redis.get(key)
            if raw is None:
                continue

            entry = json.loads(raw)
            cached_emb = np.array(entry["query_embedding"])

            # Cosine similarity (embeddings are already normalized)
            sim = float(np.dot(q_embedding, cached_emb))

            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.threshold and best_entry is not None:
            print(f"🎯 Cache HIT (similarity={best_sim:.4f})")
            return {
                "response": best_entry["response"],
                "source": best_entry["source"],
                "model_used": best_entry["model_used"],
                "confidence": best_entry["confidence"],
            }

        print(f"❌ Cache MISS (best similarity={best_sim:.4f})")
        return None

    def put(
        self,
        query: str,
        response: str,
        source: str = "",
        model_used: str = "",
        confidence: float = 0.0,
    ):
        """Store a query-response pair in the cache.

        Args:
            query: User's question
            response: Generated response
            source: Retrieved source info
            model_used: Which model generated the response
            confidence: Retrieval confidence score
        """
        # Embed the query
        q_embedding = self.embedder.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        entry = {
            "query": query,
            "query_embedding": q_embedding.tolist(),
            "response": response,
            "source": source,
            "model_used": model_used,
            "confidence": confidence,
            "timestamp": time.time(),
        }

        # Generate unique key
        entry_key = f"{self.CACHE_KEY_PREFIX}{int(time.time() * 1000)}"

        # Store with TTL
        self.redis.setex(
            entry_key,
            self.ttl,
            json.dumps(entry),
        )

        # Add key to index set
        self.redis.sadd(self.CACHE_INDEX_KEY, entry_key)

        # Evict if over max entries
        self._evict_if_needed()

        print(f"📝 Cached response for: '{query[:50]}...'")

    def clear(self):
        """Clear all cache entries."""
        entry_keys = self._get_entry_keys()
        if entry_keys:
            self.redis.delete(*entry_keys)
        self.redis.delete(self.CACHE_INDEX_KEY)
        print("🗑️ Cache cleared")

    @property
    def size(self) -> int:
        """Number of entries in cache."""
        return self.redis.scard(self.CACHE_INDEX_KEY) or 0

    def _get_entry_keys(self) -> list[str]:
        """Get all cached entry keys, cleaning up expired ones."""
        keys = self.redis.smembers(self.CACHE_INDEX_KEY)
        if not keys:
            return []

        # Decode bytes to strings if needed
        str_keys = []
        for k in keys:
            if isinstance(k, bytes):
                str_keys.append(k.decode("utf-8"))
            else:
                str_keys.append(k)

        # Clean up expired keys
        valid_keys = []
        expired = []
        for key in str_keys:
            if self.redis.exists(key):
                valid_keys.append(key)
            else:
                expired.append(key)

        if expired:
            self.redis.srem(self.CACHE_INDEX_KEY, *expired)

        return valid_keys

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds max size."""
        keys = self._get_entry_keys()
        if len(keys) <= self.max_entries:
            return

        # Sort by timestamp, evict oldest
        entries_with_ts = []
        for key in keys:
            raw = self.redis.get(key)
            if raw:
                entry = json.loads(raw)
                entries_with_ts.append((key, entry.get("timestamp", 0)))

        entries_with_ts.sort(key=lambda x: x[1])

        # Remove oldest until we're under the limit
        to_remove = len(entries_with_ts) - self.max_entries
        for i in range(to_remove):
            key = entries_with_ts[i][0]
            self.redis.delete(key)
            self.redis.srem(self.CACHE_INDEX_KEY, key)

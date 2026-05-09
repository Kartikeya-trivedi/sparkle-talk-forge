"""
KTGPT v2 — Cost-Aware Query Router
=====================================
Routes queries between two model tiers based on complexity heuristics:

- Small (Llama 3.1 8B Instruct): Simple, factual queries with high-confidence context
- Big (Gemma 4 26B-A4B-it): Complex, analytical queries or low-confidence scenarios

Also supports escalation: if the small model produces an unfaithful response,
the query is automatically re-routed to the big model.
"""

import re


class QueryRouter:
    """Cost-aware routing between small and big LLM tiers.

    Decision signals:
    ┌─────────────────────────────────────┬───────────┬───────────┐
    │ Signal                              │ → Small   │ → Big     │
    ├─────────────────────────────────────┼───────────┼───────────┤
    │ Query word count                    │ < 30      │ ≥ 30      │
    │ Retrieval confidence (reranker)     │ > 0.7     │ ≤ 0.7     │
    │ Complexity keywords present         │ No        │ Yes       │
    │ Number of context chunks used       │ ≤ 2       │ > 2       │
    │ Previous small model was unfaithful │ —         │ Escalate  │
    └─────────────────────────────────────┴───────────┴───────────┘
    """

    COMPLEX_KEYWORDS = {
        "compare", "contrast", "analyze", "analyse", "explain why",
        "what are the differences", "summarize", "summarise", "evaluate",
        "pros and cons", "implications", "trade-offs", "tradeoffs",
        "in detail", "elaborate", "critically", "comprehensive",
        "step by step", "walkthrough", "deep dive",
    }

    SIMPLE_PATTERNS = [
        r"^what is\b",
        r"^who is\b",
        r"^when did\b",
        r"^where is\b",
        r"^define\b",
        r"^how many\b",
        r"^is it true\b",
        r"^yes or no\b",
    ]

    # Model identifiers
    SMALL = "llama"   # Llama 3.1 8B Instruct
    BIG = "gemma"     # Gemma 4 26B-A4B-it

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_small_query_words: int = 30,
        max_small_chunks: int = 2,
    ):
        """
        Args:
            confidence_threshold: Below this, route to big model
            max_small_query_words: Queries longer than this go to big model
            max_small_chunks: If more chunks needed, route to big model
        """
        self.confidence_threshold = confidence_threshold
        self.max_small_query_words = max_small_query_words
        self.max_small_chunks = max_small_chunks

    def route(
        self,
        query: str,
        retrieval_score: float = 0.0,
        context_chunks: int = 0,
    ) -> str:
        """Determine which model tier to route the query to.

        Args:
            query: User's question
            retrieval_score: Best reranker score from retrieval
            context_chunks: Number of retrieved context chunks

        Returns:
            'llama' for small model, 'gemma' for big model
        """
        signals = {
            "query_length": len(query.split()),
            "retrieval_score": retrieval_score,
            "context_chunks": context_chunks,
            "has_complex_keywords": self._has_complex_keywords(query),
            "is_simple_pattern": self._is_simple_pattern(query),
        }

        # Decision logic: any complexity signal → big model
        reasons = []

        if signals["query_length"] > self.max_small_query_words:
            reasons.append(f"long query ({signals['query_length']} words)")

        if signals["has_complex_keywords"]:
            reasons.append("complex keywords detected")

        if retrieval_score > 0 and retrieval_score < self.confidence_threshold:
            reasons.append(f"low confidence ({retrieval_score:.3f})")

        if context_chunks > self.max_small_chunks:
            reasons.append(f"many chunks ({context_chunks})")

        if reasons:
            model = self.BIG
            print(f"🧠 Routing → Gemma 4 26B (reasons: {', '.join(reasons)})")
        else:
            model = self.SMALL
            reason = "simple pattern" if signals["is_simple_pattern"] else "default"
            print(f"⚡ Routing → Llama 3.1 8B ({reason})")

        return model

    def should_escalate(self, faithful: bool) -> bool:
        """Check if we should escalate from small to big model.

        Called after small model generation + faithfulness check.

        Args:
            faithful: Whether the small model's response was faithful

        Returns:
            True if we should re-generate with the big model
        """
        if not faithful:
            print("⬆️ Escalating to Gemma 4 26B due to unfaithful small model response")
            return True
        return False

    def _has_complex_keywords(self, query: str) -> bool:
        """Check if query contains complexity-indicating keywords."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in self.COMPLEX_KEYWORDS)

    def _is_simple_pattern(self, query: str) -> bool:
        """Check if query matches simple factual patterns."""
        query_lower = query.lower().strip()
        return any(re.match(p, query_lower) for p in self.SIMPLE_PATTERNS)

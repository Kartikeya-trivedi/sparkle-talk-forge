"""
KTGPT v2 — Semantic Chunking & Deduplication
==============================================
Production-grade document chunking pipeline:

1. Parse documents (PDF, TXT, MD)
2. Split into sentences
3. Semantic chunking: group consecutive sentences by embedding similarity
4. MinHash deduplication: remove near-duplicate chunks before indexing

Uses multilingual-e5-large for embedding (requires "passage: " prefix for docs).
"""

import re
import hashlib
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Document Parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_file(filename: str, content: bytes) -> str:
    """Extract plain text from .txt, .md, or .pdf files."""
    if filename.lower().endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    else:
        return content.decode("utf-8", errors="ignore")


# ─────────────────────────────────────────────────────────────────────────────
#  Sentence Splitting (Pre-processing before semantic chunking)
# ─────────────────────────────────────────────────────────────────────────────
def _split_into_sentences(text: str) -> list[str]:
    """Split raw text into individual sentences.

    Handles sentence-ending punctuation and newlines.
    Filters out very short fragments (< 3 words).
    """
    # Split on sentence boundaries and newlines
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if not s or len(s.split()) < 3:
            continue
        sentences.append(s)
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic Chunking
# ─────────────────────────────────────────────────────────────────────────────
def semantic_chunk(
    text: str,
    embedder,
    similarity_threshold: float = 0.5,
    max_chunk_tokens: int = 512,
    overlap_ratio: float = 0.1,
) -> list[str]:
    """Split text into semantically coherent chunks.

    Algorithm:
    1. Split text into sentences
    2. Embed each sentence with multilingual-e5-large
    3. Compute cosine similarity between consecutive sentence embeddings
    4. Split at points where similarity drops below threshold (breakpoints)
    5. If any chunk exceeds max_chunk_tokens, recursively split with overlap

    Args:
        text: Raw document text
        embedder: SentenceTransformer model (multilingual-e5-large)
        similarity_threshold: Cosine similarity threshold for breakpoints
        max_chunk_tokens: Maximum words per chunk (approximate token count)
        overlap_ratio: Overlap ratio for recursive splits (0.1 = 10%)

    Returns:
        List of semantically coherent text chunks
    """
    sentences = _split_into_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    # Embed all sentences with "passage: " prefix for multilingual-e5-large
    prefixed = [f"passage: {s}" for s in sentences]
    embeddings = embedder.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)

    # Compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        similarities.append(sim)

    # Find breakpoints where similarity drops below threshold
    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < similarity_threshold:
            breakpoints.append(i + 1)  # break AFTER sentence i

    # Build chunks from breakpoints
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp])
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        start = bp

    # Don't forget the last chunk
    if start < len(sentences):
        chunk_text = " ".join(sentences[start:])
        if chunk_text.strip():
            chunks.append(chunk_text.strip())

    # Recursive split for chunks that are too long
    final_chunks = []
    for chunk in chunks:
        words = chunk.split()
        if len(words) <= max_chunk_tokens:
            final_chunks.append(chunk)
        else:
            # Split into smaller pieces with overlap
            overlap_words = max(1, int(max_chunk_tokens * overlap_ratio))
            step = max_chunk_tokens - overlap_words
            for i in range(0, len(words), step):
                sub = " ".join(words[i:i + max_chunk_tokens])
                if len(sub.split()) >= 3:
                    final_chunks.append(sub)

    return final_chunks


# ─────────────────────────────────────────────────────────────────────────────
#  MinHash Deduplication
# ─────────────────────────────────────────────────────────────────────────────
def minhash_dedup(chunks: list[str], threshold: float = 0.8) -> tuple[list[str], int]:
    """Remove near-duplicate chunks using MinHash signatures.

    Uses the datasketch library for efficient Jaccard similarity estimation.

    Args:
        chunks: List of text chunks
        threshold: Jaccard similarity threshold above which chunks are considered duplicates

    Returns:
        Tuple of (deduplicated chunks, number of duplicates removed)
    """
    if len(chunks) <= 1:
        return chunks, 0

    from datasketch import MinHash, MinHashLSH

    # Build MinHash for each chunk
    def _make_minhash(text: str, num_perm: int = 128) -> MinHash:
        m = MinHash(num_perm=num_perm)
        # Use word-level 3-grams as shingles
        words = text.lower().split()
        for i in range(len(words) - 2):
            shingle = " ".join(words[i:i + 3])
            m.update(shingle.encode("utf-8"))
        return m

    # Build LSH index
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = []
    for i, chunk in enumerate(chunks):
        mh = _make_minhash(chunk)
        minhashes.append(mh)
        try:
            lsh.insert(f"chunk_{i}", mh)
        except ValueError:
            # Duplicate detected by LSH — skip
            pass

    # Find unique chunks (keep the first occurrence of each group)
    seen = set()
    unique_chunks = []
    for i, chunk in enumerate(chunks):
        if i in seen:
            continue
        unique_chunks.append(chunk)
        # Mark all similar chunks as seen
        neighbors = lsh.query(minhashes[i])
        for n in neighbors:
            idx = int(n.split("_")[1])
            if idx != i:
                seen.add(idx)

    removed = len(chunks) - len(unique_chunks)
    return unique_chunks, removed


# ─────────────────────────────────────────────────────────────────────────────
#  Full Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def parse_and_chunk(
    filename: str,
    content: bytes,
    embedder,
    similarity_threshold: float = 0.5,
    dedup_threshold: float = 0.8,
) -> tuple[list[str], int]:
    """Full document processing pipeline: parse → semantic chunk → dedup.

    Args:
        filename: Document filename (used to determine parser)
        content: Raw file bytes
        embedder: SentenceTransformer model
        similarity_threshold: Semantic chunking breakpoint threshold
        dedup_threshold: MinHash dedup Jaccard threshold

    Returns:
        Tuple of (final chunks, number of duplicates removed)
    """
    # 1. Parse document
    text = parse_file(filename, content)
    if not text.strip():
        return [], 0

    # 2. Semantic chunking
    chunks = semantic_chunk(text, embedder, similarity_threshold=similarity_threshold)
    if not chunks:
        return [], 0

    # 3. MinHash deduplication
    deduped, removed = minhash_dedup(chunks, threshold=dedup_threshold)

    print(f"📄 {filename}: {len(_split_into_sentences(text))} sentences → "
          f"{len(chunks)} chunks → {len(deduped)} after dedup ({removed} removed)")

    return deduped, removed

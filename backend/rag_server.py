"""
KTGPT v2 — Production RAG Server (Modal Serverless)
======================================================
Main orchestrator that wires together all RAG components:

  Semantic Cache → Hybrid Retrieval → Confidence Gate →
  Cost-Aware Router → vLLM Inference → NLI Faithfulness →
  Cache & Return

Endpoints:
  POST /          Chat with full RAG pipeline
  POST /upload    Upload and index a document
  GET  /stats     Retrieval index statistics
  POST /clear     Clear all indexed documents + cache
  GET  /health    System health check

Runs entirely serverless on Modal with:
  - RAG Orchestrator on T4 GPU (embeddings + reranking)
  - Llama 3.1 8B on A10G (small model tier)
  - Gemma 4 26B on A100 (big model tier)
  - Qdrant in local disk mode on shared Volume
"""

import modal

# ─────────────────────────────────────────────────────────────────────────────
#  Modal App & Shared Resources
# ─────────────────────────────────────────────────────────────────────────────
app = modal.App("ktgpt-rag-server")

vol = modal.Volume.from_name("ktgpt-rag-models", create_if_missing=True)
MOUNT = "/models"

# ── Images ───────────────────────────────────────────────────────────────────
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0",
        "transformers>=4.44.0",
        "torch>=2.1.0",
    )
)

rag_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
    )
    .pip_install(
        "fastapi[standard]",
        "pydantic>=2.0.0",
        "python-multipart",
        "sentence-transformers>=2.2.0",
        "pymupdf>=1.24.0",
        "rank_bm25",
        "datasketch",
        "numpy",
        "redis",
        "qdrant-client>=1.7.0",
        "requests",
    )
    .add_local_dir(
        "backend",
        remote_path="/app/backend",
        ignore=["**/__pycache__/**", "server.py", "download_weights.py"],
    )
)


# ─────────────────────────────────────────────────────────────────────────────
#  Small Model: Llama 3.1 8B Instruct (vLLM on A10G)
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="A10G",
    image=vllm_image,
    volumes={MOUNT: vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
class SmallModel:
    """Llama 3.1 8B Instruct — fast, cost-efficient for simple queries."""

    MODEL_DIR = f"{MOUNT}/llama-3.1-8b-instruct"
    MODEL_NAME = "llama-3.1-8b"

    @modal.enter()
    def start(self):
        from vllm import LLM, SamplingParams

        print(f"🚀 Loading Llama 3.1 8B from {self.MODEL_DIR}...")
        self.llm = LLM(
            model=self.MODEL_DIR,
            dtype="bfloat16",
            max_model_len=4096,
            gpu_memory_utilization=0.90,
        )
        self.default_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=512,
            repetition_penalty=1.1,
        )
        print("✅ Llama 3.1 8B ready")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=max_tokens,
            repetition_penalty=1.1,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  Big Model: Gemma 4 26B-A4B-it (vLLM on A100)
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="A100",
    image=vllm_image,
    volumes={MOUNT: vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=600,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=10)
class BigModel:
    """Gemma 4 26B-A4B-it — powerful MoE for complex analytical queries."""

    MODEL_DIR = f"{MOUNT}/gemma-4-26b-a4b-it"
    MODEL_NAME = "gemma-4-26b"

    @modal.enter()
    def start(self):
        from vllm import LLM, SamplingParams

        print(f"🚀 Loading Gemma 4 26B from {self.MODEL_DIR}...")
        self.llm = LLM(
            model=self.MODEL_DIR,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.90,
        )
        self.default_params = SamplingParams(
            temperature=0.4,
            top_p=0.95,
            max_tokens=1024,
            repetition_penalty=1.05,
        )
        print("✅ Gemma 4 26B ready")

    @modal.method()
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.4,
            top_p=0.95,
            max_tokens=max_tokens,
            repetition_penalty=1.05,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()


# ─────────────────────────────────────────────────────────────────────────────
#  RAG Orchestrator (FastAPI on T4)
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="T4",
    image=rag_image,
    volumes={MOUNT: vol},
    secrets=[
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("serpapi"),
        modal.Secret.from_name("redis-secret"),   # REDIS_URL
    ],
    timeout=7200,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=20)
class RAGServer:
    """Main RAG orchestrator — wires all components together.

    Pipeline:
    1. Check semantic cache (Redis)
    2. Hybrid retrieve (Qdrant local disk + BM25 → RRF → Rerank)
    3. Confidence gate (refuse if score < threshold)
    4. Route query (small vs big model)
    5. Generate response (vLLM)
    6. Faithfulness check (NLI)
    7. Escalate if needed (small → big)
    8. Cache response
    9. Return with metadata
    """

    @modal.enter()
    def load(self):
        import sys
        import os
        import redis as redis_lib
        # Add backend dir so all new modules resolve without package prefix
        sys.path.insert(0, "/app/backend")

        from sentence_transformers import SentenceTransformer, CrossEncoder

        # ── Load embedding model ────────────────────────────────────────────
        embedder_path = f"{MOUNT}/multilingual-e5-large"
        print(f"Loading embedder from {embedder_path}...")
        self.embedder = SentenceTransformer(embedder_path)

        # ── Load reranker ───────────────────────────────────────────────────
        reranker_path = f"{MOUNT}/ms-marco-MiniLM-L-6-v2"
        print(f"Loading reranker from {reranker_path}...")
        self.reranker = CrossEncoder(reranker_path)

        # ── Load NLI model ──────────────────────────────────────────────────
        nli_path = f"{MOUNT}/nli-deberta-v3-base"
        print(f"Loading NLI model from {nli_path}...")
        self.nli_model = CrossEncoder(nli_path)

        # ── Initialize components ───────────────────────────────────────────
        from retriever import HybridRetriever
        from hallucination import HallucinationGuard
        from router import QueryRouter
        from cache import SemanticCache

        # Qdrant — local disk mode on the shared Volume
        qdrant_path = f"{MOUNT}/qdrant_data"
        print(f"Initializing Qdrant at {qdrant_path}...")
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            reranker=self.reranker,
            qdrant_path=qdrant_path,
        )

        # Hallucination guard
        self.guard = HallucinationGuard(nli_model=self.nli_model)

        # Query router
        self.router = QueryRouter()

        # Redis cache
        redis_url = os.environ.get("REDIS_URL", "")
        if redis_url:
            print("Connecting to Redis...")
            redis_client = redis_lib.from_url(redis_url, decode_responses=True)
            self.cache = SemanticCache(
                embedder=self.embedder,
                redis_client=redis_client,
            )
            print("✅ Redis cache connected")
        else:
            print("⚠️ REDIS_URL not set — cache disabled")
            self.cache = None

        # Model references (Modal class instances)
        self.small_model = SmallModel()
        self.big_model = BigModel()

        print("🎉 RAG Server fully initialized")

    @modal.asgi_app()
    def serve(self):
        import re
        from fastapi import FastAPI, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware
        from models import (
            ChatRequest, ChatResponse, UploadResponse,
            StatsResponse, HealthResponse,
        )
        from chunker import parse_and_chunk
        from prompts import build_prompt
        from hallucination import HallucinationGuard

        web = FastAPI(title="KTGPT v2 RAG Server")
        web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server = self  # capture for closures

        # ── POST / — Main Chat Endpoint ─────────────────────────────────────
        @web.post("/")
        def chat(req: ChatRequest):
            import os
            context = req.context
            source = ""

            # ── Step 1: Check semantic cache ────────────────────────────────
            if server.cache is not None:
                cached = server.cache.get(req.question)
                if cached is not None:
                    return ChatResponse(
                        response=cached["response"],
                        source=cached["source"],
                        model_used=cached["model_used"],
                        confidence=cached["confidence"],
                        faithful=True,
                        cached=True,
                    )

            # ── Step 2: Web search (if enabled) ─────────────────────────────
            if req.use_web_search and not context.strip():
                try:
                    import requests as http_req
                    api_key = os.environ.get("SERPAPI_KEY", "")
                    if api_key:
                        print(f"🌐 SerpAPI searching: '{req.question}'")
                        resp = http_req.get(
                            "https://serpapi.com/search.json",
                            params={"q": req.question, "api_key": api_key,
                                    "engine": "google", "num": 5},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        results = resp.json().get("organic_results", [])
                        if results:
                            search_text = "\n".join(
                                f"{r.get('title', '')}. {r.get('snippet', '')}"
                                for r in results
                            )
                            # Clean noisy date patterns
                            search_text = re.sub(
                                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                                r'\s+\d{1,2},\s+\d{4}\s*\.\.\.\s*', '', search_text
                            )
                            search_text = re.sub(r'\.{2,}', '.', search_text)

                            from chunker import semantic_chunk, minhash_dedup
                            chunks = semantic_chunk(search_text, server.embedder)
                            chunks, _ = minhash_dedup(chunks)
                            if chunks:
                                server.retriever.ingest("web_search", chunks)
                except Exception as e:
                    print(f"⚠️ Web search failed: {e}")

            # ── Step 3: Hybrid retrieval ────────────────────────────────────
            retrieved = []
            if not context.strip() and req.use_retrieval and server.retriever.has_documents:
                retrieved = server.retriever.retrieve(req.question, top_k=5)
                if retrieved:
                    context = "\n\n".join(chunk.text for chunk in retrieved)
                    source = ", ".join(set(c.source for c in retrieved if c.source))

            retrieval_score = max((c.score for c in retrieved), default=0.0)

            # ── Step 4: Confidence gate ─────────────────────────────────────
            if retrieved and not server.guard.gate_retrieval(
                [c.score for c in retrieved]
            ):
                refusal = HallucinationGuard.refusal_response()
                return ChatResponse(
                    response=refusal,
                    source="",
                    model_used="none",
                    confidence=retrieval_score,
                    faithful=True,
                    cached=False,
                )

            # ── Step 5: Route to model ──────────────────────────────────────
            model_name = server.router.route(
                query=req.question,
                retrieval_score=retrieval_score,
                context_chunks=len(retrieved),
            )

            # ── Step 6: Build prompt & generate ─────────────────────────────
            prompt = build_prompt(model_name, context, req.question)

            if model_name == "llama":
                response_text = server.small_model.generate.remote(prompt)
            else:
                response_text = server.big_model.generate.remote(prompt)

            # ── Step 7: Faithfulness check ──────────────────────────────────
            faithful = True
            if context.strip() and response_text:
                result = server.guard.check_faithfulness(context, response_text)
                faithful = result.faithful

                # Escalate if small model was unfaithful
                if not faithful and model_name == "llama" and server.router.should_escalate(faithful):
                    print("⬆️ Escalating: regenerating with Gemma 4 26B...")
                    model_name = "gemma"
                    prompt = build_prompt("gemma", context, req.question)
                    response_text = server.big_model.generate.remote(prompt)

                    # Re-check faithfulness
                    result = server.guard.check_faithfulness(context, response_text)
                    faithful = result.faithful

            # ── Step 8: Cache response ──────────────────────────────────────
            if server.cache is not None and response_text:
                server.cache.put(
                    query=req.question,
                    response=response_text,
                    source=source,
                    model_used=model_name,
                    confidence=retrieval_score,
                )

            # ── Step 9: Return with metadata ────────────────────────────────
            model_display = {
                "llama": "llama-3.1-8b",
                "gemma": "gemma-4-26b",
            }.get(model_name, model_name)

            return ChatResponse(
                response=response_text or "No response generated.",
                source=source,
                model_used=model_display,
                confidence=round(retrieval_score, 4),
                faithful=faithful,
                cached=False,
            )

        # ── POST /upload — Document Upload & Indexing ───────────────────────
        @web.post("/upload")
        async def upload(file: UploadFile = File(...)):
            content = await file.read()
            filename = file.filename or "unknown.txt"

            chunks, dedup_removed = parse_and_chunk(
                filename, content, server.embedder
            )

            if chunks:
                server.retriever.ingest(filename, chunks)

            return UploadResponse(
                filename=filename,
                chunks=len(chunks),
                status="indexed",
                dedup_removed=dedup_removed,
            )

        # ── GET /stats — Index Statistics ───────────────────────────────────
        @web.get("/stats")
        def stats():
            r_stats = server.retriever.stats
            return StatsResponse(
                documents=r_stats["documents"],
                chunks=r_stats["chunks"],
                bm25_terms=r_stats["bm25_terms"],
                cache_entries=server.cache.size if server.cache else 0,
            )

        # ── POST /clear — Clear Everything ──────────────────────────────────
        @web.post("/clear")
        def clear():
            server.retriever.clear()
            if server.cache:
                server.cache.clear()
            return {"status": "cleared"}

        # ── GET /health — Health Check ──────────────────────────────────────
        @web.get("/health")
        def health():
            models = ["multilingual-e5-large", "ms-marco-MiniLM-L-6-v2",
                       "nli-deberta-v3-base"]
            return HealthResponse(
                status="ok",
                models_loaded=models,
                qdrant_connected=server.retriever._collection_ready,
                redis_connected=server.cache is not None,
            )

        return web

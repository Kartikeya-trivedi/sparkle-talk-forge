"""
KTGPT Inference Server + RAG Retrieval Pipeline
================================================
Exposes web endpoints for:
  - /        POST  Chat with automatic context retrieval
  - /upload  POST  Upload and index a document
  - /search  POST  Web search → index → retrieve
  - /stats   GET   Retrieval index stats
  - /clear   POST  Clear all uploaded documents

Uses @modal.asgi_app() for multi-route FastAPI serving.
"""

import re
import modal
from pydantic import BaseModel

app = modal.App("ktgpt-server")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "fastapi[standard]",
        "pydantic",
        "python-multipart",
        # RAG retrieval
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "pymupdf>=1.24.0",
        gpu="T4",
    )
    .add_local_dir(
        ".",
        remote_path="/ktgpt-source",
        ignore=["**/__pycache__/**", "**/.venv/**", "**/.git/**", "**/*.pt", "**/*.jsonl",
                "**/checkpoints/**", "**/scratch/**", "**/ktgpt_chat/**", "**/.ipynb_checkpoints/**",
                "**/frontend/**", "**/node_modules/**"]
    )
    .add_local_dir(r"C:\Projects\Large_Language_model\KTGPT\model", remote_path="/ktgpt/model")
)

vol = modal.Volume.from_name("ktgpt-finetune-vol", create_if_missing=True)
VOL_MOUNT = "/vol"

# ─────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT — exactly as in eval.py
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are KTGPT, a helpful assistant made by Mindrix.

If context is provided, answer with the help of information from that context.

Keep responses concise and natural. Avoid mentioning the word 'context' or any meta-explanations."""

MAX_CONTEXT_WORDS = 12


# ─────────────────────────────────────────────────────────────────────────────
#  DOCUMENT PARSING & SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def parse_file(filename: str, content: bytes) -> str:
    """Extract plain text from .txt, .md, or .pdf files."""
    if filename.lower().endswith(".pdf"):
        import pymupdf
        doc = pymupdf.open(stream=content, filetype="pdf")
        return "\n".join(page.get_text() for page in doc)
    else:
        return content.decode("utf-8", errors="ignore")


def split_sentences(text: str) -> list[str]:
    """Split text into individual sentences, each ~10-15 words.

    Produces chunks matching KTGPT Phase 1.5's short-context
    extraction training format.
    """
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if not s or len(s.split()) < 3:
            continue
        if len(s.split()) > 15:
            words = s.split()
            for i in range(0, len(words), MAX_CONTEXT_WORDS):
                chunk = " ".join(words[i:i + MAX_CONTEXT_WORDS])
                if len(chunk.split()) >= 3:
                    sentences.append(chunk)
        else:
            sentences.append(s)
    return sentences


def trim_to_max_words(text: str, max_words: int = MAX_CONTEXT_WORDS) -> str:
    """Trim text to at most max_words, cutting at a sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        result = text
    else:
        trimmed = " ".join(words[:max_words])
        # Try to cut at the last sentence boundary
        result = trimmed
        for sep in ['. ', '! ', '? ']:
            idx = trimmed.rfind(sep)
            if idx > len(trimmed) // 2:
                result = trimmed[:idx + 1].strip()
                break
    # Ensure it ends with punctuation so the model sees a complete sentence
    if result and result[-1] not in '.!?':
        result += '.'
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVER CLASS (in-memory ChromaDB, per-session)
# ─────────────────────────────────────────────────────────────────────────────
class Retriever:
    """Sentence-level vector retrieval with reranking.

    Stores document sentences in an in-memory ChromaDB collection.
    On query: embed -> vector search top 10 -> rerank -> return best
    sentence trimmed to ~12 words.
    """

    def __init__(self, embedder, reranker):
        self.embedder = embedder
        self.reranker = reranker
        self._client = None
        self._collection = None
        self._doc_count = 0
        self._sentence_count = 0

    def _ensure_collection(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.Client()
            self._collection = self._client.create_collection(
                name="ktgpt_docs",
                metadata={"hnsw:space": "cosine"},
            )

    def ingest(self, filename: str, raw_text: str) -> int:
        """Split text into sentences, embed, and store. Returns sentence count."""
        self._ensure_collection()
        sentences = split_sentences(raw_text)
        if not sentences:
            return 0
        embeddings = self.embedder.encode(sentences, normalize_embeddings=True).tolist()
        ids = [f"doc{self._doc_count}_s{i}" for i in range(len(sentences))]
        metadatas = [{"filename": filename, "sentence_idx": i} for i in range(len(sentences))]
        # Batch inserts to stay under ChromaDB's max batch size (5461)
        BATCH = 5000
        for start in range(0, len(sentences), BATCH):
            end = start + BATCH
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=sentences[start:end],
                metadatas=metadatas[start:end],
            )
        self._doc_count += 1
        self._sentence_count += len(sentences)
        return len(sentences)

    def retrieve(self, question: str, top_k: int = 1) -> str:
        """Embed question -> vector search -> rerank -> return best sentence."""
        if self._collection is None or self._sentence_count == 0:
            return ""
        q_emb = self.embedder.encode([question], normalize_embeddings=True).tolist()
        n_results = min(10, self._sentence_count)
        results = self._collection.query(query_embeddings=q_emb, n_results=n_results)
        candidates = results["documents"][0]
        if not candidates:
            return ""
        pairs = [(question, doc) for doc in candidates]
        scores = self.reranker.predict(pairs)
        best_idx = int(scores.argmax())
        best_sentence = candidates[best_idx]
        return trim_to_max_words(best_sentence)

    def clear(self):
        """Clear all stored documents."""
        if self._client is not None:
            try:
                self._client.delete_collection("ktgpt_docs")
            except Exception:
                pass
            self._client = None
            self._collection = None
        self._doc_count = 0
        self._sentence_count = 0

    @property
    def has_documents(self) -> bool:
        return self._sentence_count > 0

    @property
    def stats(self) -> dict:
        return {"documents": self._doc_count, "sentences": self._sentence_count}


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(context: str, question: str) -> str:
    """Build the inference prompt using the exact same format as training/eval."""
    if context.strip():
        user_content = f"Context: {context}\n\nQuestion: {question}"
    else:
        user_content = question

    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n"
        f"<|user|>\n{user_content}\n<|end|>\n"
        f"<|assistant|>\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  API MODELS
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    context: str = ""
    question: str
    use_retrieval: bool = True
    use_web_search: bool = False

class ChatResponse(BaseModel):
    response: str
    source: str = ""

class UploadResponse(BaseModel):
    filename: str
    sentences: int
    status: str


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE SERVER CLASS
# ─────────────────────────────────────────────────────────────────────────────
@app.cls(
    gpu="T4",
    timeout=7200,
    scaledown_window=1200,
    image=image,
    secrets=[modal.Secret.from_name("hf-secret"), modal.Secret.from_name("serpapi")],
    volumes={VOL_MOUNT: vol},
)
class KTGPTServer:
    @modal.enter()
    def load_model(self):
        import torch
        import sys
        import inspect
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer, CrossEncoder

        sys.path.insert(0, "/ktgpt-source")
        sys.path.insert(0, "/ktgpt")
        from model.model import KTGPT
        from model.config import KTGPTConfig

        self.device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name()}")

        # ── Load KTGPT model ────────────────────────────────────────────────
        model_dir = "ktgpt-phase15"
        merged_name = "KTGPT-Grounding-Merged.pt"

        tok_path = f"{VOL_MOUNT}/{model_dir}/tokenizer"
        print(f"Loading tokenizer from {tok_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)

        model_path = f"{VOL_MOUNT}/{model_dir}/{merged_name}"
        print(f"Loading merged KTGPT model from {model_path}...")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        config_data = checkpoint.get("config", checkpoint.get("model_args"))
        if isinstance(config_data, dict):
            sig = inspect.signature(KTGPTConfig.__init__)
            valid_keys = sig.parameters.keys()
            filtered_config = {k: v for k, v in config_data.items() if k in valid_keys}
            config = KTGPTConfig(**filtered_config)
        else:
            config = config_data

        config.vocab_size = len(self.tokenizer)

        self.model = KTGPT(config)
        clean_state = {}
        for k, v in checkpoint["model"].items():
            if ".base.weight" in k:
                clean_state[k.replace(".base.weight", ".weight")] = v
            else:
                clean_state[k] = v
        self.model.load_state_dict(clean_state, strict=False)

        if "router_biases" in checkpoint:
            for i, layer in enumerate(self.model.layers):
                key = f"layer_{i}"
                if key in checkpoint["router_biases"]:
                    layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])

        self.model.to(device=self.device, dtype=torch.bfloat16)
        self.model.eval()

        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
        print("KTGPT Model loaded successfully.")

        # ── Load RAG retrieval models ────────────────────────────────────────
        print("Loading embedding model (bge-small-en-v1.5)...")
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

        print("Loading reranker (ms-marco-MiniLM-L-6-v2)...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.retriever = Retriever(embedder, reranker)
        print("RAG retrieval pipeline ready.")

    @modal.asgi_app()
    def serve(self):
        from fastapi import FastAPI, UploadFile, File
        from fastapi.middleware.cors import CORSMiddleware

        web = FastAPI(title="KTGPT Server")
        web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        server_self = self  # capture for closures

        @web.post("/")
        def chat(req: ChatRequest):
            import torch

            context = req.context
            source = ""

            # Web search: SerpAPI → index results → retrieve best chunk
            if req.use_web_search and not context.strip():
                try:
                    import os
                    import requests as http_req

                    api_key = os.environ.get("SERPAPI_KEY", "")
                    if not api_key:
                        print("SERPAPI_KEY not set — skipping web search")
                    else:
                        print(f"SerpAPI searching: '{req.question}'")
                        resp = http_req.get(
                            "https://serpapi.com/search.json",
                            params={"q": req.question, "api_key": api_key, "engine": "google", "num": 5},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        data = resp.json()
                        results = data.get("organic_results", [])
                        print(f"Search returned {len(results)} results")

                        if results:
                            for i, r in enumerate(results):
                                print(f"  Result {i}: {r.get('title', '')[:60]}")
                            search_text = "\n".join(
                                f"{r.get('title', '')}. {r.get('snippet', '')}" for r in results
                            )
                            # Clean SerpAPI garbage that breaks sentence splitting
                            search_text = re.sub(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\s*\.\.\.\s*', '', search_text)
                            search_text = re.sub(r'\.{2,}', '.', search_text)
                            
                            n = server_self.retriever.ingest("web_search", search_text)
                            print(f"Indexed {n} sentences from search results")
                            context = server_self.retriever.retrieve(req.question)
                            print(f"Retrieved context: '{context}'")
                            if context:
                                source = f"🌐 {context}"
                        else:
                            print("No web results found")
                except Exception as e:
                    import traceback
                    print(f"Web search failed: {e}")
                    traceback.print_exc()

            # Standard file retrieval (if no web search context found)
            elif not context.strip() and req.use_retrieval and server_self.retriever.has_documents:
                context = server_self.retriever.retrieve(req.question)
                if context:
                    source = context
                    print(f"Retrieved context: '{context}'")

            prompt = build_prompt(context, req.question)
            print(f"Final prompt:\n{prompt}")
            input_ids = server_self.tokenizer.encode(prompt, return_tensors="pt").to(server_self.device)

            with torch.no_grad():
                output_ids = server_self.model.generate(
                    input_ids,
                    max_new_tokens=200,
                    temperature=0.1,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    eos_token_id=server_self.end_token_id,
                )

            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = server_self.tokenizer.decode(new_tokens, skip_special_tokens=False)
            if "<|end|>" in response:
                response = response.split("<|end|>")[0].strip()

            return ChatResponse(response=response, source=source)

        @web.post("/upload")
        async def upload(file: UploadFile = File(...)):
            content = await file.read()
            filename = file.filename or "unknown.txt"
            text = parse_file(filename, content)
            n_sentences = server_self.retriever.ingest(filename, text)
            print(f"Indexed '{filename}': {n_sentences} sentences")
            return UploadResponse(filename=filename, sentences=n_sentences, status="indexed")

        @web.get("/stats")
        def stats():
            return server_self.retriever.stats

        @web.post("/clear")
        def clear():
            server_self.retriever.clear()
            return {"status": "cleared"}

        return web

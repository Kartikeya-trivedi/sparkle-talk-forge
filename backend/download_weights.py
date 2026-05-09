"""
KTGPT v2 — One-Time Model Weight Downloader
=============================================
Downloads all model weights to a shared Modal Volume so that
inference containers can load instantly without re-downloading.

Usage:
    modal run backend/download_weights.py

Volume layout after download:
    /models/
    ├── gemma-4-26b-a4b-it/           # Gemma 4 26B MoE (instruction-tuned)
    ├── llama-3.1-8b-instruct/        # Llama 3.1 8B Instruct
    ├── multilingual-e5-large/        # Embedding model
    ├── ms-marco-MiniLM-L-6-v2/      # Cross-encoder reranker
    └── nli-deberta-v3-base/          # NLI faithfulness checker
"""

import modal

app = modal.App("ktgpt-weight-downloader")

vol = modal.Volume.from_name("ktgpt-rag-models", create_if_missing=True)
MOUNT = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "transformers>=4.44.0",
        "torch>=2.1.0",
        "sentence-transformers>=2.2.0",
        "protobuf>=3.20.0",
        "sentencepiece>=0.1.99",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ─────────────────────────────────────────────────────────────────────────────
#  Models to download
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {
    # LLM inference models (loaded by vLLM)
    "google/gemma-4-26B-A4B-it": "gemma-4-26b-a4b-it",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-instruct",
    # Embedding model (multilingual, used for semantic chunking + retrieval)
    "intfloat/multilingual-e5-large": "multilingual-e5-large",
    # Cross-encoder reranker (used after hybrid retrieval)
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-6-v2",
    # NLI model for faithfulness verification
    "cross-encoder/nli-deberta-v3-base": "nli-deberta-v3-base",
}


@app.function(
    image=image,
    volumes={MOUNT: vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=7200,   # 2 hours — Gemma 4 26B is ~50GB
    memory=32768,   # 32GB RAM for large downloads
)
def download_all():
    """Download all model weights to the shared Modal Volume."""
    from huggingface_hub import snapshot_download
    import os

    for repo_id, dirname in MODELS.items():
        local_dir = os.path.join(MOUNT, dirname)

        # Skip if already downloaded
        if os.path.exists(local_dir) and any(
            f.endswith((".safetensors", ".bin", ".json"))
            for f in os.listdir(local_dir)
        ):
            print(f"⏭️  {repo_id} already exists at {local_dir}, skipping")
            continue

        print(f"⬇️  Downloading {repo_id} → {local_dir}")
        snapshot_download(
            repo_id,
            local_dir=local_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        print(f"✅ {repo_id} downloaded successfully")

    vol.commit()
    print("\n🎉 All models downloaded and committed to volume 'ktgpt-rag-models'")
    print(f"Volume contents:")
    for entry in os.listdir(MOUNT):
        full = os.path.join(MOUNT, entry)
        if os.path.isdir(full):
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, _, fns in os.walk(full)
                for f in fns
            ) / (1024 * 1024)
            print(f"  📁 {entry}: {size_mb:.0f} MB")


@app.local_entrypoint()
def main():
    """Entry point: `modal run backend/download_weights.py`"""
    print("🚀 Starting model weight download to Modal Volume...")
    print(f"   Volume: ktgpt-rag-models")
    print(f"   Models: {len(MODELS)}")
    print()
    download_all.remote()
    print("\n✅ Done! All weights are ready in the 'ktgpt-rag-models' volume.")

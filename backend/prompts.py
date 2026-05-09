"""
KTGPT v2 — Prompt Templates
=============================
Chat prompt formatters for Llama 3.1 Instruct and Gemma 4 Instruct.
Each model uses its own special-token schema for optimal instruction following.
"""

# ─────────────────────────────────────────────────────────────────────────────
#  System Prompt (shared across models)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are KTGPT, an advanced AI assistant made by Mindrix.

If context is provided, answer using ONLY the information from that context.
If the context does not contain enough information to answer, say so honestly.

Keep responses concise, accurate, and natural. Never fabricate information.
Do not mention the word 'context' or any meta-explanations about your process."""


# ─────────────────────────────────────────────────────────────────────────────
#  Llama 3.1 Instruct Format
# ─────────────────────────────────────────────────────────────────────────────
def build_llama_prompt(context: str, question: str) -> str:
    """Build a prompt in Llama 3.1 Instruct chat format.

    Format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {user}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
    """
    if context.strip():
        user_content = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user_content = question

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Gemma 4 Instruct Format
# ─────────────────────────────────────────────────────────────────────────────
def build_gemma_prompt(context: str, question: str) -> str:
    """Build a prompt in Gemma 4 chat format.

    Format:
        <start_of_turn>user
        {system + user}<end_of_turn>
        <start_of_turn>model
    """
    if context.strip():
        user_content = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
    else:
        user_content = f"{SYSTEM_PROMPT}\n\n{question}"

    return (
        f"<start_of_turn>user\n"
        f"{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Prompt Router
# ─────────────────────────────────────────────────────────────────────────────
def build_prompt(model_name: str, context: str, question: str) -> str:
    """Build a prompt for the specified model.

    Args:
        model_name: 'llama' or 'gemma'
        context: Retrieved context (may be empty)
        question: User's question
    """
    if model_name == "llama":
        return build_llama_prompt(context, question)
    elif model_name == "gemma":
        return build_gemma_prompt(context, question)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'llama' or 'gemma'.")

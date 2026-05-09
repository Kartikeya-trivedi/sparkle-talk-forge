"""
KTGPT v2 — Hallucination Control
==================================
Three-layer defense against hallucinated responses:

1. Confidence Gating: Refuse to answer if retrieval confidence is too low
2. NLI Faithfulness Check: Verify the response is entailed by the context
3. Source Attribution: Return retrieved chunks for transparency
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaithfulnessResult:
    """Result of NLI faithfulness verification."""
    faithful: bool
    nli_label: str       # "entailment", "neutral", or "contradiction"
    nli_score: float     # confidence score for the label
    should_refuse: bool  # True if we should not return this response


class HallucinationGuard:
    """Multi-layer hallucination defense system.

    Layer 1 — Confidence Gating:
        Before generation, check if the retrieval reranker score is high enough.
        If top-k similarity < threshold → refuse to answer.

    Layer 2 — NLI Faithfulness:
        After generation, use an NLI model to check if the response is
        entailed by the retrieved context. Flags contradictions.

    Layer 3 — Source Attribution:
        Always return the retrieved chunks so the user can verify.
    """

    # NLI label mapping for cross-encoder/nli-deberta-v3-base
    # Output logits order: [contradiction, entailment, neutral]
    NLI_LABELS = ["contradiction", "entailment", "neutral"]

    def __init__(
        self,
        nli_model,
        confidence_threshold: float = 0.3,
        faithfulness_threshold: float = 0.5,
    ):
        """
        Args:
            nli_model: CrossEncoder loaded with cross-encoder/nli-deberta-v3-base
            confidence_threshold: Min reranker score to proceed with generation
            faithfulness_threshold: Min entailment score to consider faithful
        """
        self.nli_model = nli_model
        self.confidence_threshold = confidence_threshold
        self.faithfulness_threshold = faithfulness_threshold

    def gate_retrieval(self, retrieval_scores: list[float]) -> bool:
        """Check if retrieval confidence is high enough to proceed.

        Args:
            retrieval_scores: List of reranker scores from retrieval

        Returns:
            True if we should proceed with generation, False if we should refuse
        """
        if not retrieval_scores:
            return False

        top_score = max(retrieval_scores)
        should_proceed = top_score >= self.confidence_threshold

        if not should_proceed:
            print(f"🚫 Confidence gate: top score {top_score:.3f} "
                  f"< threshold {self.confidence_threshold}")

        return should_proceed

    def check_faithfulness(self, context: str, response: str) -> FaithfulnessResult:
        """Verify if the generated response is faithful to the context.

        Uses NLI (Natural Language Inference) to classify the relationship
        between context (premise) and response (hypothesis) as:
        - entailment: response is supported by context ✅
        - neutral: response neither supported nor contradicted ⚠️
        - contradiction: response contradicts context ❌

        Args:
            context: The retrieved context used for generation
            response: The model's generated response

        Returns:
            FaithfulnessResult with classification details
        """
        if not context.strip() or not response.strip():
            # No context means no faithfulness check possible
            return FaithfulnessResult(
                faithful=True,
                nli_label="neutral",
                nli_score=1.0,
                should_refuse=False,
            )

        # NLI model expects (premise, hypothesis) pairs
        scores = self.nli_model.predict(
            [(context, response)],
            apply_softmax=True,
        )

        # scores shape: (1, 3) → [contradiction, entailment, neutral]
        if hasattr(scores, '__len__') and len(scores) > 0:
            if hasattr(scores[0], '__len__'):
                score_array = scores[0]
            else:
                # Single score — model might be configured differently
                return FaithfulnessResult(
                    faithful=True,
                    nli_label="entailment",
                    nli_score=float(scores[0]),
                    should_refuse=False,
                )
        else:
            return FaithfulnessResult(
                faithful=True,
                nli_label="neutral",
                nli_score=0.5,
                should_refuse=False,
            )

        # Find the winning label
        label_idx = int(max(range(len(score_array)), key=lambda i: score_array[i]))
        label = self.NLI_LABELS[label_idx]
        confidence = float(score_array[label_idx])

        # Determine faithfulness
        is_faithful = label != "contradiction"
        entailment_score = float(score_array[1])  # entailment index

        # Should refuse if contradiction with high confidence
        should_refuse = (label == "contradiction" and confidence > self.faithfulness_threshold)

        print(f"🔍 Faithfulness: {label} (score={confidence:.3f}, "
              f"entailment={entailment_score:.3f})")

        return FaithfulnessResult(
            faithful=is_faithful,
            nli_label=label,
            nli_score=entailment_score,
            should_refuse=should_refuse,
        )

    @staticmethod
    def refusal_response() -> str:
        """Standard response when confidence is too low."""
        return ("I don't have enough information to answer that question accurately. "
                "Could you try uploading a relevant document or rephrasing your question?")

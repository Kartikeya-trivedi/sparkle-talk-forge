// Mock LLM responses — replace with real API call when backend is ready.
const RESPONSES = [
  `That's a thoughtful question. Let me think through this carefully.\n\nThere are a few angles worth considering:\n\n1. **Context matters** — the answer often depends on what you're optimizing for.\n2. **Trade-offs** are usually unavoidable. Every choice closes some doors while opening others.\n3. **Iteration beats prediction** — start small, learn, adjust.\n\nWould you like me to dig deeper into any of these?`,
  `Sure — here's a quick way to think about it.\n\nImagine you're building something for the first time. You don't yet know what'll matter most, so the best move is usually to:\n\n- Ship a small, working version\n- Watch what people actually do with it\n- Refine based on real signal, not guesses\n\nThis approach tends to outperform elaborate upfront planning, especially in unfamiliar territory.`,
  `Great prompt. Here's a structured take:\n\n## Summary\n\nThe core idea is straightforward, but the execution has nuance.\n\n## Key points\n\n- **Clarity first** — name the problem precisely before solving it\n- **Constraints help** — they narrow the search space\n- **Feedback loops** — the tighter, the better\n\n## A small example\n\n\`\`\`ts\nconst solve = (problem) => {\n  const clarified = clarify(problem);\n  const constrained = applyConstraints(clarified);\n  return iterate(constrained);\n};\n\`\`\`\n\nLet me know if you'd like me to expand any section.`,
  `Happy to help with that. A few quick thoughts:\n\n> The best questions are the ones that change how you see the problem.\n\nSo before answering directly — is your goal to *learn* this, *ship* this, or *teach* it to someone else? Each leads to a different kind of answer.`,
  `Absolutely. Here's how I'd approach it:\n\n1. Start with the user's actual need, not the feature you imagine\n2. Sketch the smallest possible version that delivers that need\n3. Build it, then talk to someone who'd use it\n4. Repeat\n\nThe tempting trap is to over-design before contact with reality. Resist it.`,
];

export function getMockResponse(): string {
  return RESPONSES[Math.floor(Math.random() * RESPONSES.length)];
}

// Simulate token-by-token streaming
export async function streamMockResponse(
  text: string,
  onToken: (chunk: string) => void,
  signal?: AbortSignal
): Promise<void> {
  // Split into word-ish chunks for realistic streaming feel
  const chunks = text.match(/(\s+|\S+)/g) ?? [text];
  for (const chunk of chunks) {
    if (signal?.aborted) return;
    await new Promise((r) => setTimeout(r, 18 + Math.random() * 40));
    onToken(chunk);
  }
}

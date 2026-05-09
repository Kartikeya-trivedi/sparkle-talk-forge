export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  source?: string;       // retrieved context chunk used (if any)
  modelUsed?: string;    // "llama-3.1-8b" | "gemma-4-26b"
  confidence?: number;   // retrieval confidence score (0-1)
  faithful?: boolean;    // NLI faithfulness check result
  cached?: boolean;      // was this a cache hit?
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
}

export const newId = () => Math.random().toString(36).slice(2, 11);

export function titleFromMessage(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, " ");
  if (trimmed.length <= 40) return trimmed;
  return trimmed.slice(0, 40).trimEnd() + "…";
}

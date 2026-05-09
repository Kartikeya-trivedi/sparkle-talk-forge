import ReactMarkdown from "react-markdown";
import { Copy, RefreshCw, ThumbsDown, ThumbsUp, Zap, Brain, AlertTriangle, Bookmark } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/lib/chatTypes";
import { useState } from "react";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

// ── Model Tier Badge ─────────────────────────────────────────────────────────
const ModelBadge = ({ modelUsed }: { modelUsed: string }) => {
  const isSmall = modelUsed.includes("llama");
  return (
    <span
      title={isSmall ? "Llama 3.1 8B (fast tier)" : "Gemma 4 26B (deep tier)"}
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide",
        isSmall
          ? "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
          : "bg-violet-500/10 text-violet-400 ring-1 ring-violet-500/20"
      )}
    >
      {isSmall ? <Zap className="h-2.5 w-2.5" /> : <Brain className="h-2.5 w-2.5" />}
      {isSmall ? "Llama 3.1 8B" : "Gemma 4 26B"}
    </span>
  );
};

// ── Confidence Bar ───────────────────────────────────────────────────────────
const ConfidenceBar = ({ score }: { score: number }) => {
  const pct = Math.round(score * 100);
  const color =
    pct >= 70 ? "bg-emerald-500" : pct >= 40 ? "bg-amber-400" : "bg-red-400";
  return (
    <span
      title={`Retrieval confidence: ${pct}%`}
      className="inline-flex items-center gap-1.5"
    >
      <span className="h-1.5 w-16 rounded-full bg-white/10 overflow-hidden">
        <span
          className={cn("h-full block rounded-full transition-all", color)}
          style={{ width: `${pct}%` }}
        />
      </span>
      <span className="text-[10px] text-muted-foreground">{pct}%</span>
    </span>
  );
};

// ── Cache Badge ──────────────────────────────────────────────────────────────
const CacheBadge = () => (
  <span
    title="Cache hit — response served from semantic cache"
    className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide bg-sky-500/10 text-sky-400 ring-1 ring-sky-500/20"
  >
    <Bookmark className="h-2.5 w-2.5" />
    Cached
  </span>
);

// ── Faithfulness Warning ─────────────────────────────────────────────────────
const FaithfulnessWarning = () => (
  <span
    title="NLI check: response may not be fully grounded in the retrieved context"
    className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-wide bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20"
  >
    <AlertTriangle className="h-2.5 w-2.5" />
    Verify
  </span>
);

export const MessageBubble = ({ message, isStreaming }: MessageBubbleProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  // Determine if we have any metadata to show
  const hasMetadata =
    !isStreaming &&
    message.content &&
    !isUser &&
    (message.modelUsed || message.confidence !== undefined || message.cached || message.faithful === false);

  if (isUser) {
    return (
      <div className="group flex justify-end animate-fade-in">
        <div className="max-w-[85%] rounded-2xl rounded-tr-md bg-user-bubble px-4 py-3 text-[15px] leading-relaxed text-user-bubble-foreground shadow-[var(--shadow-soft)]">
          <div className="whitespace-pre-wrap">{message.content}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="group flex gap-4 animate-fade-in">
      <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-primary/10 ring-1 ring-primary/20">
        <span className="font-serif text-[11px] font-semibold text-primary tracking-tight">KT</span>
      </div>
      <div className="min-w-0 flex-1 pt-1">
        <div className="prose-claude text-[15px]">
          <ReactMarkdown>{message.content || "\u200B"}</ReactMarkdown>
          {isStreaming && (
            <span className="inline-block h-4 w-[2px] translate-y-0.5 bg-foreground/70 ml-0.5 animate-blink" />
          )}
        </div>

        {/* RAG Metadata Row */}
        {hasMetadata && (
          <div className="mt-2 flex flex-wrap items-center gap-2">
            {message.modelUsed && <ModelBadge modelUsed={message.modelUsed} />}
            {message.cached && <CacheBadge />}
            {!message.cached && message.confidence !== undefined && message.confidence > 0 && (
              <ConfidenceBar score={message.confidence} />
            )}
            {message.faithful === false && <FaithfulnessWarning />}
          </div>
        )}

        {/* Action buttons */}
        {!isStreaming && message.content && (
          <div className="mt-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <IconBtn onClick={copy} label={copied ? "Copied" : "Copy"}>
              <Copy className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Retry">
              <RefreshCw className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Good response">
              <ThumbsUp className="h-3.5 w-3.5" />
            </IconBtn>
            <IconBtn label="Bad response">
              <ThumbsDown className="h-3.5 w-3.5" />
            </IconBtn>
          </div>
        )}
      </div>
    </div>
  );
};

const IconBtn = ({
  children,
  label,
  onClick,
}: {
  children: React.ReactNode;
  label: string;
  onClick?: () => void;
}) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    title={label}
    className={cn(
      "flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground",
      "hover:bg-muted hover:text-foreground transition-colors"
    )}
  >
    {children}
  </button>
);

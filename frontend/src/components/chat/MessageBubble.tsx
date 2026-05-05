import ReactMarkdown from "react-markdown";
import { Copy, RefreshCw, ThumbsDown, ThumbsUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/lib/chatTypes";
import { useState } from "react";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
}

export const MessageBubble = ({ message, isStreaming }: MessageBubbleProps) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

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

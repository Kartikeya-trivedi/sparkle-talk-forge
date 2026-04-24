import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface ComposerProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
}

export const Composer = ({ onSend, disabled, placeholder = "Reply to KT GPT…", autoFocus }: ComposerProps) => {
  const [value, setValue] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (autoFocus) taRef.current?.focus();
  }, [autoFocus]);

  // Auto-grow textarea
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 280) + "px";
  }, [value]);

  const submit = () => {
    const t = value.trim();
    if (!t || disabled) return;
    onSend(t);
    setValue("");
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="w-full">
      <div
        className={cn(
          "group relative rounded-3xl border border-border bg-surface-elevated",
          "shadow-[var(--shadow-composer)] transition-all duration-200",
          "focus-within:border-primary/40 focus-within:shadow-[0_4px_28px_-4px_hsl(var(--primary)/0.18)]"
        )}
      >
        <textarea
          ref={taRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            "w-full resize-none bg-transparent px-5 pt-4 pb-2",
            "text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/70",
            "outline-none scrollbar-thin"
          )}
        />
        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-1">
            <button
              type="button"
              className="flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              <Paperclip className="h-3.5 w-3.5" />
              Attach
            </button>
            <button
              type="button"
              className="flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Tools
            </button>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline text-[11px] text-muted-foreground/70 font-mono">
              KT GPT v1
            </span>
            <button
              type="button"
              onClick={submit}
              disabled={!value.trim() || disabled}
              aria-label="Send message"
              className={cn(
                "flex h-8 w-8 items-center justify-center rounded-full transition-all duration-200",
                value.trim() && !disabled
                  ? "bg-primary text-primary-foreground hover:opacity-90 scale-100"
                  : "bg-muted text-muted-foreground/50 scale-95"
              )}
            >
              <ArrowUp className="h-4 w-4" strokeWidth={2.5} />
            </button>
          </div>
        </div>
      </div>
      <p className="mt-2 text-center text-[11px] text-muted-foreground/70">
        KT GPT can make mistakes. Please double-check responses.
      </p>
    </div>
  );
};

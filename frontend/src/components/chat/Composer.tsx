import { useEffect, useRef, useState } from "react";
import { ArrowUp, Paperclip, Globe, FileText, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface UploadedDoc {
  filename: string;
  sentences: number;
}

interface ComposerProps {
  onSend: (text: string, webSearch?: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
  autoFocus?: boolean;
  uploadedDocs?: UploadedDoc[];
  onFileUpload?: (doc: UploadedDoc) => void;
}

export const Composer = ({
  onSend,
  disabled,
  placeholder = "Reply to KT GPT…",
  autoFocus,
  uploadedDocs = [],
  onFileUpload,
}: ComposerProps) => {
  const [value, setValue] = useState("");
  const [uploading, setUploading] = useState(false);
  const [webSearch, setWebSearch] = useState(false);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

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
    onSend(t, webSearch);
    setValue("");
  };

  const onKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !onFileUpload) return;

    setUploading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();
      onFileUpload({ filename: data.filename, sentences: data.sentences });
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  return (
    <div className="w-full">
      {/* Uploaded docs pills */}
      {uploadedDocs.length > 0 && (
        <div className="mb-2 flex flex-wrap items-center gap-1.5 px-1">
          {uploadedDocs.map((doc, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-3 py-1 text-[11px] font-medium text-primary"
            >
              <FileText className="h-3 w-3" />
              {doc.filename}
              <span className="text-primary/60">({doc.sentences} chunks)</span>
            </span>
          ))}
        </div>
      )}

      <div
        className={cn(
          "group relative rounded-3xl border border-border bg-surface-elevated",
          "shadow-[var(--shadow-composer)] transition-all duration-200",
          "focus-within:border-primary/40 focus-within:shadow-[0_4px_28px_-4px_hsl(var(--primary)/0.18)]",
          webSearch && "ring-2 ring-blue-500/30 border-blue-500/40"
        )}
      >
        {/* Web search indicator */}
        {webSearch && (
          <div className="flex items-center gap-1.5 px-5 pt-3 pb-0">
            <Globe className="h-3.5 w-3.5 text-blue-500" />
            <span className="text-[11px] font-medium text-blue-500">Web Search enabled</span>
          </div>
        )}

        <textarea
          ref={taRef}
          rows={1}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder={webSearch ? "Search the web and ask KT GPT…" : placeholder}
          disabled={disabled}
          className={cn(
            "w-full resize-none bg-transparent px-5 pt-4 pb-2",
            "text-[15px] leading-relaxed text-foreground placeholder:text-muted-foreground/70",
            "outline-none scrollbar-thin",
            webSearch && "pt-2"
          )}
        />
        <div className="flex items-center justify-between px-3 pb-3 pt-1">
          <div className="flex items-center gap-1">
            {/* Hidden file input */}
            <input
              ref={fileRef}
              type="file"
              accept=".txt,.md,.pdf"
              onChange={handleFileSelect}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileRef.current?.click()}
              disabled={uploading}
              className={cn(
                "flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors",
                uploading
                  ? "text-primary animate-pulse"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              {uploading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Paperclip className="h-3.5 w-3.5" />
              )}
              {uploading ? "Indexing…" : "Attach"}
            </button>
            <button
              type="button"
              onClick={() => setWebSearch(!webSearch)}
              className={cn(
                "flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-colors",
                webSearch
                  ? "bg-blue-500/15 text-blue-500 ring-1 ring-blue-500/25"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              <Globe className="h-3.5 w-3.5" />
              Search
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

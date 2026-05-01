import { useEffect, useRef, useState } from "react";
import { ChevronDown, Plus, Sun, Moon } from "lucide-react";
import { Welcome } from "@/components/chat/Welcome";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { Composer } from "@/components/chat/Composer";
import { type Conversation, type Message, newId, titleFromMessage } from "@/lib/chatTypes";
import { getMockResponse, streamMockResponse } from "@/lib/mockLlm";
import { useTheme } from "@/lib/theme";

const Index = () => {
  const { theme, toggle } = useTheme();
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [conversation?.messages.length, conversation?.messages[conversation.messages.length - 1]?.content]);

  const newChat = () => {
    abortRef.current?.abort();
    setStreaming(false);
    setConversation(null);
  };

  const handleSend = async (text: string) => {
    if (streaming) return;

    const userMsg: Message = { id: newId(), role: "user", content: text, createdAt: Date.now() };
    const assistantMsg: Message = { id: newId(), role: "assistant", content: "", createdAt: Date.now() };

    setConversation((prev) => {
      if (!prev) {
        return {
          id: newId(),
          title: titleFromMessage(text),
          messages: [userMsg, assistantMsg],
          createdAt: Date.now(),
        };
      }
      return { ...prev, messages: [...prev.messages, userMsg, assistantMsg] };
    });

    setStreaming(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const response = getMockResponse();
    await new Promise((r) => setTimeout(r, 350));

    await streamMockResponse(
      response,
      (chunk) => {
        setConversation((prev) => {
          if (!prev) return prev;
          const msgs = prev.messages.slice();
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") {
            msgs[msgs.length - 1] = { ...last, content: last.content + chunk };
          }
          return { ...prev, messages: msgs };
        });
      },
      ctrl.signal
    );

    setStreaming(false);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <main className="relative flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex h-14 items-center justify-between px-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 pr-2">
              <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary">
                <span className="font-serif text-[11px] font-bold text-primary-foreground tracking-tight">KT</span>
              </div>
              <span className="font-serif text-[15px] font-semibold tracking-tight">KT GPT</span>
            </div>
            <button className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium hover:bg-muted transition-colors">
              <span className="font-serif">KT GPT v1</span>
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={newChat}
              className="flex h-8 items-center gap-1.5 rounded-md border border-border bg-surface-elevated px-3 text-xs font-medium text-foreground/80 hover:bg-muted transition-colors"
            >
              <Plus className="h-3.5 w-3.5" />
              New chat
            </button>
            <button
              onClick={toggle}
              aria-label="Toggle theme"
              className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </button>
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/15 ring-1 ring-primary/25">
              <span className="text-xs font-semibold text-primary">YO</span>
            </div>
          </div>
        </header>

        {/* Chat body */}
        {!conversation ? (
          <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
            <Welcome onSend={handleSend} />
          </div>
        ) : (
          <>
            <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
              <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-40 space-y-8">
                {conversation.messages.map((m, i) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    isStreaming={streaming && i === conversation.messages.length - 1 && m.role === "assistant"}
                  />
                ))}
              </div>
            </div>
            <div className="pointer-events-none absolute inset-x-0 bottom-0">
              <div className="h-12 bg-[var(--gradient-fade)]" />
              <div className="pointer-events-auto bg-background pb-4 pt-2">
                <div className="mx-auto w-full max-w-3xl px-4">
                  <Composer onSend={handleSend} disabled={streaming} />
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
};

export default Index;

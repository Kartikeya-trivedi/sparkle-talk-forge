import { useEffect, useRef, useState } from "react";
import { ChevronDown, PanelLeftOpen, Share2 } from "lucide-react";
import { Sidebar } from "@/components/chat/Sidebar";
import { Welcome } from "@/components/chat/Welcome";
import { MessageBubble } from "@/components/chat/MessageBubble";
import { Composer } from "@/components/chat/Composer";
import { type Conversation, type Message, newId, titleFromMessage } from "@/lib/chatTypes";
import { getMockResponse, streamMockResponse } from "@/lib/mockLlm";

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const active = conversations.find((c) => c.id === activeId) ?? null;

  // Auto-scroll on new content
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [active?.messages.length, active?.messages[active.messages.length - 1]?.content]);

  const newChat = () => {
    setActiveId(null);
    abortRef.current?.abort();
    setStreaming(false);
  };

  const updateConversation = (id: string, updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  };

  const handleSend = async (text: string) => {
    if (streaming) return;
    let convId = activeId;

    const userMsg: Message = { id: newId(), role: "user", content: text, createdAt: Date.now() };
    const assistantMsg: Message = { id: newId(), role: "assistant", content: "", createdAt: Date.now() };

    if (!convId) {
      const conv: Conversation = {
        id: newId(),
        title: titleFromMessage(text),
        messages: [userMsg, assistantMsg],
        createdAt: Date.now(),
      };
      convId = conv.id;
      setConversations((prev) => [conv, ...prev]);
      setActiveId(convId);
    } else {
      updateConversation(convId, (c) => ({ ...c, messages: [...c.messages, userMsg, assistantMsg] }));
    }

    setStreaming(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const response = getMockResponse();
    // small initial delay to feel natural
    await new Promise((r) => setTimeout(r, 350));

    await streamMockResponse(
      response,
      (chunk) => {
        updateConversation(convId!, (c) => {
          const msgs = c.messages.slice();
          const last = msgs[msgs.length - 1];
          if (last?.role === "assistant") {
            msgs[msgs.length - 1] = { ...last, content: last.content + chunk };
          }
          return { ...c, messages: msgs };
        });
      },
      ctrl.signal
    );

    setStreaming(false);
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={(id) => setActiveId(id)}
        onNewChat={newChat}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((s) => !s)}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex h-14 items-center justify-between px-4 flex-shrink-0">
          <div className="flex items-center gap-2">
            {!sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
                className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              >
                <PanelLeftOpen className="h-4 w-4" />
              </button>
            )}
            <button className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium hover:bg-muted transition-colors">
              <span className="font-serif">Claude Sonnet 4</span>
              <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button className="hidden sm:flex h-8 items-center gap-1.5 rounded-md border border-border bg-surface-elevated px-3 text-xs font-medium text-foreground/80 hover:bg-muted transition-colors">
              <Share2 className="h-3.5 w-3.5" />
              Share
            </button>
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary/15 ring-1 ring-primary/25">
              <span className="text-xs font-semibold text-primary">YO</span>
            </div>
          </div>
        </header>

        {/* Chat body */}
        {!active ? (
          <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
            <Welcome onSend={handleSend} />
          </div>
        ) : (
          <>
            <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin">
              <div className="mx-auto w-full max-w-3xl px-4 pt-6 pb-40 space-y-8">
                {active.messages.map((m, i) => (
                  <MessageBubble
                    key={m.id}
                    message={m}
                    isStreaming={streaming && i === active.messages.length - 1 && m.role === "assistant"}
                  />
                ))}
              </div>
            </div>
            {/* Floating composer with fade */}
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

import { ChevronDown, FileText, Lightbulb, Pencil, Sparkles } from "lucide-react";
import { Composer } from "./Composer";

interface WelcomeProps {
  onSend: (text: string) => void;
}

const SUGGESTIONS = [
  { icon: Pencil, label: "Help me write", prompt: "Help me write a thoughtful email declining a meeting." },
  { icon: Lightbulb, label: "Brainstorm ideas", prompt: "Brainstorm five unconventional weekend project ideas for a developer." },
  { icon: FileText, label: "Summarize", prompt: "Summarize the key principles of effective product design." },
  { icon: Sparkles, label: "Explain a concept", prompt: "Explain how transformers work in machine learning, in plain language." },
];

export const Welcome = ({ onSend }: WelcomeProps) => {
  const hour = new Date().getHours();
  const greet = hour < 5 ? "Still up" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pt-16 pb-8 animate-slide-up">
      <div className="mb-10 flex flex-col items-center text-center">
        <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-2xl bg-primary shadow-[var(--shadow-elevated)]">
          <span className="font-serif text-2xl font-bold text-primary-foreground">C</span>
        </div>
        <h1 className="font-serif text-[32px] sm:text-[40px] leading-tight font-medium tracking-tight">
          <span className="text-primary">✦</span> {greet}
        </h1>
        <p className="mt-3 text-base text-muted-foreground">How can I help you today?</p>
      </div>

      <Composer onSend={onSend} placeholder="How can I help you today?" autoFocus />

      <div className="mt-8 flex flex-wrap items-center justify-center gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            onClick={() => onSend(s.prompt)}
            className="group flex items-center gap-2 rounded-full border border-border bg-surface-elevated px-4 py-2 text-[13px] text-foreground/80 hover:border-primary/40 hover:text-foreground hover:shadow-[var(--shadow-soft)] transition-all"
          >
            <s.icon className="h-3.5 w-3.5 text-primary" />
            {s.label}
          </button>
        ))}
        <button className="flex items-center gap-1 rounded-full px-3 py-2 text-[13px] text-muted-foreground hover:text-foreground transition-colors">
          More <ChevronDown className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
};

import { MessageSquarePlus, MessagesSquare, Moon, PanelLeftClose, Search, Settings, Sparkles, Sun } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/lib/theme";
import type { Conversation } from "@/lib/chatTypes";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
  open: boolean;
  onToggle: () => void;
}

export const Sidebar = ({ conversations, activeId, onSelect, onNewChat, open, onToggle }: SidebarProps) => {
  const { theme, toggle } = useTheme();
  return (
    <aside
      className={cn(
        "flex flex-col bg-sidebar border-r border-sidebar-border transition-all duration-300 ease-out",
        "overflow-hidden",
        open ? "w-[260px]" : "w-0"
      )}
    >
      <div className="flex h-14 items-center justify-between px-3 flex-shrink-0">
        <div className="flex items-center gap-2 pl-1">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary">
            <span className="font-serif text-[11px] font-bold text-primary-foreground tracking-tight">KT</span>
          </div>
          <span className="font-serif text-[15px] font-semibold tracking-tight">KT GPT</span>
        </div>
        <button
          onClick={onToggle}
          aria-label="Close sidebar"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-sidebar-accent hover:text-foreground transition-colors"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>

      <div className="px-3 space-y-0.5">
        <NavItem icon={MessageSquarePlus} label="New chat" onClick={onNewChat} primary />
        <NavItem icon={Search} label="Search chats" />
        <NavItem icon={Sparkles} label="Projects" />
        <NavItem icon={MessagesSquare} label="Chats" />
      </div>

      <div className="mt-4 px-3 pb-1">
        <p className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground/70 px-3 py-1">
          Recents
        </p>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 pb-3 scrollbar-thin">
        {conversations.length === 0 ? (
          <p className="px-3 py-2 text-xs text-muted-foreground/70">No chats yet</p>
        ) : (
          <ul className="space-y-0.5">
            {conversations.map((c) => (
              <li key={c.id}>
                <button
                  onClick={() => onSelect(c.id)}
                  className={cn(
                    "w-full truncate rounded-md px-3 py-2 text-left text-[13px] transition-colors",
                    activeId === c.id
                      ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                      : "text-sidebar-foreground hover:bg-sidebar-accent/60"
                  )}
                  title={c.title}
                >
                  {c.title}
                </button>
              </li>
            ))}
          </ul>
        )}
      </nav>

      <div className="border-t border-sidebar-border p-3 space-y-1">
        <button
          onClick={toggle}
          aria-label="Toggle theme"
          className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-[13px] text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
        >
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
        </button>
        <button className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-sidebar-accent transition-colors">
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-primary/15 ring-1 ring-primary/25">
            <span className="text-xs font-semibold text-primary">YO</span>
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-medium">Your account</p>
            <p className="truncate text-[11px] text-muted-foreground">Free plan</p>
          </div>
          <Settings className="h-4 w-4 text-muted-foreground" />
        </button>
      </div>
    </aside>
  );
};

const NavItem = ({
  icon: Icon,
  label,
  onClick,
  primary,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  onClick?: () => void;
  primary?: boolean;
}) => (
  <button
    onClick={onClick}
    className={cn(
      "flex w-full items-center gap-3 rounded-md px-3 py-2 text-[13px] transition-colors",
      primary
        ? "text-primary hover:bg-primary/10 font-medium"
        : "text-sidebar-foreground hover:bg-sidebar-accent"
    )}
  >
    <Icon className="h-4 w-4" />
    <span>{label}</span>
  </button>
);

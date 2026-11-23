"""Persistent cross-conversation memory for the Flavonoid Agent.

Fully automatic: the agent itself decides what's worth remembering (see
core/agent_tools.py's remember_fact tool + SYSTEM_PROMPT's memory-writing
guidance) -- there is no user-curated file to maintain. Mirrors the memory
file-based memory layout used by coding assistants, simplified to
one flat file with a free-text category tag instead of typed memory files.

Storage lives outside the git repo (same directory as agent_config.json),
so accumulated notes about the user/project never risk being committed.
"""

from datetime import date
from pathlib import Path

MEMORY_PATH = Path.home() / ".annomass" / "agent_memory.md"


def load_memory(max_chars: int = 4000) -> str:
    """Returns the memory file's content, empty string if absent. If longer
    than max_chars, keeps only the MOST RECENT entries (simple recency-based
    truncation -- unlike conversation compaction, this doesn't re-summarize,
    it just drops the oldest bullets to fit the budget)."""
    if not MEMORY_PATH.exists():
        return ""
    text = MEMORY_PATH.read_text(encoding="utf-8", errors="replace")
    if len(text) <= max_chars:
        return text

    lines = [l for l in text.splitlines() if l.strip()]
    kept, total = [], 0
    for line in reversed(lines):
        total += len(line) + 1
        if total > max_chars:
            break
        kept.append(line)
    kept.reverse()
    return "\n".join(kept)


def append_memory(fact: str, category: str = "general") -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    line = f"- [{today}] ({category}) {fact}\n"
    with open(MEMORY_PATH, "a", encoding="utf-8") as f:
        f.write(line)

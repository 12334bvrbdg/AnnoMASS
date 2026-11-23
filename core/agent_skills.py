"""Local skill registry for the Flavonoid Agent.

Skills are Markdown instruction files stored outside the repository under
``~/.annomass/agent_skills``. They are intentionally plain text: easy to
inspect, export, edit, and remove without a package manager. The Agent loads
enabled skills into its system prompt on each turn.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional


SKILL_DIR = Path.home() / ".annomass" / "agent_skills"
MAX_SKILL_CHARS = 12000


def _slugify(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("_")
    return text[:80] or "skill"


def _skill_path(name: str) -> Path:
    slug = _slugify(name)
    if not slug.lower().endswith(".md"):
        slug += ".md"
    return SKILL_DIR / slug


def list_skills() -> List[Dict[str, object]]:
    try:
        SKILL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return []
    rows = []
    for path in sorted(SKILL_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace")
        title = path.stem
        for line in text.splitlines():
            if line.startswith("#"):
                title = line.lstrip("#").strip() or title
                break
        rows.append({
            "name": path.stem,
            "title": title,
            "path": str(path),
            "chars": len(text),
        })
    return rows


def install_skill(name: str, content: Optional[str] = None, source_path: Optional[str] = None) -> Dict[str, object]:
    """Install/update a local Markdown skill.

    Provide either ``content`` or ``source_path``. The target filename is always
    sanitized and kept inside ``SKILL_DIR``.
    """
    try:
        SKILL_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return {"error": f"Cannot create skill directory {SKILL_DIR}: {exc}"}
    out_path = _skill_path(name)
    if source_path:
        src = Path(source_path).expanduser()
        if not src.exists() or not src.is_file():
            return {"error": f"Skill source file not found: {source_path}"}
        shutil.copyfile(src, out_path)
    else:
        text = str(content or "").strip()
        if not text:
            return {"error": "Skill content is empty. Provide Markdown content or source_path."}
        if not text.startswith("#"):
            text = f"# {_slugify(name)}\n\n{text}"
        out_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return {"name": out_path.stem, "path": str(out_path), "installed": True}


def read_skill(name: str) -> Dict[str, object]:
    path = _skill_path(name)
    if not path.exists() or not path.is_file():
        return {"error": f"Skill not found: {name}"}
    return {"name": path.stem, "path": str(path), "content": path.read_text(encoding="utf-8", errors="replace")}


def delete_skill(name: str) -> Dict[str, object]:
    path = _skill_path(name)
    if not path.exists() or not path.is_file():
        return {"error": f"Skill not found: {name}"}
    try:
        path.unlink()
    except OSError as exc:
        return {"error": f"Could not delete skill {name}: {exc}"}
    return {"name": path.stem, "deleted": True}


def load_skills_prompt(max_chars: int = MAX_SKILL_CHARS) -> str:
    rows = list_skills()
    if not rows:
        return ""
    chunks = []
    used = 0
    for row in rows:
        path = Path(str(row["path"]))
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        header = f"\n\n[Installed Agent Skill: {row['name']}]\n"
        budget = max_chars - used - len(header)
        if budget <= 0:
            break
        if len(text) > budget:
            text = text[:budget].rstrip() + "\n[truncated]"
        chunks.append(header + text)
        used += len(header) + len(text)
    return "".join(chunks).strip()

"""Config/secrets storage for the Flavonoid Agent (see core/agent_llm_client.py,
core/agent_tools.py, and the "Flavonoid Agent" tab in annomass_app.py).

Stored OUTSIDE the git repo entirely (~/.annomass/agent_config.json), since
this project has no existing secrets convention to build on -- this keeps an
API key from ever ending up in a commit by construction, not by discipline.
"""

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".annomass"
CONFIG_PATH = CONFIG_DIR / "agent_config.json"


def migrate_legacy_config_dir():
    """Migrate legacy per-user config directories into ~/.annomass."""
    new_dir = CONFIG_DIR
    if new_dir.exists():
        return

    for old_dir in (Path.home() / ".flavomass", Path.home() / ".deepmass2"):
        if not old_dir.exists():
            continue
        try:
            import shutil
            shutil.move(str(old_dir), str(new_dir))
            print(f"[AnnoMASS] Migrated config directory: {old_dir} -> {new_dir}")
        except Exception as e:
            print(f"[AnnoMASS] Config directory migration failed (starting fresh): {e}")
        return

# Pre-fill suggestions only -- every field (base_url, model, api_key) stays
# user-editable in the Settings dialog. "protocol" picks which adapter in
# core/agent_llm_client.py handles the provider: "openai_compatible" (the
# vast majority -- DeepSeek/OpenAI/Qwen-DashScope-compatible-mode/Doubao-Ark-
# OpenAI-compatible-mode all speak this) or "anthropic" (Claude's Messages API).
# "context_window" is a best-effort approximate token budget used only to
# decide when to trigger auto-compaction (core/agent_tools.py's
# run_agent_loop) -- not an exact per-model guarantee. Deliberately
# conservative for providers/models we can't pin down precisely (Doubao's
# endpoint-specific limits, "custom"), so compaction triggers comfortably
# before a real hard failure rather than right at the edge.
PROVIDER_PRESETS = {
    "deepseek": {
        "label": "DeepSeek",
        "protocol": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        # DeepSeek's two current model IDs. Shown as suggestions in an
        # editable combo box -- the field stays free text so a newer model
        # id doesn't require a code change.
        "model_choices": ["deepseek-chat", "deepseek-reasoner"],
        "context_window": 64000,
    },
    "openai": {
        "label": "OpenAI",
        "protocol": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "model_choices": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o3-mini"],
        "context_window": 128000,
    },
    "qwen": {
        "label": "Qwen (DashScope)",
        "protocol": "openai_compatible",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "model_choices": ["qwen-plus", "qwen-turbo", "qwen-max"],
        "context_window": 128000,
    },
    "doubao": {
        "label": "Doubao (Volcengine Ark)",
        "protocol": "openai_compatible",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        # Ark identifies deployments by an opaque "endpoint ID" (ep-...), not
        # a friendly model name -- left blank on purpose, see the Settings
        # dialog's placeholder text for this provider. No fixed choices
        # make sense here since every user has different endpoint IDs.
        "model": "",
        "model_choices": [],
        "context_window": 32000,
    },
    "volcengine_ark": {
        "label": "Volcengine Ark",
        "protocol": "openai_compatible",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        # Ark accepts OpenAI-compatible chat/completions calls, but the value
        # sent in the "model" field can be either a model/endpoint identifier
        # exposed by the user's Ark console or an endpoint ID. Keep this
        # editable; these are only convenience suggestions.
        "model": "",
        "model_choices": [
            "doubao-seed-1-6",
            "doubao-seed-1-6-thinking",
            "doubao-seed-1-6-flash",
            "doubao-1-5-pro-32k",
            "doubao-1-5-lite-32k",
            "deepseek-v3",
            "deepseek-r1",
            "ep-xxxxxxxxxxxxxxxxxxxxx",
        ],
        "context_window": 128000,
    },
    "moonshot": {
        "label": "Moonshot (Kimi)",
        "protocol": "openai_compatible",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-32k",
        "model_choices": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "context_window": 128000,
    },
    "zhipu": {
        "label": "Zhipu (GLM)",
        "protocol": "openai_compatible",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "model_choices": ["glm-4-plus", "glm-4-air", "glm-4-flash"],
        "context_window": 128000,
    },
    "claude": {
        "label": "Claude (Anthropic)",
        "protocol": "anthropic",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-5",
        "model_choices": ["claude-sonnet-5", "claude-opus-4-8", "claude-haiku-4-5-20251001", "claude-fable-5"],
        "context_window": 200000,
    },
    "custom": {
        "label": "Custom (OpenAI-compatible)",
        "protocol": "openai_compatible",
        "base_url": "",
        "model": "",
        "model_choices": [],
        "context_window": 32000,
    },
}

# How much the agent explores/verifies before answering -- a global,
# provider-independent behavior lever (not a per-provider setting, so it
# lives alongside active_provider rather than inside a provider entry).
# Implemented as a system-prompt instruction (see core/agent_tools.py's
# EFFORT_INSTRUCTIONS), not a provider-native "reasoning" API parameter --
# deliberately, since most configured providers here (DeepSeek/Qwen/Doubao/
# custom OpenAI-compatible) have no such parameter, and a native "thinking"
# feature (e.g. Anthropic's) would itself spend MORE tokens, working against
# the "fewest tokens for the best result" goal this setting is for. "low"
# genuinely uses fewer tokens (fewer tool calls, terser answers); "high"
# spends more only when the task (e.g. an ambiguous isomer call) warrants it.
THINKING_EFFORT_LEVELS = ["low", "medium", "high"]
DEFAULT_THINKING_EFFORT = "medium"

DEFAULT_CONFIG = {"active_provider": None, "providers": {}, "thinking_effort": DEFAULT_THINKING_EFFORT}


def load_config() -> dict:
    """Never raises -- returns a default-shaped dict if the file is absent or unreadable."""
    if not CONFIG_PATH.exists():
        return dict(DEFAULT_CONFIG)
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_CONFIG)
    data.setdefault("active_provider", None)
    data.setdefault("providers", {})
    if data.get("thinking_effort") not in THINKING_EFFORT_LEVELS:
        data["thinking_effort"] = DEFAULT_THINKING_EFFORT
    return data


def get_thinking_effort(cfg: dict = None) -> str:
    cfg = cfg if cfg is not None else load_config()
    level = cfg.get("thinking_effort")
    return level if level in THINKING_EFFORT_LEVELS else DEFAULT_THINKING_EFFORT


def set_thinking_effort(cfg: dict, level: str) -> dict:
    cfg["thinking_effort"] = level if level in THINKING_EFFORT_LEVELS else DEFAULT_THINKING_EFFORT
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        os.chmod(CONFIG_PATH, 0o600)
    except OSError:
        pass  # best-effort hardening; some filesystems (e.g. certain mounts) don't support chmod


def get_active_provider_config(cfg: dict = None):
    """Return (provider_key, provider_cfg dict with 'protocol'/'base_url'/'model'/'api_key') or (None, None)."""
    cfg = cfg if cfg is not None else load_config()
    key = cfg.get("active_provider")
    if not key or key not in cfg.get("providers", {}):
        return None, None
    stored = cfg["providers"][key]
    preset = PROVIDER_PRESETS.get(key, {})
    protocol = stored.get("protocol") or preset.get("protocol")
    if not protocol:
        return None, None
    provider_cfg = {
        "provider_key": key,
        "protocol": protocol,
        "base_url": stored.get("base_url") or preset.get("base_url", ""),
        "model": stored.get("model") or preset.get("model", ""),
        "api_key": stored.get("api_key", ""),
        "context_window": stored.get("context_window") or preset.get("context_window", 32000),
    }
    return key, provider_cfg


def set_provider_config(cfg: dict, provider_key: str, base_url: str, model: str, api_key: str, protocol: str = None) -> dict:
    """Update cfg in place with one provider's settings and mark it active. Returns cfg."""
    cfg.setdefault("providers", {})
    entry = {"base_url": base_url, "model": model, "api_key": api_key}
    if provider_key == "custom" or protocol is not None:
        entry["protocol"] = protocol or PROVIDER_PRESETS.get(provider_key, {}).get("protocol", "openai_compatible")
    cfg["providers"][provider_key] = entry
    cfg["active_provider"] = provider_key
    return cfg

"""Provider-agnostic LLM client for the Flavonoid Agent.

Hand-rolled HTTP via `requests` (already a project dependency) rather than
adding the `openai`/`anthropic`/`dashscope`/`zhipuai`/`volcengine` SDKs --
none of those are installed, and DeepSeek/OpenAI/Qwen(DashScope compatible
mode)/Doubao(Volcengine Ark OpenAI-compatible mode) all speak the same
OpenAI-style /chat/completions + tools shape, so one adapter covers four of
the five providers named in core/agent_config.py's PROVIDER_PRESETS. Only
Claude (Anthropic Messages API) needs a second adapter.

Both adapters normalize to the same ChatReply/ToolCall shape so nothing
downstream (core/agent_tools.py's run_agent_loop, AgentWorker in
annomass_app.py) needs to know which provider is active.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

DEFAULT_TIMEOUT = 120


class LLMClientError(Exception):
    """Any HTTP/auth/parse failure talking to a provider. Never let a raw
    requests/JSON exception escape past this module -- callers (AgentWorker)
    catch this one type and turn it into a `failed` signal."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ChatReply:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_assistant_message: Any = None  # provider-native shape, needed to build the next request
    usage: Optional[Dict[str, int]] = None  # normalized {"input_tokens","output_tokens","total_tokens"}


def translate_tool_schemas(canonical_schemas: List[Dict[str, Any]], protocol: str) -> List[Dict[str, Any]]:
    """canonical_schemas: [{"name", "description", "parameters": <JSON schema object>}, ...]"""
    if protocol == "anthropic":
        return [
            {
                "name": s["name"],
                "description": s.get("description", ""),
                "input_schema": s.get("parameters") or {"type": "object", "properties": {}},
            }
            for s in canonical_schemas
        ]
    # openai_compatible (default)
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for s in canonical_schemas
    ]


class OpenAICompatibleAdapter:
    """DeepSeek / OpenAI / Qwen (DashScope compatible-mode) / Doubao (Ark
    OpenAI-compatible-mode) / custom -- all speak this shape."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
             system: Optional[str] = None) -> ChatReply:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        sendable_messages = ([{"role": "system", "content": system}] + messages) if system else messages
        body: Dict[str, Any] = {"model": self.model, "messages": sendable_messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMClientError(f"Request to {url} failed: {exc}") from exc
        if resp.status_code != 200:
            raise LLMClientError(f"{self.base_url} returned HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
            choice = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMClientError(f"Unexpected response shape from {self.base_url}: {exc}") from exc

        tool_calls = []
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except (ValueError, KeyError):
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))

        raw_usage = data.get("usage") or {}
        usage = None
        if raw_usage:
            usage = {
                "input_tokens": raw_usage.get("prompt_tokens", 0),
                "output_tokens": raw_usage.get("completion_tokens", 0),
                "total_tokens": raw_usage.get("total_tokens", 0),
            }

        return ChatReply(text=choice.get("content") or "", tool_calls=tool_calls, raw_assistant_message=choice, usage=usage)

    def build_tool_result_message(self, tool_call: ToolCall, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        }

    def build_assistant_message(self, reply: ChatReply) -> Dict[str, Any]:
        return reply.raw_assistant_message


class AnthropicAdapter:
    """Claude only -- different request/response shape (Messages API)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = DEFAULT_TIMEOUT, max_tokens: int = 4096):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None,
             system: Optional[str] = None) -> ChatReply:
        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body: Dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": messages}
        if system:
            body["system"] = system
        if tools:
            body["tools"] = tools
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise LLMClientError(f"Request to {url} failed: {exc}") from exc
        if resp.status_code != 200:
            raise LLMClientError(f"{self.base_url} returned HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            data = resp.json()
            content_blocks = data["content"]
        except (ValueError, KeyError) as exc:
            raise LLMClientError(f"Unexpected response shape from {self.base_url}: {exc}") from exc

        text_parts, tool_calls = [], []
        for block in content_blocks:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(id=block["id"], name=block["name"], arguments=block.get("input") or {}))

        raw_usage = data.get("usage") or {}
        usage = None
        if raw_usage:
            input_tok = raw_usage.get("input_tokens", 0)
            output_tok = raw_usage.get("output_tokens", 0)
            usage = {"input_tokens": input_tok, "output_tokens": output_tok, "total_tokens": input_tok + output_tok}

        return ChatReply(text="".join(text_parts), tool_calls=tool_calls, raw_assistant_message=content_blocks, usage=usage)

    def build_tool_result_message(self, tool_call: ToolCall, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }],
        }

    def build_assistant_message(self, reply: ChatReply) -> Dict[str, Any]:
        return {"role": "assistant", "content": reply.raw_assistant_message}


def build_llm_client(provider_cfg: Dict[str, Any]):
    """provider_cfg: {"protocol", "base_url", "model", "api_key"} (see core.agent_config)."""
    protocol = provider_cfg.get("protocol", "openai_compatible")
    base_url = provider_cfg.get("base_url", "")
    model = provider_cfg.get("model", "")
    api_key = provider_cfg.get("api_key", "")
    if not base_url or not model or not api_key:
        raise LLMClientError("Provider is not fully configured (base_url/model/api_key required).")
    if protocol == "anthropic":
        return AnthropicAdapter(base_url, api_key, model)
    return OpenAICompatibleAdapter(base_url, api_key, model)


def test_connection(provider_cfg: Dict[str, Any]):
    """One minimal no-tools chat call. Returns (ok: bool, message: str) -- never raises."""
    try:
        client = build_llm_client(provider_cfg)
        protocol = provider_cfg.get("protocol", "openai_compatible")
        messages = [{"role": "user", "content": "Reply with the single word: OK"}]
        if protocol == "anthropic":
            reply = client.chat(messages, tools=None, system=None)
        else:
            reply = client.chat(messages, tools=None)
        preview = (reply.text or "").strip()[:80]
        return True, f"Connected. Model replied: {preview!r}" if preview else "Connected."
    except LLMClientError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - defensive catch-all for a UI-facing check
        return False, f"Unexpected error: {exc}"

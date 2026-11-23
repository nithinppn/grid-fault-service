"""LLM backends for the operator report.

The report loop in `explain.py` speaks the Anthropic Messages format. `GroqAdapter` gives a
Groq (OpenAI-compatible) client the same `client.messages.create(...)` interface by
translating requests and responses, so the loop and the grounding check are identical for
both providers.

Provider selection: GRIDFAULT_LLM_PROVIDER=anthropic|groq, or automatically whichever of
ANTHROPIC_API_KEY / GROQ_API_KEY is set (Anthropic first). GRIDFAULT_LLM_MODEL overrides the
default model for the chosen provider.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "groq": "openai/gpt-oss-120b"}


@dataclass
class LLM:
    client: Any  # exposes .messages.create(...) in Anthropic format
    model: str
    provider: str


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    names: dict[str, str] = {}
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b["type"] == "text")
            calls = []
            for b in content:
                if b["type"] == "tool_use":
                    names[b["id"]] = b["name"]
                    calls.append(
                        {
                            "id": b["id"],
                            "type": "function",
                            "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                        }
                    )
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:  # user turn carrying tool results
            for b in content:
                if b["type"] == "tool_result":
                    body = b["content"]
                    if b.get("is_error"):
                        body = f"ERROR: {body}"
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": b["tool_use_id"],
                            "name": names.get(b["tool_use_id"], ""),
                            "content": body,
                        }
                    )
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def _parse_arguments(raw: str | None) -> dict:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_unparseable_arguments": raw}
    return args if isinstance(args, dict) else {"_unparseable_arguments": raw}


class GroqAdapter:
    """Wrap a Groq / OpenAI-style client so it answers in Anthropic message format."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.messages = self

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        tools: list[dict],
        tool_choice: dict,
        messages: list[dict],
    ) -> SimpleNamespace:
        resp = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=_to_openai_messages(system, messages),
            tools=_to_openai_tools(tools),
            tool_choice=tool_choice.get("type", "auto"),
        )
        msg = resp.choices[0].message
        blocks = []
        if msg.content:
            blocks.append(SimpleNamespace(type="text", text=msg.content))
        for call in msg.tool_calls or []:
            blocks.append(
                SimpleNamespace(
                    type="tool_use",
                    id=call.id,
                    name=call.function.name,
                    input=_parse_arguments(call.function.arguments),
                )
            )
        stop = "tool_use" if msg.tool_calls else "end_turn"
        return SimpleNamespace(content=blocks, stop_reason=stop)


def make_llm() -> LLM | None:
    """Build the configured backend, or None if no API key is available."""
    provider = os.environ.get("GRIDFAULT_LLM_PROVIDER")
    if provider is None:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            return None
    model = os.environ.get("GRIDFAULT_LLM_MODEL", DEFAULT_MODELS[provider])
    if provider == "anthropic":
        import anthropic

        return LLM(anthropic.Anthropic(), model, provider)
    if provider == "groq":
        from groq import Groq

        return LLM(GroqAdapter(Groq()), model, provider)
    raise ValueError(f"unknown GRIDFAULT_LLM_PROVIDER {provider!r}")

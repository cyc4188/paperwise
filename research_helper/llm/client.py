from __future__ import annotations
import json
from research_helper import config

_BASE_URLS = {
    "anthropic": "",
    "openai": "",
    "deepseek": "https://api.deepseek.com",
    "qwen":     "https://dashscope.aliyuncs.com/compatible-mode/v1",
}

_DEFAULT_HEADERS = {
    "User-Agent": "Zed/0.211.6 (macos; x86_64)",
}

_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "deepseek":  "deepseek-chat",
    "qwen":      "qwen-plus",
}


class LLMRequestError(RuntimeError):
    """User-friendly error surfaced when an upstream LLM request fails."""


def _friendly_network_error(provider: str, model: str, exc: Exception) -> LLMRequestError:
    provider_label = provider.capitalize()
    hint = (
        f"{provider_label} request failed for model {model}. "
        "Please check your network, proxy, and API endpoint configuration."
    )
    if provider == "deepseek":
        hint += " If you are using a local proxy, make sure it is running and can reach api.deepseek.com."
    return LLMRequestError(f"{hint}\n\nOriginal error: {exc}")


def _record_openai_usage(resp, model: str) -> None:
    from research_helper.utils import cost_tracker

    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if not usage:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")

    if prompt_tokens is not None and completion_tokens is not None:
        cost_tracker.record_llm(model, int(prompt_tokens), int(completion_tokens))


def _extract_sse_text(text: str) -> str:
    chunks: list[str] = []
    found_sse = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        found_sse = True
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = data.get("choices") or []
        if not choices:
            continue
        choice = choices[0] or {}

        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunks.append(item.get("text", ""))
                continue

        message = choice.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                chunks.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunks.append(item.get("text", ""))

    if found_sse:
        return "".join(chunks)
    return text


def _extract_openai_text(resp) -> str:
    if isinstance(resp, str):
        return _extract_sse_text(resp)

    choices = getattr(resp, "choices", None)
    if choices is None and isinstance(resp, dict):
        choices = resp.get("choices")
    if not choices:
        raise LLMRequestError(
            f"LLM response did not contain choices. Response type: {type(resp).__name__}"
        )

    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")

    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)
    if content is None:
        raise LLMRequestError(
            f"LLM response did not contain message content. Response type: {type(resp).__name__}"
        )
    return str(content)


def complete(system: str, user: str, max_tokens: int = 4096) -> str:
    provider = config.LLM_PROVIDER
    model = config.LLM_MODEL or _DEFAULT_MODELS.get(provider, "")

    if provider == "anthropic":
        return _anthropic(system, user, model, max_tokens)
    elif provider in ("openai", "deepseek", "qwen"):
        return _openai_compat(system, user, model, max_tokens, provider)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: {provider!r}. "
            "Choose from: anthropic, openai, deepseek, qwen"
        )


def _anthropic(system: str, user: str, model: str, max_tokens: int) -> str:
    import anthropic
    from research_helper.utils import cost_tracker

    kwargs = {"api_key": config.ANTHROPIC_API_KEY}
    if config.ANTHROPIC_BASE_URL:
        kwargs["base_url"] = config.ANTHROPIC_BASE_URL
    kwargs["default_headers"] = _DEFAULT_HEADERS
    client = anthropic.Anthropic(**kwargs)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:
        name = exc.__class__.__name__
        if "Timeout" in name or "Connection" in name or "Connect" in name or "Network" in name:
            raise _friendly_network_error("anthropic", model, exc) from exc
        raise
    cost_tracker.record_llm(model, msg.usage.input_tokens, msg.usage.output_tokens)
    return msg.content[0].text


def _openai_compat(system: str, user: str, model: str, max_tokens: int, provider: str) -> str:
    from openai import OpenAI

    api_key = _api_key(provider)
    base_url = _base_url(provider)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=_DEFAULT_HEADERS,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
    except Exception as exc:
        name = exc.__class__.__name__
        if "Timeout" in name or "Connection" in name or "Connect" in name or "Network" in name:
            raise _friendly_network_error(provider, model, exc) from exc
        raise
    _record_openai_usage(resp, model)
    return _extract_openai_text(resp)


def _api_key(provider: str) -> str:
    keys = {
        "openai":   config.OPENAI_API_KEY,
        "deepseek": config.DEEPSEEK_API_KEY,
        "qwen":     config.QWEN_API_KEY,
    }
    key = keys.get(provider, "")
    if not key:
        raise EnvironmentError(
            f"{provider.upper()}_API_KEY is not set. Add it to config.toml or the environment."
        )
    return key


def _base_url(provider: str) -> str | None:
    configured = {
        "anthropic": config.ANTHROPIC_BASE_URL,
        "openai": config.OPENAI_BASE_URL,
        "deepseek": config.DEEPSEEK_BASE_URL,
        "qwen": config.QWEN_BASE_URL,
    }
    return configured.get(provider) or _BASE_URLS.get(provider)

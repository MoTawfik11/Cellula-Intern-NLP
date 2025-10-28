import os
from typing import List, Dict, Any, Optional

from openai import OpenAI

DEFAULT_SYSTEM = (
    "You are a helpful Python code assistant. Prefer correct, concise solutions. "
    "When generating code, include a brief explanation after the code."
)


def _provider_client() -> tuple[OpenAI, str]:
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    openai_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    or_key: Optional[str] = os.environ.get("OPENROUTER_API_KEY")
    groq_key: Optional[str] = os.environ.get("GROQ_API_KEY")

    # 1) If base_url explicitly targets a provider and corresponding key exists, prefer it
    if base_url:
        lower = base_url.lower()
        if "openrouter" in lower and or_key:
            return OpenAI(api_key=or_key, base_url=base_url), base_url
        if "groq" in lower and groq_key:
            return OpenAI(api_key=groq_key, base_url=base_url), base_url
        # If base_url is set but no matching provider key, try OPENAI_API_KEY as a last resort
        if openai_key:
            return OpenAI(api_key=openai_key, base_url=base_url), base_url

    # 2) No decisive base_url → prefer provider keys in this order: OpenRouter, Groq, then OpenAI
    if or_key:
        return OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1"), "https://openrouter.ai/api/v1"
    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), "https://api.groq.com/openai/v1"
    if openai_key:
        return OpenAI(api_key=openai_key), "https://api.openai.com/v1"

    # 3) No key at all
    return (None, base_url or "")  # type: ignore


def call_llm(messages: List[Dict[str, str]], model: Optional[str] = None, system: Optional[str] = None) -> str:
    client, provider_url = _provider_client()
    if client is None:
        return (
            "No API key detected. To use a free-friendly provider, set one of:\n"
            "- OPENROUTER_API_KEY and OPENAI_BASE_URL=https://openrouter.ai/api/v1\n"
            "- GROQ_API_KEY and OPENAI_BASE_URL=https://api.groq.com/openai/v1\n"
            "Or set OPENAI_API_KEY for OpenAI. Then set MODEL env variable."
        )

    model_name = model or os.environ.get("MODEL") or _default_model_for(provider_url)

    sys_prompt = system or DEFAULT_SYSTEM
    full_messages = ([{"role": "system", "content": sys_prompt}] + messages)

    resp = client.chat.completions.create(
        model=model_name,
        messages=full_messages,
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    return text


def _default_model_for(provider_url: str) -> str:
    lower = (provider_url or "").lower()
    if "openrouter" in lower:
        return "meta-llama/llama-3.1-70b-instruct"
    if "groq" in lower:
        return "llama-3.1-70b-versatile"
    return "gpt-4o-mini"

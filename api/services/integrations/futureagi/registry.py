from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import httpx
from loguru import logger

from .client import FutureAGIAuthError

# Future AGI's eval catalogue. Returns every eval registered for the account,
# including custom ones, with the exact input keys each accepts.
_REGISTRY_PATH = "sdk/api/v1/get-evals/"

# Aliases from a key we can supply to the canonical key an eval asks for,
# mirroring the vendor SDK's own table (fi/evals/core/cloud_registry.py).
# Only the entries Dograh can actually satisfy from a voice call are kept.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "output": ("output", "response", "answer", "generated", "input"),
    "input": ("input", "query", "question", "prompt_input", "output"),
    "context": ("context", "contexts"),
    "text": ("text", "output", "content"),
    "conversation": ("conversation", "messages", "transcript"),
    "prompt": ("prompt", "instructions", "system_prompt", "context"),
    "system_prompt": ("system_prompt", "prompt", "instructions", "context"),
    "instruction": ("instruction", "prompt", "system_prompt"),
    "instructions": ("instructions", "prompt", "system_prompt"),
    "hypothesis": ("hypothesis", "output", "response"),
    "generated_transcript": ("generated_transcript", "transcript", "output"),
    # The whole customer_agent_* / intake / misselling family asks for the
    # agent's own instructions under this name. That is our system prompt.
    "agent_prompt": ("agent_prompt", "system_prompt", "prompt", "context"),
    # Audio evals take a URL, not bytes: Future AGI fetches the media and
    # sends it as a content part. Dograh supplies the call recording URL.
    "input_audio": ("input_audio", "audio", "audio_url", "recording_url"),
    "audio": ("audio", "input_audio", "audio_url", "recording_url"),
    "generated_audio": ("generated_audio", "audio", "input_audio", "audio_url"),
}

# Cached catalogues, keyed by (base_url, credential fingerprint) so one
# organization never reads another's custom evals. The key is hashed rather
# than stored, matching dataset.py.
_REGISTRY_CACHE: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
_CACHE_LIMIT = 64
_LOAD_LOCK = asyncio.Lock()


def _cache_key(base_url: str, api_key: str) -> tuple[str, str]:
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return (base_url.rstrip("/"), fingerprint)


def clear_registry_cache() -> None:
    """Drop cached catalogues. Used by tests and after an auth change."""
    _REGISTRY_CACHE.clear()


async def load_registry(
    *,
    api_key: str,
    secret_key: str,
    base_url: str,
    timeout_seconds: int = 60,
) -> dict[str, dict[str, Any]]:
    """
    Fetch the account's eval catalogue as ``{eval_name: info}``.

    Returns an empty dict when the catalogue cannot be read. Callers treat
    that as "unknown" and fall back to sending inputs unfiltered, so a
    registry outage degrades to the previous behaviour instead of blocking
    every evaluation.
    """
    base_url = base_url.rstrip("/")
    key = _cache_key(base_url, api_key)

    cached = _REGISTRY_CACHE.get(key)
    if cached is not None:
        return cached

    async with _LOAD_LOCK:
        cached = _REGISTRY_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.get(
                    f"{base_url}/{_REGISTRY_PATH}",
                    headers={"X-Api-Key": api_key, "X-Secret-Key": secret_key},
                )
        except Exception as exc:
            logger.warning(f"[futureagi] could not load eval catalogue: {exc}")
            return {}

        if response.status_code in (401, 403):
            raise FutureAGIAuthError(
                f"Future AGI rejected the API key/secret pair "
                f"(HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            logger.warning(
                f"[futureagi] eval catalogue returned {response.status_code}: "
                f"{response.text[:200]}"
            )
            return {}

        try:
            items = (response.json() or {}).get("result") or []
        except ValueError:
            logger.warning("[futureagi] eval catalogue returned non-JSON")
            return {}

        by_name = {
            item["name"]: item
            for item in items
            if isinstance(item, dict) and item.get("name")
        }

        if len(_REGISTRY_CACHE) >= _CACHE_LIMIT:
            _REGISTRY_CACHE.clear()
        _REGISTRY_CACHE[key] = by_name
        logger.info(f"[futureagi] loaded {len(by_name)} eval template(s)")
        return by_name


def required_keys_for(registry: dict[str, dict[str, Any]], eval_name: str) -> list[str]:
    """The input keys this eval accepts, or [] when it is not in the catalogue."""
    info = registry.get(eval_name)
    if not isinstance(info, dict):
        return []
    config = info.get("config")
    if not isinstance(config, dict):
        return []
    keys = config.get("required_keys")
    return [k for k in keys if isinstance(k, str)] if isinstance(keys, list) else []


def map_inputs(
    registry: dict[str, dict[str, Any]],
    eval_name: str,
    available: dict[str, str],
) -> dict[str, str]:
    """
    Narrow ``available`` to exactly the keys this eval accepts.

    The API rejects supersets: sending ``{input, output, context}`` to a
    template that only declares ``{output}`` is a 400, not a tolerated extra.
    Each required key is taken literally when present, otherwise resolved
    through the alias table.

    An eval missing from the catalogue gets the inputs unfiltered, so the API
    can return its own validation error rather than us silently sending {}.
    """
    required = required_keys_for(registry, eval_name)
    if not required:
        return dict(available)

    mapped: dict[str, str] = {}
    for key in required:
        if key in available:
            mapped[key] = available[key]
            continue
        for alias in _KEY_ALIASES.get(key, ()):
            if alias != key and alias in available:
                mapped[key] = available[alias]
                break
    return mapped


def unmet_keys(
    registry: dict[str, dict[str, Any]],
    eval_name: str,
    available: dict[str, str],
) -> list[str]:
    """
    Required keys this call cannot supply.

    Some evals need inputs a transcript simply does not contain: ground truth
    (``expected``), reference translations, or audio when the run produced no
    recording. Sending those is a guaranteed 400, so the caller reports them
    up front instead of paying for a failed evaluation.
    """
    required = required_keys_for(registry, eval_name)
    if not required:
        return []
    mapped = map_inputs(registry, eval_name, available)
    return [key for key in required if key not in mapped]

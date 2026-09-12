"""List the Future AGI evaluations your account has, and the inputs each needs.

Future AGI's API rejects any input key an evaluation does not declare, so a
name that looks right can still 400 if it wants different keys than Dograh is
sending. This prints the catalogue the Dograh integration reads, so you can
check a node's Evaluations field against what actually exists.

    FI_API_KEY=... FI_SECRET_KEY=... python -m scripts.futureagi_evals

Pass eval names to check just those:

    FI_API_KEY=... FI_SECRET_KEY=... python -m scripts.futureagi_evals tone toxicity
"""

from __future__ import annotations

import asyncio
import os
import sys

import httpx

BASE_URL = os.environ.get("FI_BASE_URL", "https://api.futureagi.com")

# What the Dograh integration can supply for a call. Mirrors
# api/services/integrations/futureagi/collector.py:build_eval_inputs.
DOGRAH_SUPPLIES = {
    "input",
    "query",
    "question",
    "output",
    "response",
    "answer",
    "context",
    "prompt",
    "system_prompt",
    "instructions",
    "text",
    "content",
    "conversation",
    "messages",
    "transcript",
}


async def main() -> int:
    api_key = os.environ.get("FI_API_KEY")
    secret_key = os.environ.get("FI_SECRET_KEY")
    if not api_key or not secret_key:
        print("Set FI_API_KEY and FI_SECRET_KEY first.")
        return 2

    wanted = {name.lower() for name in sys.argv[1:]}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            f"{BASE_URL.rstrip('/')}/sdk/api/v1/get-evals/",
            headers={"X-Api-Key": api_key, "X-Secret-Key": secret_key},
        )

    if response.status_code in (401, 403):
        print(f"Credentials rejected (HTTP {response.status_code}).")
        return 1
    if response.status_code >= 400:
        print(f"HTTP {response.status_code}: {response.text[:400]}")
        return 1

    items = (response.json() or {}).get("result") or []
    catalogue = {
        item["name"]: item
        for item in items
        if isinstance(item, dict) and item.get("name")
    }
    print(f"{len(catalogue)} evaluation(s) available on this account\n")

    if wanted:
        missing = wanted - {name.lower() for name in catalogue}
        for name in sorted(missing):
            print(f"  MISSING   {name}  <- not in this account, would 400")

    print(f"{'eval name':<42} {'required inputs':<40} dograh can supply?")
    print("-" * 100)
    for name in sorted(catalogue):
        if wanted and name.lower() not in wanted:
            continue
        config = catalogue[name].get("config") or {}
        keys = config.get("required_keys") or []
        keys = [k for k in keys if isinstance(k, str)]
        unmet = [k for k in keys if k not in DOGRAH_SUPPLIES]
        verdict = "yes" if not unmet else f"NO - needs {', '.join(unmet)}"
        print(f"{name[:41]:<42} {', '.join(keys)[:39]:<40} {verdict}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

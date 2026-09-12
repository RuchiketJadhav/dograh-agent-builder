from __future__ import annotations

from typing import Any

import httpx
from loguru import logger

DEFAULT_FUTUREAGI_BASE_URL = "https://api.futureagi.com"

# Future AGI's evaluation endpoint. Taken from the ai-evaluation SDK's own
# route table (fi/utils/routes.py, ``Routes.evaluatev2``), which is what its
# Evaluator.evaluate() posts to.
_EVAL_PATH = "sdk/api/v1/new-eval/"


# Dograh talks to this API over httpx rather than depending on Future AGI's
# ``ai-evaluation`` SDK. Three reasons, each independently sufficient:
#
#   1. Dependency conflict. ai-evaluation 1.1.0 pins ``openai<2``; Dograh runs
#      openai 2.54.0. Installing it downgrades openai by a major version
#      underneath the LLM services the pipeline depends on.
#   2. It does not install. ai-evaluation pulls ``litellm>=1.77.4``, whose
#      metadata generation fails on this Python — a heavyweight transitive
#      dependency an eval export has no need for.
#   3. Cross-tenant credential race. ``fi.api.auth.APIKeyAuth`` assigns
#      credentials to ``self.__class__``, not the instance, so constructing a
#      second client with different BYOK keys silently rebinds the first one's
#      credentials process-wide. Two organizations completing calls in the same
#      arq worker would evaluate under each other's keys.
#
# The wire contract below is small and stable enough to own directly, and
# headers are passed per request, so nothing is shared between tenants.


class FutureAGIAuthError(Exception):
    """Credentials were rejected (HTTP 401 or 403)."""


class FutureAGIAPIError(Exception):
    """
    A non-auth error response, carrying the API's own explanation.

    httpx's ``raise_for_status`` message is just the status line and a link to
    MDN, which tells an operator nothing about *why* a request was rejected.
    Future AGI puts the actual reason in the body, so it is surfaced here and
    ends up in the workflow-run annotation.
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body}")


def _extract_evaluations(body: Any) -> list[dict[str, Any]]:
    """
    Pull the evaluation entries out of a response body.

    The documented shape is::

        {"result": [{"evaluations": [{"name", "output", "reason", ...}]}]}

    but the async and error-localization paths return the evaluation wrapped
    directly in ``result`` instead. Both are accepted, and anything else yields
    an empty list rather than raising — a shape change upstream should cost a
    metric, not a completion job.
    """
    if not isinstance(body, dict):
        return []

    entries: list[dict[str, Any]] = []
    for result in body.get("result") or []:
        if not isinstance(result, dict):
            continue
        if isinstance(result.get("evaluations"), list):
            entries.extend(e for e in result["evaluations"] if isinstance(e, dict))
        else:
            entries.append(result)
    return entries


def parse_eval_response(body: Any) -> dict[str, Any]:
    """Flatten one evaluation response into a JSON-safe ``{name: value}`` dict."""
    metrics: dict[str, Any] = {}

    for evaluation in _extract_evaluations(body):
        name = evaluation.get("name") or "value"
        # ``output`` is the score under the revamped backend; ``value`` is the
        # older field name and still turns up on some paths.
        value = evaluation.get("output", evaluation.get("value"))
        # Depending on the eval's output type this is a number, a string, or a
        # list of deterministic choices — all JSON-safe. Anything else is
        # stringified so it survives the JSON column it is written into.
        if not isinstance(value, (int, float, bool, str, list, dict, type(None))):
            value = str(value)
        metrics[str(name)] = value

        reason = evaluation.get("reason")
        if reason:
            metrics.setdefault("reason", str(reason)[:2000])
        output_type = evaluation.get("output_type")
        if output_type:
            metrics.setdefault("output_type", str(output_type))

    return metrics


async def run_evaluation(
    *,
    api_key: str,
    secret_key: str,
    base_url: str,
    eval_name: str,
    inputs: dict[str, str],
    model_name: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """
    Run one evaluation template against one call and return its metrics.

    One request per template: Future AGI's own SDK accepts a list for
    ``eval_name`` but evaluates only the first element, silently discarding the
    rest, so batching here would quietly lose evaluations.

    Raises ``FutureAGIAuthError`` on 403 and ``httpx.HTTPStatusError`` on other
    failures; the caller records those per template.
    """
    url = f"{base_url.rstrip('/')}/{_EVAL_PATH}"
    headers = {
        "X-Api-Key": api_key,
        "X-Secret-Key": secret_key,
    }
    payload: dict[str, Any] = {
        "eval_name": eval_name,
        "inputs": inputs,
        "is_async": False,
    }
    if model_name:
        payload["model"] = model_name

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(url, json=payload, headers=headers)

    # Both codes are treated as auth failures. The API actually answers bad
    # credentials with 401 ("authentication_failed"), while Future AGI's own
    # SDK only special-cases 403 — so the SDK reports a wrong key as a generic
    # error. Verified against the live endpoint on 2026-09-12.
    if response.status_code in (401, 403):
        raise FutureAGIAuthError(
            f"Future AGI rejected the API key/secret pair (HTTP {response.status_code})"
        )

    if response.status_code >= 400:
        body = response.text[:600]
        logger.error(
            f"[futureagi] eval '{eval_name}' failed with status "
            f"{response.status_code}: {body}"
        )
        raise FutureAGIAPIError(response.status_code, body)

    return parse_eval_response(response.json())

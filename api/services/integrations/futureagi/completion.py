from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
from loguru import logger

from api.constants import BACKEND_API_ENDPOINT
from api.services.integrations.base import IntegrationCompletionContext

from .client import FutureAGIAuthError, run_evaluation
from .collector import (
    FUTUREAGI_PAYLOAD_LOG_KEY,
    FUTUREAGI_SNAPSHOT_SCHEMA_VERSION,
    build_eval_inputs,
)
from .dataset import append_call_row, ensure_dataset
from .node import FutureAGINodeData, parse_eval_templates
from .registry import load_registry, map_inputs, unmet_keys

# Concurrent in-flight evaluations per node. Each one is a billed API call, so
# this bounds both connection count and spend spike per completed call.
_EVAL_CONCURRENCY = 4


async def _run_one_eval(
    node_config: FutureAGINodeData,
    template: str,
    available: dict[str, str],
    registry: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict[str, Any]]:
    """Run a single evaluation template and return its named outcome.

    Never raises: one template failing (a typo'd name, a validation error on
    its required keys, a provider outage) must not discard the templates that
    did succeed, since they have already been billed.
    """
    # The API rejects any key the template does not declare, so the inputs are
    # narrowed to that template's own key set before sending.
    inputs = map_inputs(registry, template, available)

    if registry and template not in registry:
        # The catalogue loaded and this name is not in it. Reporting that
        # directly beats letting the API answer with an opaque 400.
        known = ", ".join(sorted(registry)[:8])
        logger.warning(
            f"Future AGI eval '{template}' is not in this account's catalogue "
            f"(known: {known}{'...' if len(registry) > 8 else ''})"
        )
        return template, {"error": f"unknown_eval: '{template}' not found in account"}

    missing = unmet_keys(registry, template, available)
    if missing:
        # A transcript cannot supply ground truth, a reference translation, or
        # audio for a run with no recording. Sending anyway is a guaranteed
        # 400 on a billed endpoint, so it is reported without calling out.
        logger.info(
            f"Future AGI eval '{template}' needs {', '.join(missing)}, which "
            "this call cannot supply — skipping"
        )
        return template, {"error": f"missing_input: {', '.join(missing)}"}

    async with semaphore:
        try:
            metrics = await run_evaluation(
                api_key=node_config.futureagi_api_key or "",
                secret_key=node_config.futureagi_secret_key or "",
                base_url=node_config.futureagi_base_url,
                eval_name=template,
                inputs=inputs,
                model_name=node_config.futureagi_model_name,
                timeout_seconds=node_config.futureagi_timeout_seconds,
            )
        except FutureAGIAuthError as exc:
            # Surfaced distinctly: every template on this node will fail the
            # same way, and the fix is credentials, not the eval name.
            logger.warning(f"Future AGI eval '{template}': {exc}")
            return template, {"error": "invalid_credentials"}
        except httpx.TimeoutException:
            logger.warning(f"Future AGI eval '{template}' timed out")
            return template, {"error": "timeout"}
        except Exception as exc:
            logger.warning(f"Future AGI eval '{template}' failed: {exc}")
            return template, {"error": str(exc)[:500]}

        if not metrics:
            return template, {"error": "no_metrics_returned"}
        return template, {"metrics": metrics}


def _build_recording_url(context: IntegrationCompletionContext) -> str | None:
    """Public URL for the call recording, for the audio-level evals.

    Future AGI fetches media by URL rather than accepting an upload, so this
    has to be reachable from their servers. On a local install pointing at
    localhost it will not be, and those evals will fail on their side — which
    is why audio evals are reported separately rather than silently skipped.
    """
    workflow_run = context.workflow_run
    if context.public_token:
        base_url = (
            f"{BACKEND_API_ENDPOINT}/api/v1/public/download/workflow/"
            f"{context.public_token}"
        )
        return f"{base_url}/recording" if workflow_run.recording_url else None
    return workflow_run.recording_url


def _flatten_scores(evaluations: dict[str, dict[str, Any]]) -> str:
    """Render the eval outcomes as one compact cell value."""
    parts: list[str] = []
    for template, outcome in evaluations.items():
        metrics = outcome.get("metrics")
        if not metrics:
            parts.append(f"{template}=error:{outcome.get('error', 'unknown')}")
            continue
        scored = ", ".join(
            f"{k}={v}" for k, v in metrics.items() if k not in ("reason", "output_type")
        )
        parts.append(f"{template}=[{scored}]" if scored else f"{template}=[]")
    return "; ".join(parts)


async def _file_dataset_row(
    fi_data: FutureAGINodeData,
    snapshot: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """
    File this call as a row in the configured Future AGI dataset.

    Optional and strictly secondary to the evaluations: the scores are already
    recorded on the workflow run by the time this runs, so a dataset failure is
    reported but never downgrades an otherwise successful evaluation.
    """
    dataset_name = (fi_data.futureagi_dataset_name or "").strip()
    if not dataset_name:
        return None

    attributes = snapshot.get("attributes") or {}
    try:
        dataset_id = await ensure_dataset(
            api_key=fi_data.futureagi_api_key or "",
            secret_key=fi_data.futureagi_secret_key or "",
            base_url=fi_data.futureagi_base_url,
            dataset_name=dataset_name,
            timeout_seconds=fi_data.futureagi_timeout_seconds,
        )
        await append_call_row(
            api_key=fi_data.futureagi_api_key or "",
            secret_key=fi_data.futureagi_secret_key or "",
            base_url=fi_data.futureagi_base_url,
            dataset_id=dataset_id,
            timeout_seconds=fi_data.futureagi_timeout_seconds,
            values={
                "workflow_run_id": attributes.get("dograh.workflow_run_id"),
                "agent_version": attributes.get("dograh.agent_version"),
                "call_disposition": attributes.get("dograh.call_disposition"),
                "input": snapshot.get("input"),
                "output": snapshot.get("output"),
                "transcript": snapshot.get("transcript"),
                "context": snapshot.get("context"),
                "eval_scores": _flatten_scores(evaluations),
            },
        )
    except FutureAGIAuthError:
        logger.warning(f"[futureagi] dataset '{dataset_name}': invalid credentials")
        return {"status": "failed", "error": "invalid_credentials"}
    except Exception as exc:
        logger.warning(f"[futureagi] dataset '{dataset_name}' row failed: {exc}")
        return {"status": "failed", "error": str(exc)[:300]}

    logger.info(f"[futureagi] filed call row in dataset '{dataset_name}'")
    return {
        "status": "row_added",
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
    }


async def run_completion(
    nodes: list[dict[str, Any]],
    context: IntegrationCompletionContext,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    logs = context.workflow_run.logs
    snapshot = logs.get(FUTUREAGI_PAYLOAD_LOG_KEY) if isinstance(logs, dict) else None
    # A persisted snapshot that is present but not a dict (corrupt/tampered
    # log) would crash every later ``snapshot.get(...)`` outside the per-node
    # try/except, losing every node's truthful result. Flag it so the loop
    # records a bounded per-node error instead of raising.
    snapshot_malformed = snapshot is not None and not isinstance(snapshot, dict)
    recording_url = _build_recording_url(context)

    for node in nodes:
        node_id = node.get("id", "unknown")
        try:
            fi_data = FutureAGINodeData.model_validate(node.get("data", {}))
        except Exception as exc:
            logger.warning(
                f"Future AGI node #{node_id} failed validation, skipping: {exc}"
            )
            results[f"futureagi_{node_id}"] = {"error": "validation_failed"}
            continue

        if not fi_data.futureagi_enabled:
            logger.debug(f"Future AGI node '{fi_data.name}' is disabled, skipping")
            continue

        if snapshot_malformed:
            logger.warning(
                f"Future AGI payload for node '{fi_data.name}' (#{node_id}) is "
                f"malformed (type {type(snapshot).__name__}), skipping evaluation"
            )
            results[f"futureagi_{node_id}"] = {"error": "malformed_payload"}
            continue

        if not snapshot:
            # Not an error: runs that never start a pipeline (e.g. telephony
            # failures) legitimately produce no conversation.
            logger.info(
                f"No Future AGI payload for node '{fi_data.name}' (#{node_id}) — "
                "run produced no transcript"
            )
            results[f"futureagi_{node_id}"] = {"status": "no_transcript"}
            continue

        schema_version = snapshot.get("schema_version")
        if schema_version != FUTUREAGI_SNAPSHOT_SCHEMA_VERSION:
            logger.warning(
                f"Future AGI snapshot schema_version {schema_version} is not "
                f"{FUTUREAGI_SNAPSHOT_SCHEMA_VERSION}, skipping node #{node_id}"
            )
            results[f"futureagi_{node_id}"] = {"error": "unsupported_schema_version"}
            continue

        templates = parse_eval_templates(fi_data.futureagi_eval_templates)
        available = build_eval_inputs(snapshot)

        # Audio-level evals (dead_air_detection, audio_quality, ASR accuracy)
        # take a media URL. Only offered when the run actually produced a
        # recording — an empty string here would turn a clear "we have no
        # audio" into an opaque API error.
        if recording_url:
            available["recording_url"] = recording_url
            available["audio_url"] = recording_url
            available["audio"] = recording_url
            available["input_audio"] = recording_url
            available["generated_audio"] = recording_url

        # Loaded once per node and shared across its templates. Tells us which
        # input keys each eval accepts; the API 400s on anything else.
        try:
            registry = await load_registry(
                api_key=fi_data.futureagi_api_key or "",
                secret_key=fi_data.futureagi_secret_key or "",
                base_url=fi_data.futureagi_base_url,
                timeout_seconds=fi_data.futureagi_timeout_seconds,
            )
        except FutureAGIAuthError:
            logger.warning(
                f"Future AGI node '{fi_data.name}': credentials rejected loading "
                "the eval catalogue"
            )
            results[f"futureagi_{node_id}"] = {"error": "invalid_credentials"}
            continue

        semaphore = asyncio.Semaphore(_EVAL_CONCURRENCY)
        # One request per template: Future AGI evaluates only the first entry
        # when handed a list, silently dropping the rest.
        outcomes = await asyncio.gather(
            *(
                _run_one_eval(fi_data, template, available, registry, semaphore)
                for template in templates
            )
        )

        evaluations = {template: outcome for template, outcome in outcomes}
        succeeded = sum(1 for outcome in evaluations.values() if "metrics" in outcome)

        dataset_result = await _file_dataset_row(fi_data, snapshot, evaluations)

        results[f"futureagi_{node_id}"] = {
            "status": "evaluated" if succeeded else "all_evals_failed",
            "evaluations": evaluations,
            "evals_succeeded": succeeded,
            "evals_failed": len(evaluations) - succeeded,
            "turns_evaluated": len(snapshot.get("turns") or []),
            "transcript_truncated": bool(snapshot.get("truncated")),
            "evaluated_at": datetime.now(UTC).isoformat(),
            **({"dataset": dataset_result} if dataset_result else {}),
        }
        logger.info(
            f"Future AGI node '{fi_data.name}': {succeeded}/{len(evaluations)} "
            f"evaluation(s) succeeded"
        )

    return results

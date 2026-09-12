from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

from api.services.integrations.base import (
    IntegrationRuntimeContext,
    IntegrationRuntimeSession,
)

from .collector import FUTUREAGI_PAYLOAD_LOG_KEY, build_transcript_snapshot


def _format_model_label(provider: Any, model: Any) -> str:
    # provider/model may be enum members (e.g. ServiceProviders.DOGRAH) —
    # label with the wire value, not the enum repr.
    provider = getattr(provider, "value", provider)
    model = getattr(model, "value", model)
    if provider and model:
        return f"{provider}/{model}"
    if model:
        return model
    return provider or ""


def _resolve_model_labels(context: IntegrationRuntimeContext) -> dict[str, str]:
    user_config = context.user_config

    if context.is_realtime and user_config.realtime:
        llm_model = _format_model_label(
            user_config.realtime.provider, user_config.realtime.model
        )
        return {"stt": "", "llm": llm_model, "tts": ""}

    return {
        "stt": _format_model_label(
            getattr(user_config.stt, "provider", None),
            getattr(user_config.stt, "model", None),
        ),
        "llm": _format_model_label(
            getattr(user_config.llm, "provider", None),
            getattr(user_config.llm, "model", None),
        ),
        "tts": _format_model_label(
            getattr(user_config.tts, "provider", None),
            getattr(user_config.tts, "model", None),
        ),
    }


class FutureAGIRuntimeSession(IntegrationRuntimeSession):
    name = "futureagi"

    def __init__(
        self,
        messages_provider: Callable[[], list[dict[str, Any]]],
        call_attributes: dict[str, Any],
    ) -> None:
        self._messages_provider = messages_provider
        self._call_attributes = call_attributes

    def attach(self, task: Any) -> None:
        """No live wiring.

        Evaluations are post-call by design, so this integration attaches no
        observer and opens no socket while the conversation is running. The
        whole payload is read from the context once the call has ended, which
        keeps it off the latency path entirely.
        """
        return

    async def on_call_finished(
        self,
        *,
        gathered_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        attributes = dict(self._call_attributes)
        disposition = gathered_context.get("call_disposition")
        if disposition:
            attributes["dograh.call_disposition"] = disposition
        mapped = gathered_context.get("mapped_call_disposition")
        if mapped:
            attributes["dograh.mapped_call_disposition"] = mapped

        # Reading the context must never be what fails a call teardown: the
        # provider closes over live pipeline state that may already be tearing
        # down by the time this runs.
        try:
            messages = self._messages_provider()
        except Exception as exc:
            logger.warning(f"Future AGI: could not read conversation context: {exc}")
            return None

        snapshot = build_transcript_snapshot(messages, attributes)
        if snapshot is None:
            logger.info(
                "Future AGI: no gradeable conversation for this call — "
                "nothing to evaluate"
            )
            return None

        logger.info(
            f"Future AGI: captured {len(snapshot['turns'])} turn(s) "
            f"({len(snapshot['transcript'])} chars"
            f"{', truncated' if snapshot['truncated'] else ''}) — "
            "snapshot persisted for completion-phase evaluation"
        )
        return {FUTUREAGI_PAYLOAD_LOG_KEY: snapshot}


def create_runtime_sessions(
    context: IntegrationRuntimeContext,
) -> list[IntegrationRuntimeSession]:
    futureagi_nodes = [
        node
        for node in context.workflow_graph.nodes.values()
        if node.node_type == "futureagi"
        and getattr(node.data, "futureagi_enabled", True)
    ]
    if not futureagi_nodes:
        return []

    models = _resolve_model_labels(context)
    call_attributes: dict[str, Any] = {
        "dograh.workflow_run_id": context.workflow_run_id,
        "dograh.mode": getattr(context.workflow_run, "mode", None),
        "dograh.agent_version": getattr(context.run_definition, "version_number", None),
        "stt.model_label": models["stt"],
        "llm.model_label": models["llm"],
        "tts.model_label": models["tts"],
    }

    logger.info(f"Future AGI integration active for run {context.workflow_run_id}")
    return [FutureAGIRuntimeSession(context.context_messages_provider, call_attributes)]

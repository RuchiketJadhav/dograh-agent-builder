from __future__ import annotations

from typing import Any

# Key under which the transcript snapshot is parked in ``workflow_run.logs``
# for the completion phase to pick up.
FUTUREAGI_PAYLOAD_LOG_KEY = "futureagi_payload"

# Bumped whenever the snapshot shape changes. run_completion refuses a snapshot
# it does not recognise rather than mis-reading an older one: a run that
# started before a deploy can very much finish after it.
FUTUREAGI_SNAPSHOT_SCHEMA_VERSION = 1

# The snapshot lands in a JSON column and is read back by an arq worker. A long
# call, or one with a runaway tool loop, can otherwise park megabytes per run.
_MAX_TURNS = 400
_MAX_TRANSCRIPT_CHARS = 100_000

_USER_ROLES = frozenset({"user"})
_AGENT_ROLES = frozenset({"assistant"})
_CONTEXT_ROLES = frozenset({"system", "developer"})


def _content_to_text(content: Any) -> str:
    """
    Flatten one message's content to plain text.

    Pipecat carries OpenAI-shaped messages, so ``content`` is usually a string,
    but multimodal turns arrive as a list of parts (``{"type": "text", ...}``)
    and tool results can be arbitrary objects. Evals only grade text, so parts
    that carry none are dropped rather than stringified into noise.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return " ".join(parts)
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    return str(content)


def build_transcript_snapshot(
    messages: Any,
    call_attributes: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Build the eval payload from the live conversation context.

    Returns ``None`` when the call produced no gradeable exchange — a run that
    never connected, or one where the agent spoke and nobody answered. That is
    not an error, and the completion phase reports it as ``no_transcript``
    rather than failing the node.

    The snapshot deliberately carries four views of the same conversation,
    because Future AGI's eval templates disagree about what they want: some
    take ``input``/``output`` pairs, conversation-level ones want the whole
    transcript, and groundedness-style ones want the system prompt as context.
    Computing all four once here beats re-deriving them per template later.
    """
    if not isinstance(messages, list) or not messages:
        return None

    turns: list[dict[str, str]] = []
    user_parts: list[str] = []
    agent_parts: list[str] = []
    context_parts: list[str] = []
    truncated = len(messages) > _MAX_TURNS

    for message in messages[:_MAX_TURNS]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not isinstance(role, str):
            continue
        text = _content_to_text(message.get("content")).strip()
        if not text:
            # Tool-call requests carry their payload in `tool_calls`, not
            # `content`. They are real conversation events but nothing an eval
            # template scores, so they are left out of the graded text.
            continue

        if role in _CONTEXT_ROLES:
            context_parts.append(text)
            # System prompts are context, not conversation — keeping them in
            # `turns` would let a conversation-level eval grade our own prompt
            # as if the agent had said it.
            continue

        turns.append({"role": role, "content": text})
        if role in _USER_ROLES:
            user_parts.append(text)
        elif role in _AGENT_ROLES:
            agent_parts.append(text)

    if not turns:
        return None

    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        # Keep the tail: the end of a call carries the outcome, the objection
        # handling and the close, which is what almost every eval is judging.
        transcript = transcript[-_MAX_TRANSCRIPT_CHARS:]
        truncated = True

    return {
        "schema_version": FUTUREAGI_SNAPSHOT_SCHEMA_VERSION,
        "attributes": dict(call_attributes),
        "turns": turns,
        "transcript": transcript,
        "input": "\n".join(user_parts),
        "output": "\n".join(agent_parts),
        "context": "\n".join(context_parts),
        "truncated": truncated,
    }


def build_eval_inputs(snapshot: dict[str, Any]) -> dict[str, str]:
    """
    Everything about this call an eval template might ask for.

    This is a *source* dict, not a payload. Future AGI's API is strict and
    **rejects supersets**: sending ``{input, output, context}`` to a template
    that declares only ``{output}`` is a 400, not a tolerated extra. So this
    gets narrowed per template by ``registry.map_inputs`` before it is sent,
    using the key set the backend itself declares for that eval.

    Keys are deliberately redundant (``input``/``query``, ``output``/
    ``response``) so the alias resolution has something to match whichever
    canonical name a given template happens to use.
    """
    caller = snapshot.get("input", "")
    agent = snapshot.get("output", "")
    transcript = snapshot.get("transcript", "")
    system_prompt = snapshot.get("context", "")

    return {
        "input": caller,
        "query": caller,
        "question": caller,
        "output": agent,
        "response": agent,
        "answer": agent,
        "context": system_prompt,
        "prompt": system_prompt,
        "system_prompt": system_prompt,
        "instructions": system_prompt,
        # The customer_agent_* family names the agent's instructions this way.
        "agent_prompt": system_prompt,
        "text": transcript,
        "content": transcript,
        "conversation": transcript,
        "messages": transcript,
        "transcript": transcript,
    }

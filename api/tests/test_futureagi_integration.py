"""Tests for the Future AGI integration package (api/services/integrations/futureagi).

Covers the AGENTS.md testing minimum for a new integration:
- node spec property order + registry/masking wiring for both sensitive fields
- node validation (required-when-enabled, positive timeout)
- create_runtime_sessions: disabled → [], enabled → one session
- the transcript snapshot built from live pipeline context
- run_completion: validation failure, disabled skip, missing snapshot,
  schema-version guard, and the happy path

The Future AGI API is faked at the package's own seam (run_evaluation), so
these tests make no network calls. Dograh talks to the API over httpx rather
than depending on the ai-evaluation SDK; see client.py for why.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.services.configuration.masking import mask_key, mask_workflow_definition
from api.services.integrations.futureagi.client import (
    FutureAGIAPIError,
    FutureAGIAuthError,
    parse_eval_response,
    run_evaluation,
)
from api.services.integrations.futureagi.collector import (
    FUTUREAGI_PAYLOAD_LOG_KEY,
    FUTUREAGI_SNAPSHOT_SCHEMA_VERSION,
    build_eval_inputs,
    build_transcript_snapshot,
)
from api.services.integrations.futureagi.completion import run_completion
from api.services.integrations.futureagi.dataset import (
    _DATASET_ID_CACHE,
    DATASET_COLUMNS,
    MAX_CELL_CHARS,
    append_call_row,
    ensure_dataset,
    truncate_cell,
)
from api.services.integrations.futureagi.node import (
    NODE,
    FutureAGINodeData,
    parse_eval_templates,
)
from api.services.integrations.futureagi.registry import (
    clear_registry_cache,
    load_registry,
    map_inputs,
    required_keys_for,
    unmet_keys,
)
from api.services.integrations.futureagi.runtime import (
    FutureAGIRuntimeSession,
    create_runtime_sessions,
)
from api.services.workflow.node_specs import all_specs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workflow_def(nodes):
    return {"nodes": nodes, "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}


def _fi_node(node_id="fi-1", api_key="", secret_key="", **extra_data):
    data = {
        "name": "Future AGI",
        "futureagi_enabled": True,
        "futureagi_eval_templates": "Tone",
        **extra_data,
    }
    if api_key:
        data["futureagi_api_key"] = api_key
    if secret_key:
        data["futureagi_secret_key"] = secret_key
    return {
        "id": node_id,
        "type": "futureagi",
        "position": {"x": 0, "y": 0},
        "data": data,
    }


def _graph_node(data: FutureAGINodeData):
    return types.SimpleNamespace(node_type="futureagi", data=data)


def _messages():
    return [
        {"role": "system", "content": "You are a polite scheduling agent."},
        {
            "role": "assistant",
            "content": "Hi, this is Dograh calling about your appointment.",
        },
        {"role": "user", "content": "Sure, what time works?"},
        {"role": "assistant", "content": "How about Tuesday at 3pm?"},
    ]


def _runtime_context(nodes, messages=None):
    return types.SimpleNamespace(
        workflow_run_id=42,
        workflow_run=types.SimpleNamespace(mode="phonecall"),
        workflow_graph=types.SimpleNamespace(
            nodes={f"n{i}": node for i, node in enumerate(nodes)}
        ),
        run_definition=types.SimpleNamespace(version_number=3),
        user_config=types.SimpleNamespace(
            stt=types.SimpleNamespace(provider="deepgram", model="nova-3"),
            llm=types.SimpleNamespace(provider="openai", model="gpt-4o"),
            tts=types.SimpleNamespace(provider="elevenlabs", model="turbo"),
            realtime=None,
        ),
        is_realtime=False,
        context_messages_provider=lambda: _messages() if messages is None else messages,
    )


def _node_data(**overrides):
    data = {
        "name": "Future AGI",
        "futureagi_enabled": True,
        "futureagi_api_key": "fi-key",
        "futureagi_secret_key": "fi-secret",
        "futureagi_eval_templates": "Tone",
        **overrides,
    }
    return FutureAGINodeData.model_validate(data)


def _snapshot(schema_version=FUTUREAGI_SNAPSHOT_SCHEMA_VERSION, **overrides):
    base = {
        "schema_version": schema_version,
        "attributes": {"dograh.workflow_run_id": 42},
        "turns": [{"role": "user", "content": "hello"}],
        "transcript": "user: hello",
        "input": "hello",
        "output": "hi there",
        "context": "system prompt",
        "truncated": False,
    }
    base.update(overrides)
    return base


def _completion_context(logs=None):
    return types.SimpleNamespace(
        workflow_run_id=42,
        workflow_run=types.SimpleNamespace(logs=logs or {}, recording_url=None),
        workflow_definition={},
        definition_id=1,
        organization_id=1,
        public_token=None,
    )


def _wire_body(name="Tone", output="positive", reason=None):
    """A response shaped exactly like Future AGI's /sdk/api/v1/new-eval/."""
    evaluation = {"name": name, "output": output, "output_type": "choice"}
    if reason:
        evaluation["reason"] = reason
    return {"result": [{"evaluations": [evaluation]}]}


# ---------------------------------------------------------------------------
# Node spec + registry wiring
# ---------------------------------------------------------------------------


def test_futureagi_spec_property_order_stable():
    spec = next(spec for spec in all_specs() if spec.name == "futureagi")
    assert [prop.name for prop in spec.properties] == [
        "name",
        "futureagi_enabled",
        "futureagi_api_key",
        "futureagi_secret_key",
        "futureagi_eval_templates",
        "futureagi_dataset_name",
        "futureagi_model_name",
        "futureagi_base_url",
        "futureagi_timeout_seconds",
    ]


def test_futureagi_node_registered_with_both_secrets():
    assert NODE.type_name == "futureagi"
    assert set(NODE.sensitive_fields) == {
        "futureagi_api_key",
        "futureagi_secret_key",
    }


def test_spec_example_validates_against_model():
    spec = next(spec for spec in all_specs() if spec.name == "futureagi")
    for example in spec.examples:
        FutureAGINodeData.model_validate(example.data)


def test_enabled_node_requires_credentials_and_templates():
    for missing in (
        {"futureagi_api_key": None},
        {"futureagi_secret_key": None},
        {"futureagi_eval_templates": None},
        {"futureagi_eval_templates": "  ,  "},
    ):
        try:
            _node_data(**missing)
        except ValueError:
            continue
        raise AssertionError(f"expected validation failure for {missing}")


def test_enabled_node_rejects_non_positive_timeout():
    try:
        _node_data(futureagi_timeout_seconds=0)
    except ValueError:
        return
    raise AssertionError("expected validation failure for zero timeout")


def test_disabled_node_validates_without_credentials():
    data = FutureAGINodeData.model_validate(
        {"name": "Future AGI", "futureagi_enabled": False}
    )
    assert data.futureagi_enabled is False


def test_parse_eval_templates_trims_and_drops_blanks():
    assert parse_eval_templates(" Tone , ,Toxicity ") == ["Tone", "Toxicity"]
    assert parse_eval_templates(None) == []


def test_masks_both_futureagi_secrets():
    real_key = "fi_live_abcdefghijklmnop"
    real_secret = "fisk_live_qrstuvwxyz012345"
    wf = _make_workflow_def([_fi_node(api_key=real_key, secret_key=real_secret)])

    masked = mask_workflow_definition(wf)

    masked_data = masked["nodes"][0]["data"]
    assert masked_data["futureagi_api_key"] == mask_key(real_key)
    assert masked_data["futureagi_secret_key"] == mask_key(real_secret)
    assert real_key not in str(masked)
    assert real_secret not in str(masked)


# ---------------------------------------------------------------------------
# Runtime sessions
# ---------------------------------------------------------------------------


def test_create_runtime_sessions_empty_without_futureagi_node():
    assert create_runtime_sessions(_runtime_context([])) == []


def test_create_runtime_sessions_skips_disabled_node():
    context = _runtime_context([_graph_node(_node_data(futureagi_enabled=False))])
    assert create_runtime_sessions(context) == []


def test_create_runtime_sessions_builds_one_session():
    context = _runtime_context([_graph_node(_node_data())])
    sessions = create_runtime_sessions(context)
    assert len(sessions) == 1
    assert sessions[0].name == "futureagi"


def test_attach_does_no_live_wiring():
    """The integration must stay off the live call path entirely."""
    session = FutureAGIRuntimeSession(list, {})
    task = MagicMock()
    assert session.attach(task) is None
    task.add_observer.assert_not_called()


async def test_on_call_finished_returns_snapshot_with_disposition():
    context = _runtime_context([_graph_node(_node_data())])
    session = create_runtime_sessions(context)[0]

    logs = await session.on_call_finished(
        gathered_context={"call_disposition": "appointment_booked"}
    )

    snapshot = logs[FUTUREAGI_PAYLOAD_LOG_KEY]
    assert snapshot["attributes"]["dograh.call_disposition"] == "appointment_booked"
    assert snapshot["attributes"]["dograh.workflow_run_id"] == 42
    assert snapshot["schema_version"] == FUTUREAGI_SNAPSHOT_SCHEMA_VERSION


async def test_on_call_finished_none_when_no_conversation():
    context = _runtime_context([_graph_node(_node_data())], messages=[])
    session = create_runtime_sessions(context)[0]
    assert await session.on_call_finished(gathered_context={}) is None


async def test_on_call_finished_survives_context_read_failure():
    def _boom():
        raise RuntimeError("pipeline already torn down")

    session = FutureAGIRuntimeSession(_boom, {})
    assert await session.on_call_finished(gathered_context={}) is None


# ---------------------------------------------------------------------------
# Transcript snapshot
# ---------------------------------------------------------------------------


def test_snapshot_splits_roles_and_excludes_system_prompt_from_turns():
    snapshot = build_transcript_snapshot(_messages(), {"dograh.mode": "phonecall"})

    assert [turn["role"] for turn in snapshot["turns"]] == [
        "assistant",
        "user",
        "assistant",
    ]
    assert "polite scheduling agent" in snapshot["context"]
    assert "polite scheduling agent" not in snapshot["transcript"]
    assert snapshot["input"] == "Sure, what time works?"
    assert "Tuesday at 3pm" in snapshot["output"]


def test_snapshot_flattens_multimodal_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "check my"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                {"type": "text", "text": "booking"},
            ],
        }
    ]
    snapshot = build_transcript_snapshot(messages, {})
    assert snapshot["input"] == "check my booking"


def test_snapshot_none_for_empty_or_contentless_conversation():
    assert build_transcript_snapshot([], {}) is None
    assert build_transcript_snapshot(None, {}) is None
    # Tool-call requests carry no `content` — nothing gradeable.
    assert (
        build_transcript_snapshot(
            [{"role": "assistant", "tool_calls": [{"id": "1"}]}], {}
        )
        is None
    )


def test_snapshot_truncates_long_conversation_keeping_the_tail():
    messages = [
        {"role": "user", "content": f"turn {i} " + "x" * 500} for i in range(500)
    ]
    snapshot = build_transcript_snapshot(messages, {})
    assert snapshot["truncated"] is True
    assert len(snapshot["transcript"]) <= 100_000


def test_build_eval_inputs_supplies_every_expected_key():
    inputs = build_eval_inputs(_snapshot())
    assert set(inputs) >= {"input", "output", "context", "query", "response", "text"}
    assert inputs["query"] == inputs["input"]


# ---------------------------------------------------------------------------
# Wire response parsing
# ---------------------------------------------------------------------------


def test_parse_eval_response_reads_documented_shape():
    metrics = parse_eval_response(_wire_body(reason="warm and helpful"))
    assert metrics["Tone"] == "positive"
    assert metrics["reason"] == "warm and helpful"
    assert metrics["output_type"] == "choice"


def test_parse_eval_response_accepts_unwrapped_async_shape():
    """The async / error-localization paths omit the `evaluations` wrapper."""
    body = {"result": [{"name": "Toxicity", "output": 0.02}]}
    assert parse_eval_response(body)["Toxicity"] == 0.02


def test_parse_eval_response_keeps_list_values_and_falls_back_to_value():
    body = {"result": [{"evaluations": [{"name": "Choices", "value": ["a", "b"]}]}]}
    assert parse_eval_response(body)["Choices"] == ["a", "b"]


def test_parse_eval_response_empty_for_garbage():
    assert parse_eval_response(None) == {}
    assert parse_eval_response({"result": ["nonsense", 3]}) == {}
    assert parse_eval_response({}) == {}


# ---------------------------------------------------------------------------
# Completion handler
# ---------------------------------------------------------------------------

_RUN_EVAL = "api.services.integrations.futureagi.completion.run_evaluation"
_ENSURE_DATASET = "api.services.integrations.futureagi.completion.ensure_dataset"
_APPEND_ROW = "api.services.integrations.futureagi.completion.append_call_row"
_LOAD_REGISTRY = "api.services.integrations.futureagi.completion.load_registry"


@pytest.fixture(autouse=True)
def _stub_eval_catalogue(request):
    """Keep completion tests off the network.

    Every run_completion call now loads the account's eval catalogue first. An
    empty catalogue means "unknown", which passes inputs through unfiltered --
    the behaviour these tests were written against. Mark a test with
    @pytest.mark.real_registry to opt out and drive load_registry yourself.
    """
    if "real_registry" in request.keywords:
        yield
        return
    with patch(_LOAD_REGISTRY, new=AsyncMock(return_value={})):
        yield


async def test_completion_records_validation_failure():
    nodes = [{"id": "bad", "data": {"futureagi_enabled": True}}]
    results = await run_completion(nodes, _completion_context())
    assert results["futureagi_bad"] == {"error": "validation_failed"}


async def test_completion_skips_disabled_node():
    nodes = [{"id": "off", "data": {"name": "Future AGI", "futureagi_enabled": False}}]
    results = await run_completion(nodes, _completion_context())
    assert results == {}


async def test_completion_reports_no_transcript_when_snapshot_absent():
    nodes = [_fi_node(api_key="k", secret_key="s")]
    results = await run_completion(nodes, _completion_context(logs={}))
    assert results["futureagi_fi-1"] == {"status": "no_transcript"}


async def test_completion_rejects_unknown_schema_version():
    nodes = [_fi_node(api_key="k", secret_key="s")]
    ctx = _completion_context(
        logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot(schema_version=999)}
    )
    results = await run_completion(nodes, ctx)
    assert results["futureagi_fi-1"] == {"error": "unsupported_schema_version"}


async def test_completion_malformed_snapshot_yields_error_without_raising():
    nodes = [_fi_node(api_key="k", secret_key="s")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: "not-a-dict"})
    results = await run_completion(nodes, ctx)
    assert results["futureagi_fi-1"] == {"error": "malformed_payload"}


async def test_completion_happy_path_runs_one_request_per_template():
    nodes = [
        _fi_node(api_key="k", secret_key="s", futureagi_eval_templates="Tone, Toxicity")
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    calls = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return {kwargs["eval_name"]: "positive"}

    with patch(_RUN_EVAL, new=_fake):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["status"] == "evaluated"
    assert outcome["evals_succeeded"] == 2
    assert outcome["evals_failed"] == 0
    assert set(outcome["evaluations"]) == {"Tone", "Toxicity"}
    assert outcome["turns_evaluated"] == 1

    # One request per template, each carrying this node's own credentials —
    # never a batched list (the API grades only the first entry of one).
    assert [c["eval_name"] for c in calls] == ["Tone", "Toxicity"]
    for call in calls:
        assert isinstance(call["eval_name"], str)
        assert call["api_key"] == "k"
        assert call["secret_key"] == "s"


async def test_completion_one_failing_template_does_not_lose_the_others():
    nodes = [
        _fi_node(api_key="k", secret_key="s", futureagi_eval_templates="Good, Bad")
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake(**kwargs):
        if kwargs["eval_name"] == "Bad":
            raise RuntimeError("no such eval")
        return {"Good": 1.0}

    with patch(_RUN_EVAL, new=_fake):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["evals_succeeded"] == 1
    assert outcome["evals_failed"] == 1
    assert outcome["evaluations"]["Good"]["metrics"] == {"Good": 1.0}
    assert "no such eval" in outcome["evaluations"]["Bad"]["error"]


async def test_completion_reports_invalid_credentials_distinctly():
    nodes = [_fi_node(api_key="k", secret_key="wrong")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake(**kwargs):
        raise FutureAGIAuthError("rejected")

    with patch(_RUN_EVAL, new=_fake):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["status"] == "all_evals_failed"
    assert outcome["evaluations"]["Tone"] == {"error": "invalid_credentials"}


async def test_completion_reports_timeout_as_timeout():
    nodes = [_fi_node(api_key="k", secret_key="s")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake(**kwargs):
        raise httpx.ReadTimeout("too slow")

    with patch(_RUN_EVAL, new=_fake):
        results = await run_completion(nodes, ctx)

    assert results["futureagi_fi-1"]["evaluations"]["Tone"] == {"error": "timeout"}


async def test_completion_empty_metrics_counted_as_failure():
    nodes = [_fi_node(api_key="k", secret_key="s")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake(**kwargs):
        return {}

    with patch(_RUN_EVAL, new=_fake):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["status"] == "all_evals_failed"
    assert outcome["evaluations"]["Tone"] == {"error": "no_metrics_returned"}


async def test_completion_passes_model_name_and_timeout_when_configured():
    nodes = [
        _fi_node(
            api_key="k",
            secret_key="s",
            futureagi_model_name="TURING_FLASH",
            futureagi_timeout_seconds=30,
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    calls = []

    async def _fake(**kwargs):
        calls.append(kwargs)
        return {"Tone": "ok"}

    with patch(_RUN_EVAL, new=_fake):
        await run_completion(nodes, ctx)

    assert calls[0]["model_name"] == "TURING_FLASH"
    assert calls[0]["timeout_seconds"] == 30


# ---------------------------------------------------------------------------
# HTTP client (wire contract)
# ---------------------------------------------------------------------------


def _mock_transport(handler):
    """Patch httpx.AsyncClient so run_evaluation hits `handler` instead of the net."""
    real_init = httpx.AsyncClient.__init__

    def _init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        real_init(self, *args, **kwargs)

    return patch.object(httpx.AsyncClient, "__init__", _init)


async def test_run_evaluation_sends_documented_request():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_wire_body())

    with _mock_transport(handler):
        metrics = await run_evaluation(
            api_key="key-1",
            secret_key="secret-1",
            base_url="https://api.futureagi.com",
            eval_name="Tone",
            inputs={"input": "hi", "output": "hello"},
            model_name="TURING_FLASH",
        )

    assert seen["url"] == "https://api.futureagi.com/sdk/api/v1/new-eval/"
    assert seen["headers"]["x-api-key"] == "key-1"
    assert seen["headers"]["x-secret-key"] == "secret-1"
    assert seen["body"]["eval_name"] == "Tone"
    assert seen["body"]["model"] == "TURING_FLASH"
    assert seen["body"]["is_async"] is False
    assert metrics["Tone"] == "positive"


async def test_run_evaluation_omits_model_when_unset():
    def handler(request):
        assert "model" not in json.loads(request.content)
        return httpx.Response(200, json=_wire_body())

    with _mock_transport(handler):
        await run_evaluation(
            api_key="k",
            secret_key="s",
            base_url="https://api.futureagi.com",
            eval_name="Tone",
            inputs={},
        )


async def test_run_evaluation_trims_trailing_slash_on_base_url():
    def handler(request):
        assert str(request.url) == "https://api.futureagi.com/sdk/api/v1/new-eval/"
        return httpx.Response(200, json=_wire_body())

    with _mock_transport(handler):
        await run_evaluation(
            api_key="k",
            secret_key="s",
            base_url="https://api.futureagi.com/",
            eval_name="Tone",
            inputs={},
        )


async def test_run_evaluation_maps_401_and_403_to_auth_error():
    """The live API answers bad credentials with 401; their SDK only checks 403."""
    for status in (401, 403):

        def handler(request, _s=status):
            return httpx.Response(_s, json={"detail": "Invalid API key or secret key"})

        with _mock_transport(handler):
            try:
                await run_evaluation(
                    api_key="bad",
                    secret_key="bad",
                    base_url="https://api.futureagi.com",
                    eval_name="Tone",
                    inputs={},
                )
            except FutureAGIAuthError:
                continue
            raise AssertionError(f"HTTP {status} should raise FutureAGIAuthError")


async def test_run_evaluation_raises_on_server_error():
    def handler(request):
        return httpx.Response(500, text="boom")

    with _mock_transport(handler):
        try:
            await run_evaluation(
                api_key="k",
                secret_key="s",
                base_url="https://api.futureagi.com",
                eval_name="Tone",
                inputs={},
            )
        except FutureAGIAPIError as exc:
            assert exc.status_code == 500
            # The API's own explanation must survive into the message, which
            # httpx's raise_for_status text does not carry.
            assert "boom" in exc.body
            return
    raise AssertionError("expected FutureAGIAPIError on 500")


# ---------------------------------------------------------------------------
# Dataset filing (dashboard visibility)
# ---------------------------------------------------------------------------


def test_truncate_cell_respects_future_agi_limit():
    assert truncate_cell(None) == ""
    assert truncate_cell(123) == "123"
    long_value = "x" * (MAX_CELL_CHARS + 5000)
    out = truncate_cell(long_value)
    assert len(out) == MAX_CELL_CHARS
    assert out.endswith("x")


async def test_completion_skips_dataset_when_name_blank():
    """Scoring-only stays the default; nothing is stored on Future AGI."""
    nodes = [_fi_node(api_key="k", secret_key="s")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake_eval(**kwargs):
        return {"Tone": "positive"}

    with patch(_RUN_EVAL, new=_fake_eval), patch(_ENSURE_DATASET) as ensure_mock:
        results = await run_completion(nodes, ctx)

    ensure_mock.assert_not_called()
    assert "dataset" not in results["futureagi_fi-1"]


async def test_completion_files_dataset_row_with_scores():
    nodes = [
        _fi_node(
            api_key="k",
            secret_key="s",
            futureagi_eval_templates="Tone, Toxicity",
            futureagi_dataset_name="dograh-calls",
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    rows = []

    async def _fake_eval(**kwargs):
        return {kwargs["eval_name"]: "positive"}

    async def _fake_ensure(**kwargs):
        assert kwargs["dataset_name"] == "dograh-calls"
        return "ds-123"

    async def _fake_append(**kwargs):
        rows.append(kwargs)

    with (
        patch(_RUN_EVAL, new=_fake_eval),
        patch(_ENSURE_DATASET, new=_fake_ensure),
        patch(_APPEND_ROW, new=_fake_append),
    ):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["dataset"] == {
        "status": "row_added",
        "dataset_name": "dograh-calls",
        "dataset_id": "ds-123",
    }

    values = rows[0]["values"]
    assert values["workflow_run_id"] == 42
    assert values["transcript"] == "user: hello"
    assert "Tone=[Tone=positive]" in values["eval_scores"]
    assert "Toxicity=[Toxicity=positive]" in values["eval_scores"]


async def test_dataset_failure_does_not_downgrade_successful_evals():
    """Scores are already recorded; a dataset outage must not hide them."""
    nodes = [
        _fi_node(api_key="k", secret_key="s", futureagi_dataset_name="dograh-calls")
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake_eval(**kwargs):
        return {"Tone": "positive"}

    async def _boom(**kwargs):
        raise RuntimeError("dataset service down")

    with patch(_RUN_EVAL, new=_fake_eval), patch(_ENSURE_DATASET, new=_boom):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    assert outcome["status"] == "evaluated"
    assert outcome["evals_succeeded"] == 1
    assert outcome["dataset"]["status"] == "failed"
    assert "dataset service down" in outcome["dataset"]["error"]


async def test_ensure_dataset_reuses_exact_name_match_without_creating():
    _DATASET_ID_CACHE.clear()
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "get-datasets-names" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "datasets": [
                            {"datasetId": "other", "datasetName": "dograh-calls-stg"},
                            {"datasetId": "ds-1", "datasetName": "dograh-calls"},
                        ]
                    }
                },
            )
        raise AssertionError("unexpected request")

    with _mock_transport(handler):
        dataset_id = await ensure_dataset(
            api_key="k",
            secret_key="s",
            base_url="https://api.futureagi.com",
            dataset_name="dograh-calls",
        )

    assert dataset_id == "ds-1"
    assert not any("create-empty-dataset" in u for u in calls)
    _DATASET_ID_CACHE.clear()


async def test_ensure_dataset_creates_and_adds_columns_when_absent():
    _DATASET_ID_CACHE.clear()
    seen = []

    def handler(request):
        url = str(request.url)
        seen.append(url)
        if "get-datasets-names" in url:
            return httpx.Response(200, json={"result": {"datasets": []}})
        if "create-empty-dataset" in url:
            body = json.loads(request.content)
            assert body["new_dataset_name"] == "fresh"
            assert body["model_type"] == "GenerativeLLM"
            return httpx.Response(200, json={"result": {"datasetId": "ds-new"}})
        if "add_columns" in url:
            cols = json.loads(request.content)["new_columns_data"]
            assert [c["name"] for c in cols] == [n for n, _ in DATASET_COLUMNS]
            return httpx.Response(200, json={})
        raise AssertionError("unexpected request")

    with _mock_transport(handler):
        dataset_id = await ensure_dataset(
            api_key="k",
            secret_key="s",
            base_url="https://api.futureagi.com",
            dataset_name="fresh",
        )

    assert dataset_id == "ds-new"
    assert any("add_columns" in u for u in seen)
    _DATASET_ID_CACHE.clear()


async def test_ensure_dataset_caches_per_credential_not_globally():
    """Two orgs using the same dataset name must not share a resolved id."""
    _DATASET_ID_CACHE.clear()

    def make_handler(dataset_id):
        def handler(request):
            if "get-datasets-names" in str(request.url):
                return httpx.Response(
                    200,
                    json={
                        "result": {
                            "datasets": [
                                {"datasetId": dataset_id, "datasetName": "calls"}
                            ]
                        }
                    },
                )
            raise AssertionError("should not create")

        return handler

    with _mock_transport(make_handler("org-a-ds")):
        first = await ensure_dataset(
            api_key="key-a",
            secret_key="s",
            base_url="https://api.futureagi.com",
            dataset_name="calls",
        )
    with _mock_transport(make_handler("org-b-ds")):
        second = await ensure_dataset(
            api_key="key-b",
            secret_key="s",
            base_url="https://api.futureagi.com",
            dataset_name="calls",
        )

    assert first == "org-a-ds"
    assert second == "org-b-ds"
    _DATASET_ID_CACHE.clear()


async def test_append_call_row_sends_named_cells():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={})

    with _mock_transport(handler):
        await append_call_row(
            api_key="k",
            secret_key="s",
            base_url="https://api.futureagi.com",
            dataset_id="ds-1",
            values={"workflow_run_id": 7, "transcript": "user: hi"},
        )

    assert captured["url"].endswith("/model-hub/develops/ds-1/add_rows/")
    assert captured["headers"]["x-api-key"] == "k"
    cells = captured["body"]["rows"][0]["cells"]
    by_name = {c["column_name"]: c["value"] for c in cells}
    assert by_name["workflow_run_id"] == "7"
    assert by_name["transcript"] == "user: hi"
    assert set(by_name) == {name for name, _ in DATASET_COLUMNS}
    assert len({c["row_id"] for c in cells}) == 1


async def test_append_call_row_maps_401_to_auth_error():
    def handler(request):
        return httpx.Response(401, json={"detail": "Invalid API key or secret key"})

    with _mock_transport(handler):
        try:
            await append_call_row(
                api_key="bad",
                secret_key="bad",
                base_url="https://api.futureagi.com",
                dataset_id="ds-1",
                values={},
            )
        except FutureAGIAuthError:
            return
    raise AssertionError("expected FutureAGIAuthError")


# ---------------------------------------------------------------------------
# Eval catalogue + input narrowing
#
# Future AGI's API rejects supersets: sending {input, output, context} to a
# template that declares only {output} is a 400. These cover the narrowing that
# prevents it.
# ---------------------------------------------------------------------------


def _registry(**evals):
    return {
        name: {"name": name, "config": {"required_keys": keys}}
        for name, keys in evals.items()
    }


def test_required_keys_for_reads_config_block():
    reg = _registry(toxicity=["output"])
    assert required_keys_for(reg, "toxicity") == ["output"]


def test_required_keys_for_tolerates_malformed_entries():
    assert required_keys_for({}, "missing") == []
    assert required_keys_for({"a": "not-a-dict"}, "a") == []
    assert required_keys_for({"a": {"name": "a"}}, "a") == []
    assert required_keys_for({"a": {"config": {"required_keys": "nope"}}}, "a") == []


def test_map_inputs_narrows_to_declared_keys_only():
    """The whole point: an eval wanting only `output` must receive only that."""
    reg = _registry(toxicity=["output"])
    available = build_eval_inputs(_snapshot())

    mapped = map_inputs(reg, "toxicity", available)

    assert set(mapped) == {"output"}
    assert mapped["output"] == "hi there"


def test_map_inputs_resolves_through_aliases():
    """A template asking for `generated_transcript` is fed our transcript."""
    reg = _registry(dead_air=["generated_transcript"])
    available = build_eval_inputs(_snapshot())

    mapped = map_inputs(reg, "dead_air", available)

    assert mapped == {"generated_transcript": "user: hello"}


def test_map_inputs_handles_multi_key_templates():
    reg = _registry(coherence=["input", "output", "context"])
    mapped = map_inputs(reg, "coherence", build_eval_inputs(_snapshot()))
    assert set(mapped) == {"input", "output", "context"}
    assert mapped["input"] == "hello"
    assert mapped["context"] == "system prompt"


def test_map_inputs_omits_keys_it_cannot_satisfy():
    reg = _registry(grounded=["output", "expected_answer"])
    mapped = map_inputs(reg, "grounded", build_eval_inputs(_snapshot()))
    assert set(mapped) == {"output"}


def test_map_inputs_passes_through_when_eval_unknown():
    """An empty catalogue must not silently send {} and score nothing."""
    available = build_eval_inputs(_snapshot())
    assert map_inputs({}, "anything", available) == available


@pytest.mark.real_registry
async def test_completion_sends_only_declared_keys_per_template():
    nodes = [
        _fi_node(
            api_key="k", secret_key="s", futureagi_eval_templates="toxicity, coherence"
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    sent = {}

    async def _fake_registry(**kwargs):
        return _registry(toxicity=["output"], coherence=["input", "output"])

    async def _fake_eval(**kwargs):
        sent[kwargs["eval_name"]] = kwargs["inputs"]
        return {kwargs["eval_name"]: 0.1}

    with patch(_LOAD_REGISTRY, new=_fake_registry), patch(_RUN_EVAL, new=_fake_eval):
        results = await run_completion(nodes, ctx)

    assert results["futureagi_fi-1"]["evals_succeeded"] == 2
    assert set(sent["toxicity"]) == {"output"}
    assert set(sent["coherence"]) == {"input", "output"}


@pytest.mark.real_registry
async def test_completion_reports_unknown_eval_name_clearly():
    """A misspelled eval should say so, not surface an opaque 400."""
    nodes = [_fi_node(api_key="k", secret_key="s", futureagi_eval_templates="tonne")]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _fake_registry(**kwargs):
        return _registry(tone=["output"], toxicity=["output"])

    async def _never_called(**kwargs):
        raise AssertionError("must not call the API for an unknown eval")

    with patch(_LOAD_REGISTRY, new=_fake_registry), patch(_RUN_EVAL, new=_never_called):
        results = await run_completion(nodes, ctx)

    error = results["futureagi_fi-1"]["evaluations"]["tonne"]["error"]
    assert "unknown_eval" in error
    assert "tonne" in error


@pytest.mark.real_registry
async def test_completion_reports_catalogue_auth_failure_once_per_node():
    nodes = [
        _fi_node(
            api_key="k", secret_key="bad", futureagi_eval_templates="tone, toxicity"
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})

    async def _reject(**kwargs):
        raise FutureAGIAuthError("nope")

    with patch(_LOAD_REGISTRY, new=_reject):
        results = await run_completion(nodes, ctx)

    assert results["futureagi_fi-1"] == {"error": "invalid_credentials"}


async def test_load_registry_indexes_by_name_and_caches():
    clear_registry_cache()
    hits = []

    def handler(request):
        hits.append(str(request.url))
        return httpx.Response(
            200,
            json={
                "result": [
                    {"name": "tone", "config": {"required_keys": ["output"]}},
                    {"name": "toxicity", "config": {"required_keys": ["output"]}},
                    {"no_name": True},
                ]
            },
        )

    with _mock_transport(handler):
        first = await load_registry(
            api_key="k", secret_key="s", base_url="https://api.futureagi.com"
        )
        second = await load_registry(
            api_key="k", secret_key="s", base_url="https://api.futureagi.com"
        )

    assert set(first) == {"tone", "toxicity"}
    assert second is first
    assert len(hits) == 1
    assert hits[0].endswith("/sdk/api/v1/get-evals/")
    clear_registry_cache()


async def test_load_registry_caches_per_credential():
    """One org's custom evals must never be served to another."""
    clear_registry_cache()

    def make(name):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "result": [{"name": name, "config": {"required_keys": ["output"]}}]
                },
            )

        return handler

    with _mock_transport(make("org_a_eval")):
        a = await load_registry(
            api_key="key-a", secret_key="s", base_url="https://api.futureagi.com"
        )
    with _mock_transport(make("org_b_eval")):
        b = await load_registry(
            api_key="key-b", secret_key="s", base_url="https://api.futureagi.com"
        )

    assert set(a) == {"org_a_eval"}
    assert set(b) == {"org_b_eval"}
    clear_registry_cache()


async def test_load_registry_degrades_to_empty_on_server_error():
    """A catalogue outage must not block evaluation."""
    clear_registry_cache()

    def handler(request):
        return httpx.Response(500, text="down")

    with _mock_transport(handler):
        reg = await load_registry(
            api_key="k", secret_key="s", base_url="https://api.futureagi.com"
        )

    assert reg == {}
    clear_registry_cache()


async def test_load_registry_raises_auth_error_on_401():
    clear_registry_cache()

    def handler(request):
        return httpx.Response(401, json={"detail": "Invalid API key or secret key"})

    with _mock_transport(handler):
        try:
            await load_registry(
                api_key="bad", secret_key="bad", base_url="https://api.futureagi.com"
            )
        except FutureAGIAuthError:
            clear_registry_cache()
            return
    raise AssertionError("expected FutureAGIAuthError")


# ---------------------------------------------------------------------------
# Real catalogue signatures
#
# Key sets below are the ones this account actually returned from
# /sdk/api/v1/get-evals/, so these lock in behaviour against the real API
# rather than an invented shape.
# ---------------------------------------------------------------------------

_REAL_EVALS = {
    "tone": ["output"],
    "toxicity": ["output"],
    "conversation_coherence": ["conversation"],
    "customer_agent_task_completion": ["agent_prompt", "conversation"],
    "customer_agent_prompt_conformance": ["system_prompt", "conversation"],
    "dead_air_detection": ["input_audio"],
    "audio_quality": ["input_audio"],
    "ASR/STT_accuracy": ["audio", "generated_transcript"],
    "accuracy": ["output", "expected"],
    "groundedness": ["output", "input", "context"],
}


def _real_registry():
    return {
        name: {"name": name, "config": {"required_keys": keys}}
        for name, keys in _REAL_EVALS.items()
    }


def test_agent_prompt_resolves_to_the_system_prompt():
    """Unlocks the whole customer_agent_* / intake / misselling family."""
    reg = _real_registry()
    mapped = map_inputs(
        reg, "customer_agent_task_completion", build_eval_inputs(_snapshot())
    )
    assert set(mapped) == {"agent_prompt", "conversation"}
    assert mapped["agent_prompt"] == "system prompt"
    assert mapped["conversation"] == "user: hello"


def test_system_prompt_variant_also_resolves():
    reg = _real_registry()
    mapped = map_inputs(
        reg, "customer_agent_prompt_conformance", build_eval_inputs(_snapshot())
    )
    assert mapped["system_prompt"] == "system prompt"


def test_transcript_only_evals_need_no_audio():
    reg = _real_registry()
    available = build_eval_inputs(_snapshot())
    for name in ("tone", "toxicity", "conversation_coherence"):
        assert unmet_keys(reg, name, available) == []


def test_audio_evals_unmet_without_a_recording():
    """No recording must read as 'we have no audio', not an opaque 400."""
    reg = _real_registry()
    available = build_eval_inputs(_snapshot())
    assert unmet_keys(reg, "dead_air_detection", available) == ["input_audio"]
    assert unmet_keys(reg, "audio_quality", available) == ["input_audio"]


def test_ground_truth_evals_are_always_unmet_from_a_call():
    """A live call has no expected answer; flagging beats paying for a 400."""
    reg = _real_registry()
    assert unmet_keys(reg, "accuracy", build_eval_inputs(_snapshot())) == ["expected"]


def test_audio_evals_satisfied_once_a_recording_url_is_offered():
    reg = _real_registry()
    available = build_eval_inputs(_snapshot())
    available["input_audio"] = "https://dograh.example/rec.wav"
    available["audio"] = "https://dograh.example/rec.wav"

    assert unmet_keys(reg, "dead_air_detection", available) == []
    mapped = map_inputs(reg, "dead_air_detection", available)
    assert mapped == {"input_audio": "https://dograh.example/rec.wav"}

    # ASR accuracy needs the transcript alongside the audio; we have both.
    assert unmet_keys(reg, "ASR/STT_accuracy", available) == []
    asr = map_inputs(reg, "ASR/STT_accuracy", available)
    assert set(asr) == {"audio", "generated_transcript"}
    assert asr["generated_transcript"] == "user: hello"


@pytest.mark.real_registry
async def test_completion_offers_recording_url_to_audio_evals():
    nodes = [
        _fi_node(
            api_key="k",
            secret_key="s",
            futureagi_eval_templates="dead_air_detection",
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})
    ctx.workflow_run.recording_url = "https://dograh.example/rec.wav"

    sent = {}

    async def _fake_registry(**kwargs):
        return _real_registry()

    async def _fake_eval(**kwargs):
        sent.update(kwargs["inputs"])
        return {"dead_air_detection": 0.0}

    with patch(_LOAD_REGISTRY, new=_fake_registry), patch(_RUN_EVAL, new=_fake_eval):
        results = await run_completion(nodes, ctx)

    assert results["futureagi_fi-1"]["evals_succeeded"] == 1
    assert sent == {"input_audio": "https://dograh.example/rec.wav"}


@pytest.mark.real_registry
async def test_completion_skips_audio_eval_without_burning_a_call():
    nodes = [
        _fi_node(
            api_key="k", secret_key="s", futureagi_eval_templates="dead_air_detection"
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})
    ctx.workflow_run.recording_url = None

    async def _fake_registry(**kwargs):
        return _real_registry()

    async def _never_called(**kwargs):
        raise AssertionError("must not pay for an eval we cannot satisfy")

    with patch(_LOAD_REGISTRY, new=_fake_registry), patch(_RUN_EVAL, new=_never_called):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]["evaluations"]["dead_air_detection"]
    assert outcome == {"error": "missing_input: input_audio"}


@pytest.mark.real_registry
async def test_completion_mixed_batch_reports_each_eval_honestly():
    """The five evals actually configured on this account, in one run."""
    nodes = [
        _fi_node(
            api_key="k",
            secret_key="s",
            futureagi_eval_templates=(
                "tone, toxicity, conversation_coherence, "
                "customer_agent_task_completion, dead_air_detection"
            ),
        )
    ]
    ctx = _completion_context(logs={FUTUREAGI_PAYLOAD_LOG_KEY: _snapshot()})
    ctx.workflow_run.recording_url = None

    async def _fake_registry(**kwargs):
        return _real_registry()

    async def _fake_eval(**kwargs):
        return {kwargs["eval_name"]: "ok"}

    with patch(_LOAD_REGISTRY, new=_fake_registry), patch(_RUN_EVAL, new=_fake_eval):
        results = await run_completion(nodes, ctx)

    outcome = results["futureagi_fi-1"]
    evaluations = outcome["evaluations"]

    # Four now succeed, including the one that needed agent_prompt.
    assert outcome["evals_succeeded"] == 4
    for name in (
        "tone",
        "toxicity",
        "conversation_coherence",
        "customer_agent_task_completion",
    ):
        assert "metrics" in evaluations[name], name

    # Only the audio one is held back, with the reason named.
    assert evaluations["dead_air_detection"] == {"error": "missing_input: input_audio"}

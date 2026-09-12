from __future__ import annotations

from pydantic import model_validator

from api.services.integrations.base import IntegrationNodeRegistration
from api.services.workflow.node_data import BaseNodeData
from api.services.workflow.node_specs._base import (
    GraphConstraints,
    NodeCategory,
    NodeExample,
    PropertyType,
)
from api.services.workflow.node_specs.model_spec import (
    build_spec,
    node_spec,
    spec_field,
)

from .client import DEFAULT_FUTUREAGI_BASE_URL


def parse_eval_templates(raw: str | None) -> list[str]:
    """Split the comma-separated template field into clean names.

    Shared by the validator and the completion handler so the node can never
    validate against one list and then evaluate against another.
    """
    if not raw:
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]


@node_spec(
    name="futureagi",
    display_name="Future AGI",
    description="Score the completed call with Future AGI evaluation templates",
    llm_hint=(
        "Future AGI is a post-call evaluation export. It does not participate in "
        "the conversation graph and should not be connected to other nodes."
    ),
    category=NodeCategory.integration,
    icon="ClipboardCheck",
    examples=[
        NodeExample(
            name="futureagi_eval",
            data={
                "name": "Call Quality Evals",
                "futureagi_enabled": True,
                "futureagi_api_key": "fi_xxxxxxxx",
                "futureagi_secret_key": "fisk_xxxxxxxx",
                "futureagi_eval_templates": "Tone, Toxicity",
            },
        )
    ],
    graph_constraints=GraphConstraints(
        min_incoming=0, max_incoming=0, min_outgoing=0, max_outgoing=0, max_instances=1
    ),
    property_order=(
        "name",
        "futureagi_enabled",
        "futureagi_api_key",
        "futureagi_secret_key",
        "futureagi_eval_templates",
        "futureagi_dataset_name",
        "futureagi_model_name",
        "futureagi_base_url",
        "futureagi_timeout_seconds",
    ),
    field_overrides={
        "name": {
            "spec_default": "Future AGI",
            "description": "Short identifier for this Future AGI evaluation configuration.",
        },
        "futureagi_enabled": {
            "display_name": "Enabled",
            "description": "When false, Dograh skips evaluating this call with Future AGI.",
        },
        "futureagi_api_key": {
            "display_name": "Future AGI API Key",
            "description": "API key from the Future AGI dashboard, under Developer → Keys.",
            "required": True,
        },
        "futureagi_secret_key": {
            "display_name": "Future AGI Secret Key",
            "description": "Secret key issued alongside the API key.",
            "required": True,
        },
        "futureagi_dataset_name": {
            "display_name": "Dataset name",
            "description": (
                "Optional. When set, each completed call is also filed as a row "
                "in this Future AGI dataset so it is visible in their dashboard. "
                "Created on first use. Leave blank to score calls without "
                "storing anything on Future AGI."
            ),
        },
        "futureagi_eval_templates": {
            "display_name": "Evaluations",
            "description": (
                "Comma-separated evaluation names to run against the call, either "
                "Future AGI built-ins (Tone, Toxicity) or custom evals you created "
                "in their dashboard. Each runs as its own call and is billed "
                "separately."
            ),
            "required": True,
        },
    },
)
class FutureAGINodeData(BaseNodeData):
    futureagi_enabled: bool = spec_field(
        default=True,
        ui_type=PropertyType.boolean,
        display_name="Enabled",
        description="When false, Dograh skips evaluating this call with Future AGI.",
    )
    futureagi_api_key: str | None = spec_field(
        default=None,
        ui_type=PropertyType.string,
        display_name="Future AGI API Key",
        description="API key from the Future AGI dashboard, under Developer → Keys.",
    )
    futureagi_secret_key: str | None = spec_field(
        default=None,
        ui_type=PropertyType.string,
        display_name="Future AGI Secret Key",
        description="Secret key issued alongside the API key.",
    )
    futureagi_eval_templates: str | None = spec_field(
        default=None,
        ui_type=PropertyType.string,
        display_name="Evaluations",
        description=(
            "Comma-separated evaluation names to run against the call, either "
            "Future AGI built-ins (Tone, Toxicity) or custom evals you created "
            "in their dashboard. Each runs as its own call and is billed separately."
        ),
    )
    futureagi_dataset_name: str | None = spec_field(
        default=None,
        ui_type=PropertyType.string,
        display_name="Dataset name",
        description=(
            "Optional. When set, each completed call is also filed as a row in "
            "this Future AGI dataset so it is visible in their dashboard. The "
            "dataset is created on first use. Leave blank to score calls "
            "without storing anything on Future AGI."
        ),
    )
    futureagi_model_name: str | None = spec_field(
        default=None,
        ui_type=PropertyType.string,
        display_name="Evaluation model",
        description=(
            "Optional Future AGI agent to score with: TURING_LARGE, TURING_SMALL, "
            "TURING_FLASH, PROTECT or PROTECT_FLASH. Leave blank to use the "
            "evaluation's own configured model."
        ),
    )
    futureagi_base_url: str = spec_field(
        default=DEFAULT_FUTUREAGI_BASE_URL,
        ui_type=PropertyType.string,
        display_name="API base URL",
        description="Override only for a self-hosted or regional Future AGI deployment.",
    )
    futureagi_timeout_seconds: int = spec_field(
        default=60,
        ui_type=PropertyType.number,
        display_name="Timeout (seconds)",
        description=(
            "Per-evaluation timeout. Evaluations run after the call, so this "
            "trades completeness against how long the post-call job may run."
        ),
    )

    @model_validator(mode="after")
    def _validate_enabled_config(self):
        if not self.futureagi_enabled:
            return self

        missing: list[str] = []
        if not self.futureagi_api_key or not self.futureagi_api_key.strip():
            missing.append("futureagi_api_key")
        if not self.futureagi_secret_key or not self.futureagi_secret_key.strip():
            missing.append("futureagi_secret_key")
        if not parse_eval_templates(self.futureagi_eval_templates):
            missing.append("futureagi_eval_templates")

        if missing:
            fields = ", ".join(missing)
            raise ValueError(
                f"Future AGI node is enabled but missing required fields: {fields}"
            )

        if self.futureagi_timeout_seconds <= 0:
            raise ValueError("futureagi_timeout_seconds must be a positive integer")

        return self


SPEC = build_spec(FutureAGINodeData)


NODE = IntegrationNodeRegistration(
    type_name="futureagi",
    data_model=FutureAGINodeData,
    node_spec=SPEC,
    sensitive_fields=("futureagi_api_key", "futureagi_secret_key"),
)

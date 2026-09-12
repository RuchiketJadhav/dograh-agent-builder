from __future__ import annotations

import asyncio
import hashlib
import uuid
from typing import Any

import httpx
from loguru import logger

from .client import FutureAGIAuthError

# Dataset routes, taken from the futureagi SDK's own route table
# (fi/utils/routes.py). Same reasoning as client.py: Dograh owns the wire
# contract rather than taking the SDK dependency.
_DATASET_LOOKUP_PATH = "model-hub/develops/get-datasets-names/"
_DATASET_CREATE_PATH = "model-hub/develops/create-empty-dataset/"
_DATASET_ADD_COLUMNS_PATH = "model-hub/develops/{dataset_id}/add_columns/"
_DATASET_ADD_ROWS_PATH = "model-hub/develops/{dataset_id}/add_rows/"

# Future AGI rejects any cell longer than this (fi/datasets/types.py,
# Cell.validate_value). Independent of, and stricter than, the transcript cap
# in collector.py — a snapshot that is legal to store in Dograh can still be
# too long to file as a dataset cell.
MAX_CELL_CHARS = 65_535

_MODEL_TYPE = "GenerativeLLM"
_SOURCE_OTHERS = "OTHERS"

# Columns Dograh files each call under. Order is the display order.
DATASET_COLUMNS: tuple[tuple[str, str], ...] = (
    ("workflow_run_id", "text"),
    ("agent_version", "text"),
    ("call_disposition", "text"),
    ("input", "text"),
    ("output", "text"),
    ("transcript", "text"),
    ("context", "text"),
    ("eval_scores", "text"),
)

# Resolved dataset ids, keyed by (base_url, credential fingerprint, name) so a
# second organization never reuses the first one's dataset. The API key is
# hashed rather than stored, so a stray log of this dict cannot leak it.
_DATASET_ID_CACHE: dict[tuple[str, str, str], str] = {}
_CACHE_LIMIT = 256
_RESOLVE_LOCK = asyncio.Lock()


def _cache_key(base_url: str, api_key: str, dataset_name: str) -> tuple[str, str, str]:
    fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return (base_url, fingerprint, dataset_name)


def truncate_cell(value: Any) -> str:
    """Coerce a value to a string Future AGI will accept as a cell."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= MAX_CELL_CHARS:
        return text
    # Keep the tail for the same reason collector.py does: the end of a call
    # carries the outcome.
    return text[-MAX_CELL_CHARS:]


def _headers(api_key: str, secret_key: str) -> dict[str, str]:
    return {"X-Api-Key": api_key, "X-Secret-Key": secret_key}


def _raise_for_auth(response: httpx.Response) -> None:
    if response.status_code in (401, 403):
        raise FutureAGIAuthError(
            f"Future AGI rejected the API key/secret pair (HTTP {response.status_code})"
        )


async def _lookup_dataset_id(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    dataset_name: str,
) -> str | None:
    """Return the id of an existing dataset with this exact name, else None."""
    response = await client.get(
        f"{base_url}/{_DATASET_LOOKUP_PATH}",
        params={"search_text": dataset_name},
        headers=headers,
    )
    _raise_for_auth(response)
    if response.status_code >= 400:
        return None

    body = response.json()
    datasets = ((body or {}).get("result") or {}).get("datasets") or []
    # The lookup is a search, not an exact match, so "calls" would also return
    # "calls-staging". Only an exact name is ours.
    for dataset in datasets:
        if isinstance(dataset, dict) and dataset.get("datasetName") == dataset_name:
            return str(dataset.get("datasetId"))
    return None


async def _create_dataset(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    dataset_name: str,
) -> str:
    response = await client.post(
        f"{base_url}/{_DATASET_CREATE_PATH}",
        json={"new_dataset_name": dataset_name, "model_type": _MODEL_TYPE},
        headers=headers,
    )
    _raise_for_auth(response)
    response.raise_for_status()
    body = response.json()
    return str(((body or {}).get("result") or {})["datasetId"])


async def _add_columns(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict[str, str],
    dataset_id: str,
) -> None:
    columns = [
        {
            "id": str(uuid.uuid4()),
            "name": name,
            "data_type": data_type,
            "source": _SOURCE_OTHERS,
            "source_id": None,
            "metadata": {},
        }
        for name, data_type in DATASET_COLUMNS
    ]
    response = await client.post(
        f"{base_url}/{_DATASET_ADD_COLUMNS_PATH.format(dataset_id=dataset_id)}",
        json={"new_columns_data": columns},
        headers=headers,
    )
    _raise_for_auth(response)
    if response.status_code >= 400:
        # Not fatal. The common cause is columns that already exist, and rows
        # are matched on column_name anyway, so a failed add here still lets
        # the row land.
        logger.warning(
            f"[futureagi] add_columns on dataset {dataset_id} returned "
            f"{response.status_code}: {response.text[:200]}"
        )


async def ensure_dataset(
    *,
    api_key: str,
    secret_key: str,
    base_url: str,
    dataset_name: str,
    timeout_seconds: int = 60,
) -> str:
    """
    Resolve ``dataset_name`` to a dataset id, creating it on first use.

    Serialized behind one lock: two calls completing at once for the same new
    dataset would otherwise both miss the lookup and create duplicates that
    the API then refuses to disambiguate by name.
    """
    base_url = base_url.rstrip("/")
    key = _cache_key(base_url, api_key, dataset_name)

    cached = _DATASET_ID_CACHE.get(key)
    if cached:
        return cached

    async with _RESOLVE_LOCK:
        # Re-check: another completion may have resolved it while we waited.
        cached = _DATASET_ID_CACHE.get(key)
        if cached:
            return cached

        headers = _headers(api_key, secret_key)
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            dataset_id = await _lookup_dataset_id(
                client, base_url, headers, dataset_name
            )
            if dataset_id is None:
                dataset_id = await _create_dataset(
                    client, base_url, headers, dataset_name
                )
                # Columns are added only for a dataset we just created. An
                # existing one already has them, and re-adding would either
                # error or duplicate.
                await _add_columns(client, base_url, headers, dataset_id)
                logger.info(
                    f"[futureagi] created dataset '{dataset_name}' ({dataset_id})"
                )

        if len(_DATASET_ID_CACHE) >= _CACHE_LIMIT:
            _DATASET_ID_CACHE.clear()
        _DATASET_ID_CACHE[key] = dataset_id
        return dataset_id


async def append_call_row(
    *,
    api_key: str,
    secret_key: str,
    base_url: str,
    dataset_id: str,
    values: dict[str, Any],
    timeout_seconds: int = 60,
) -> None:
    """File one completed call as a row in the dataset.

    Cells carry ``column_name``; the ids are placeholders the API matches by
    name, which is how the vendor's own SDK builds rows from plain dicts.
    """
    row_id = str(uuid.uuid4())
    cells = [
        {
            "column_id": str(uuid.uuid4()),
            "row_id": row_id,
            "column_name": name,
            "value": truncate_cell(values.get(name)),
            "value_infos": [],
            "metadata": {},
            "status": None,
            "failure_reason": None,
        }
        for name, _ in DATASET_COLUMNS
    ]

    payload = {"rows": [{"id": row_id, "order": 0, "cells": cells}]}

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/{_DATASET_ADD_ROWS_PATH.format(dataset_id=dataset_id)}",
            json=payload,
            headers=_headers(api_key, secret_key),
        )

    _raise_for_auth(response)
    if response.status_code >= 400:
        logger.error(
            f"[futureagi] add_rows to dataset {dataset_id} failed with "
            f"{response.status_code}: {response.text[:200]}"
        )
    response.raise_for_status()

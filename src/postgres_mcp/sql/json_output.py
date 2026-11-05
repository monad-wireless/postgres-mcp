"""JSON for tool output: every value psycopg returns, as a value any JSON parser accepts.

``str(rows)`` produced Python literals (``Decimal('1.1000')``, ``UUID('…')``, ``b'\\xde'``, ``nan``,
``None``) that no JSON parser reads. ``to_jsonable`` maps each Python type the driver returns:

=============================  =================================================================
Python value                   JSON
=============================  =================================================================
``Decimal``                    string, so precision survives (``"1.1000"``, ``"NaN"``)
``float`` NaN / +-inf          ``"NaN"``, ``"Infinity"``, ``"-Infinity"``
``UUID``, IP address/network   string
``datetime``, ``date``,        ISO 8601 string, offset kept
``time``
``timedelta``                  ISO-ish ``str(td)``; intervals load as text (``type_adapters``), so
                               this only meets values from other sources
``bytes``, ``memoryview``      ``"\\x…"`` hex, as PostgreSQL prints ``bytea``
``Range``                      ``{"lower", "upper", "bounds", "empty"}``
``Multirange``                 list of ranges
``tuple`` (a record)           list
``Vector``                     list of floats, or ``{"dims", "preview"}`` past ``vector_preview``
``SparseVector``               ``{"dim", "indices", "values"}``
anything else                  ``str(value)``; encoding never raises
=============================  =================================================================

``float`` needs a pre-pass rather than ``json.dumps(default=…)``: ``dumps`` handles floats natively
and never calls ``default`` for them, so NaN would print as the invalid token ``NaN``.

``render_table`` adds the three output bounds: characters per cell, rows (fetched with
``fetchmany``, see ``SqlDriver.execute_table``), and bytes per response.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from psycopg.types.multirange import Multirange
from psycopg.types.range import Range

from .type_adapters import SparseVector
from .type_adapters import Vector


@dataclass(frozen=True, slots=True)
class OutputLimits:
    """Bounds on what one ``execute_sql`` call returns. ``None`` disables a bound."""

    max_rows: int | None = 500
    max_cell_chars: int | None = 2000
    max_result_bytes: int | None = 256 * 1024
    vector_preview: int | None = 8

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_cell_chars", "max_result_bytes", "vector_preview"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass(frozen=True, slots=True)
class ResultColumn:
    name: str
    type: str


@dataclass(frozen=True, slots=True)
class QueryTable:
    """The last statement's result: columns in order (duplicates kept), rows as tuples.

    ``more`` is true when the statement produced rows beyond the ``max_rows`` that were fetched.
    """

    columns: list[ResultColumn]
    rows: list[tuple[Any, ...]]
    more: bool = False


def _float(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _vector(values: tuple[float, ...], preview: int | None) -> Any:
    floats = [_float(v) for v in values]
    if preview is None or len(floats) <= preview:
        return floats
    return {"dims": len(floats), "preview": floats[:preview]}


def to_jsonable(value: Any, vector_preview: int | None = None) -> Any:
    """Map one value (recursively) to str, int, float, bool, None, list or dict."""
    # An isinstance chain, not ``match``: the fork's ruff target is still py39.
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _float(value)
    if isinstance(value, (Decimal, uuid.UUID, dt.timedelta)):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "\\x" + bytes(value).hex()
    if isinstance(value, Vector):
        return _vector(value.values, vector_preview)
    if isinstance(value, SparseVector):
        return {"dim": value.dim, "indices": list(value.indices), "values": [_float(v) for v in value.values]}
    if isinstance(value, Range):
        return {
            "lower": to_jsonable(value.lower, vector_preview),
            "upper": to_jsonable(value.upper, vector_preview),
            "bounds": value.bounds,
            "empty": value.isempty,
        }
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, vector_preview) for k, v in value.items()}
    if isinstance(value, (list, tuple, Multirange)):
        return [to_jsonable(v, vector_preview) for v in value]
    return str(value)  # IP addresses and networks, and any type not listed above


def dumps(value: Any, vector_preview: int | None = None) -> str:
    """Serialise ``value``; always valid JSON (``allow_nan=False`` cannot trip after the pre-pass)."""
    return json.dumps(to_jsonable(value, vector_preview), ensure_ascii=False, allow_nan=False)


def _truncate_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…[truncated {len(text) - limit} chars]"


def encode_cell(value: Any, limits: OutputLimits) -> Any:
    """One cell as JSON-ready data, long text cut to ``max_cell_chars``.

    A JSON document or array cell (``jsonb``, ``hstore``, arrays) that is too long becomes its JSON
    text, cut and marked, so the cut is visible instead of a silently shortened structure. A vector
    is already bounded by ``vector_preview``.
    """
    encoded = to_jsonable(value, limits.vector_preview)
    limit = limits.max_cell_chars
    if limit is None:
        return encoded
    if isinstance(encoded, str):
        return _truncate_text(encoded, limit)
    if isinstance(encoded, (dict, list)) and not isinstance(value, (Vector, SparseVector)):
        text = json.dumps(encoded, ensure_ascii=False, allow_nan=False)
        if len(text) > limit:
            return _truncate_text(text, limit)
    return encoded


def render_table(table: QueryTable, limits: OutputLimits) -> str:
    """``{"columns": [...], "rows": [[...]], "row_count": n, "note": ...}`` within ``max_result_bytes``.

    Rows are serialised one by one; the response keeps the longest prefix of rows that fits the
    byte budget. ``note`` appears only when something was cut, and says what and by which flag.
    """
    columns = [{"name": c.name, "type": c.type} for c in table.columns]
    rows = [json.dumps([encode_cell(v, limits) for v in row], ensure_ascii=False, allow_nan=False) for row in table.rows]

    notes: list[str] = []
    if table.more:
        notes.append(f"more rows exist; showing the first {len(rows)} (--max-rows)")
    kept = len(rows)
    if limits.max_result_bytes is not None:
        # Fixed part plus one comma per row, then rows while they fit.
        budget = limits.max_result_bytes - len(json.dumps(columns).encode()) - 200
        used = 0
        kept = 0
        for row in rows:
            size = len(row.encode()) + 1
            if used + size > budget:
                break
            used += size
            kept += 1
        if kept < len(rows):
            notes.append(f"response cut to {kept} of {len(rows)} fetched rows to stay under {limits.max_result_bytes} bytes (--max-result-bytes)")

    head = {"columns": columns, "row_count": kept}
    if notes:
        head["note"] = "; ".join(notes)
    head_json = json.dumps(head, ensure_ascii=False)
    return f'{head_json[:-1]}, "rows": [{", ".join(rows[:kept])}]}}'

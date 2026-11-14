"""Apache AGE graphs: the ``cypher()`` statement the tools run, and catalogue reads.

``build_cypher_statement`` writes ``SELECT * FROM ag_catalog.cypher('graph', $tag$query$tag$
[, $1]) AS ("col" ag_catalog.agtype, ...)`` so a caller passes only the graph, the Cypher and the
names of the returned columns. Graph and column names must be plain identifiers; the query goes
inside a dollar quote whose tag does not occur in it. Parameters travel as one JSON map bound to
the third argument, which AGE requires to be a bind parameter; Cypher reads them as ``$name``.

Catalogue facts used here (AGE 1.8.0): ``ag_catalog.ag_graph`` lists graphs with their schema,
``ag_catalog.ag_label`` lists labels with ``kind`` ``'v'``/``'e'`` and the backing table. Every
label table inherits from ``_ag_label_vertex`` or ``_ag_label_edge``, so samples read ``FROM ONLY``.
The upper 16 bits of a graph id are the label id (``id >> 48``; checked against ``ag_label`` on
a probe graph), which names the endpoint labels of an edge without calling an AGE function.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from psycopg.sql import Identifier

from ..sql import SafeSqlDriver
from ..sql import SqlDriver

AGE_MISSING = "Apache AGE is not installed in this database (it needs CREATE EXTENSION age)."
LABEL_ID_SHIFT = 48
DEFAULT_LABELS = frozenset({"_ag_label_vertex", "_ag_label_edge"})
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class CypherStatement:
    """``sql`` runs (psycopg placeholders); ``validate_as`` is the same statement for pglast; ``params`` binds."""

    sql: str
    validate_as: str
    params: list[Any]


def _identifier(value: str, what: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"{what} {value!r} must be a plain identifier (letters, digits, '_', not starting with a digit)")
    return value


def build_cypher_statement(graph: str, query: str, columns: list[str], params: dict[str, Any] | None = None) -> CypherStatement:
    _identifier(graph, "graph name")
    if not columns:
        raise ValueError("columns must name at least one returned column")
    for column in columns:
        _identifier(column, "column name")
    if len(set(columns)) != len(columns):
        raise ValueError("column names must be unique")
    if not query.strip():
        raise ValueError("query is empty")
    tag = "cypher"
    while f"${tag}$" in query:
        tag += "_"
    column_list = ", ".join(f'"{c}" ag_catalog.agtype' for c in columns)

    def statement(body: str, param: str) -> str:
        return f"SELECT * FROM ag_catalog.cypher('{graph}', ${tag}${body}${tag}${param}) AS ({column_list})"

    if params is None:
        text = statement(query, "")
        return CypherStatement(sql=text, validate_as=text, params=[])
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object; Cypher reads each key as $key")
    return CypherStatement(
        sql=statement(query.replace("%", "%%"), ", %s"),
        validate_as=statement(query, ", $1"),
        params=[json.dumps(params)],
    )


async def age_version(driver: SqlDriver) -> str | None:
    """The installed AGE version, or ``None`` when the extension is absent."""
    rows = await driver.execute_query("SELECT extversion FROM pg_catalog.pg_extension WHERE extname = 'age'")
    return str(rows[0].cells["extversion"]) if rows else None


_GRAPHS = """
SELECT g.name::text AS graph, g.namespace::text AS schema,
       count(l.name) FILTER (WHERE l.kind = 'v' AND l.name <> '_ag_label_vertex') AS vertex_labels,
       count(l.name) FILTER (WHERE l.kind = 'e' AND l.name <> '_ag_label_edge') AS edge_labels
FROM ag_catalog.ag_graph g
LEFT JOIN ag_catalog.ag_label l ON l.graph = g.graphid
GROUP BY g.name, g.namespace
ORDER BY g.name
"""

_LABELS = """
SELECT l.name::text AS label, l.id::integer AS label_id, l.kind::text AS kind,
       n.nspname::text AS table_schema, c.relname::text AS table_name, c.reltuples::bigint AS estimated_rows
FROM ag_catalog.ag_label l
JOIN ag_catalog.ag_graph g ON g.graphid = l.graph
JOIN pg_catalog.pg_class c ON c.oid = l.relation
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE g.name = {}
ORDER BY l.kind DESC, l.name
"""


async def list_graphs(driver: SqlDriver) -> list[dict[str, Any]]:
    rows = await driver.execute_query(_GRAPHS)
    return [dict(row.cells) for row in rows or []]


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, Decimal):
        return "numeric"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "map"
    return type(value).__name__


def _property_types(samples: list[Any]) -> dict[str, list[str]]:
    types: dict[str, set[str]] = {}
    for properties in samples:
        if isinstance(properties, dict):
            for key, value in properties.items():
                types.setdefault(str(key), set()).add(_type_name(value))
    return {key: sorted(found) for key, found in sorted(types.items())}


async def graph_schema(driver: SqlDriver, graph: str, sample_size: int) -> dict[str, Any] | None:
    """Labels of ``graph`` with row estimates, property keys and types from a sample, and edge endpoints.

    ``None`` when the graph does not exist. Property keys and endpoint pairs come from at most
    ``sample_size`` rows per label, so a key or pair that is rare can be missing.
    """
    _identifier(graph, "graph name")
    rows = await SafeSqlDriver.execute_param_query(driver, _LABELS, [graph])
    if not rows:
        return None
    labels = [dict(row.cells) for row in rows]
    names_by_id = {int(label["label_id"]): str(label["label"]) for label in labels}
    vertices: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for label in labels:
        table = f"{Identifier(label['table_schema']).as_string()}.{Identifier(label['table_name']).as_string()}"
        is_edge = label["kind"] == "e"
        columns = "start_id::text AS start_id, end_id::text AS end_id, properties" if is_edge else "properties"
        sample = await driver.execute_query(f"SELECT {columns} FROM ONLY {table} LIMIT {int(sample_size)}")  # type: ignore[arg-type]
        cells = [row.cells for row in sample or []]
        if label["label"] in DEFAULT_LABELS and not cells:
            continue
        estimate = label["estimated_rows"]
        entry: dict[str, Any] = {
            "label": label["label"],
            "table": f"{label['table_schema']}.{label['table_name']}",
            "estimated_rows": int(estimate) if estimate is not None and int(estimate) >= 0 else None,
            "sampled": len(cells),
            "properties": _property_types([c["properties"] for c in cells]),
        }
        if is_edge:
            pairs = Counter(
                (
                    names_by_id.get(int(c["start_id"]) >> LABEL_ID_SHIFT, "?"),
                    names_by_id.get(int(c["end_id"]) >> LABEL_ID_SHIFT, "?"),
                )
                for c in cells
            )
            entry["endpoints"] = [{"from": a, "to": b, "sampled": n} for (a, b), n in pairs.most_common()]
            edges.append(entry)
        else:
            vertices.append(entry)
    return {"graph": graph, "sample_size": sample_size, "vertex_labels": vertices, "edge_labels": edges}

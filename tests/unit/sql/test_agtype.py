"""agtype text parsing. Fixture strings are copied verbatim from psql output of AGE 1.8.0 on PostgreSQL 18."""

import json
import math
from decimal import Decimal

import pytest

from postgres_mcp.sql.agtype import AgtypeSyntaxError
from postgres_mcp.sql.agtype import GraphEdge
from postgres_mcp.sql.agtype import GraphPath
from postgres_mcp.sql.agtype import GraphVertex
from postgres_mcp.sql.agtype import parse_agtype
from postgres_mcp.sql.json_output import OutputLimits
from postgres_mcp.sql.json_output import QueryTable
from postgres_mcp.sql.json_output import ResultColumn
from postgres_mcp.sql.json_output import dumps
from postgres_mcp.sql.json_output import render_table
from postgres_mcp.sql.type_adapters import AGE_LOAD_STATEMENTS
from postgres_mcp.sql.type_adapters import AgtypeLoader
from postgres_mcp.sql.type_adapters import ExtensionTypeCache

VERTEX = (
    '{"id": 844424930131969, "label": "Person", "properties": {"age": 30, "big": 12345678901234600000::numeric, '
    '"name": "Ann \\"x\\"::vertex", "tags": ["a", "b"], "score": 1.5, "nested": {"k": [1, {"z": null}]}}}::vertex'
)
EDGE = '{"id": 1125899906842625, "label": "KNOWS", "end_id": 844424930131970, "start_id": 844424930131969, "properties": {"since": 2020}}::edge'
BOB = '{"id": 844424930131970, "label": "Person", "properties": {"name": "Bob"}}::vertex'
PATH = f"[{VERTEX}, {EDGE}, {BOB}]::path"


def test_vertex_with_numeric_and_suffix_inside_a_string():
    v = parse_agtype(VERTEX)
    assert isinstance(v, GraphVertex)
    assert v.id == 844424930131969 and v.label == "Person"
    assert v.properties["name"] == 'Ann "x"::vertex'  # a suffix inside a string stays data
    assert v.properties["big"] == Decimal("12345678901234600000")
    assert v.properties["score"] == 1.5 and v.properties["age"] == 30
    assert v.properties["nested"] == {"k": [1, {"z": None}]}


def test_edge_and_path():
    e = parse_agtype(EDGE)
    assert e == GraphEdge(id=1125899906842625, label="KNOWS", start_id=844424930131969, end_id=844424930131970, properties={"since": 2020})
    p = parse_agtype(PATH)
    assert isinstance(p, GraphPath)
    assert [type(x) for x in p.elements] == [GraphVertex, GraphEdge, GraphVertex]
    assert p.elements[2].label == "Person"


def test_list_of_vertices_without_path_suffix_is_a_list():
    value = parse_agtype(f"[{VERTEX}, {BOB}]")
    assert isinstance(value, list) and all(isinstance(x, GraphVertex) for x in value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.5::numeric", Decimal("1.5")),
        ("NaN::numeric", Decimal("NaN")),
        ("3.0::numeric", Decimal("3.0")),
        ("3.0", 3.0),
        ("1e+20", 1e20),
        ("0.1", 0.1),
        ("-7", -7),
        ("true", True),
        ('"str"', "str"),
        ("[1, 2.5]", [1, 2.5]),
        ('{"a": "x::edge", "b": "quo\\"te", "c": "tab\\there"}', {"a": "x::edge", "b": 'quo"te', "c": "tab\there"}),
        ('["a", "b"]', ["a", "b"]),
    ],
)
def test_scalars_and_containers(text, expected):
    value = parse_agtype(text)
    assert value == expected or (isinstance(value, Decimal) and value.is_nan() and expected.is_nan())


def test_float_words():
    assert math.isnan(parse_agtype("NaN"))
    assert parse_agtype("Infinity") == math.inf
    assert parse_agtype("-Infinity") == -math.inf


@pytest.mark.parametrize("text", ["", "{", '{"a" 1}', "[1,", "1 2", '{"id": 1}::vertex', "[1]::path", '"x"::numeric', "@"])
def test_malformed_text_raises(text):
    with pytest.raises(AgtypeSyntaxError):
        parse_agtype(text)


def test_unknown_suffix_keeps_the_value():
    assert parse_agtype('"x"::future_type') == "x"


def test_age_session_setup_tries_both_load_forms():
    """Without these, the first cypher() of a session fails on a server that does not preload AGE.

    The Docker test test_first_cypher_call_works_without_preload covers the behaviour; this one
    runs where Docker does not (an emptied tuple shipped once in c863148).
    """
    assert AGE_LOAD_STATEMENTS == ("LOAD 'age'", "LOAD '$libdir/plugins/age'")
    assert ExtensionTypeCache().age_loads == AGE_LOAD_STATEMENTS


def test_loader_keeps_unparseable_text():
    loader = AgtypeLoader(0)
    assert loader.load(b"not agtype @") == "not agtype @"
    assert isinstance(loader.load(VERTEX.encode()), GraphVertex)


def test_json_output_tags_graph_values():
    out = json.loads(dumps([parse_agtype(PATH), parse_agtype("NaN"), parse_agtype("1.5::numeric")]))
    path, nan, num = out
    assert path["kind"] == "path"
    assert [e["kind"] for e in path["elements"]] == ["vertex", "edge", "vertex"]
    assert path["elements"][0]["properties"]["big"] == "12345678901234600000"
    assert path["elements"][1] == {
        "kind": "edge",
        "id": 1125899906842625,
        "label": "KNOWS",
        "start_id": 844424930131969,
        "end_id": 844424930131970,
        "properties": {"since": 2020},
    }
    assert nan == "NaN" and num == "1.5"


def test_render_table_cuts_a_long_vertex_visibly():
    table = QueryTable([ResultColumn("v", "agtype")], [(parse_agtype(VERTEX),)])
    out = json.loads(render_table(table, OutputLimits(max_cell_chars=40)))
    assert out["rows"][0][0].startswith('{"kind": "vertex"') and "[truncated" in out["rows"][0][0]

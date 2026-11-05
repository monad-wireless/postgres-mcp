import datetime as dt
import ipaddress
import json
import uuid
from decimal import Decimal
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import mcp.types as types
import pytest
from psycopg.types.multirange import Multirange
from psycopg.types.range import Range

from postgres_mcp.server import ResponseType
from postgres_mcp.server import execute_sql
from postgres_mcp.server import format_text_response
from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql.json_output import OutputLimits
from postgres_mcp.sql.json_output import QueryTable
from postgres_mcp.sql.json_output import ResultColumn
from postgres_mcp.sql.json_output import dumps
from postgres_mcp.sql.json_output import render_table
from postgres_mcp.sql.type_adapters import SparseVector
from postgres_mcp.sql.type_adapters import Vector


def _text(out: ResponseType) -> str:
    assert isinstance(out[0], types.TextContent)
    return out[0].text


def _table(n: int, more: bool = False) -> QueryTable:
    return QueryTable([ResultColumn("i", "integer")], [(i,) for i in range(n)], more=more)


def _driver(table: QueryTable | None = None) -> Mock:
    driver = Mock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[])
    driver.execute_table = AsyncMock(return_value=table)
    return driver


@pytest.mark.asyncio
async def test_rejection_names_the_reason_and_the_flag():
    safe = SafeSqlDriver(_driver())
    with pytest.raises(ValueError, match=r"Error validating query: .*\nReason: Function pg_read_file is not allowed") as exc:
        await safe.execute_query("SELECT pg_read_file('/etc/passwd')")
    assert "--allow-extension-functions" in str(exc.value)
    assert "Operators are not checked" in str(exc.value)


@pytest.mark.asyncio
async def test_parse_error_names_the_reason():
    safe = SafeSqlDriver(_driver())
    with pytest.raises(ValueError, match=r"Failed to parse SQL statement: .+"):
        await safe.execute_query("SELEC 1")


@pytest.mark.asyncio
async def test_column_definition_lists_in_from_are_accepted():
    inner = _driver()
    safe = SafeSqlDriver(inner)
    await safe.execute_query("""SELECT * FROM json_to_recordset('[{"a":1}]') AS t(a int)""")
    await safe.execute_query("SELECT * FROM ROWS FROM (generate_series(1,2), generate_series(3,4)) AS t(a,b)")
    assert inner.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_safe_execute_table_validates_and_forces_read_only():
    inner = _driver(_table(1))
    safe = SafeSqlDriver(inner)
    with pytest.raises(ValueError, match="not allowed"):
        await safe.execute_table("SELECT pg_read_file('/etc/passwd')")
    await safe.execute_table("SELECT 1", force_readonly=False, max_rows=7)
    inner.execute_table.assert_awaited_once()
    assert inner.execute_table.await_args.kwargs == {"params": None, "force_readonly": True, "max_rows": 7}
    assert inner.execute_table.await_args.args[0].startswith("/* crystaldba */ ")


@pytest.mark.asyncio
async def test_execute_sql_returns_json_columns_and_rows():
    with (
        patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver(_table(5)))),
        patch("postgres_mcp.server.output_limits", OutputLimits()),
    ):
        out = json.loads(_text(await execute_sql(sql="SELECT 1")))
    assert out == {"columns": [{"name": "i", "type": "integer"}], "row_count": 5, "rows": [[0], [1], [2], [3], [4]]}


@pytest.mark.asyncio
async def test_execute_sql_passes_max_rows_and_says_when_more_exist():
    driver = _driver(_table(2, more=True))
    with (
        patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=driver)),
        patch("postgres_mcp.server.output_limits", OutputLimits(max_rows=2)),
    ):
        out = json.loads(_text(await execute_sql(sql="SELECT 1")))
    assert driver.execute_table.await_args.kwargs == {"max_rows": 2}
    assert out["row_count"] == 2
    assert "more rows exist" in out["note"] and "--max-rows" in out["note"]


@pytest.mark.asyncio
async def test_execute_sql_without_rows_says_so():
    with patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver(None))):
        assert _text(await execute_sql(sql="SET x = 1")) == "No results"


def test_duplicate_column_names_are_all_kept():
    table = QueryTable([ResultColumn("a", "integer"), ResultColumn("a", "integer")], [(1, 2)])
    out = json.loads(render_table(table, OutputLimits()))
    assert [c["name"] for c in out["columns"]] == ["a", "a"]
    assert out["rows"] == [[1, 2]]


def test_long_cells_are_cut_and_marked():
    table = QueryTable([ResultColumn("repeat", "text")], [("z" * 200_000,)])
    cell = json.loads(render_table(table, OutputLimits(max_cell_chars=2000)))["rows"][0][0]
    assert cell == "z" * 2000 + "…[truncated 198000 chars]"


def test_long_json_cells_become_marked_text():
    table = QueryTable([ResultColumn("j", "jsonb")], [({"k": "v" * 50},)])
    cell = json.loads(render_table(table, OutputLimits(max_cell_chars=20)))["rows"][0][0]
    assert cell.startswith('{"k": "vvvv') and "…[truncated" in cell


def test_byte_budget_drops_rows_and_says_so():
    table = QueryTable([ResultColumn("t", "text")], [("x" * 1000,) for _ in range(50)])
    text = render_table(table, OutputLimits(max_result_bytes=10_000, max_cell_chars=None))
    out = json.loads(text)
    assert len(text.encode()) <= 10_000
    assert 0 < out["row_count"] == len(out["rows"]) < 50
    assert "--max-result-bytes" in out["note"]


def test_type_probe_row_is_valid_json():
    """Every Python type psycopg returns for the acceptance type probe, encoded without a repr."""
    tz = dt.timezone(dt.timedelta(hours=2))
    row = (
        Decimal("1.1000"),
        Decimal("NaN"),
        float("inf"),
        float("-inf"),
        float("nan"),
        b"\xde\xad\xbe\xef",
        memoryview(b"\x01"),
        ["a", None],
        {"a": [1, {"b": None}]},
        ipaddress.ip_network("192.168.1.0/24"),
        ipaddress.ip_interface("10.0.0.1/32"),
        dt.time(12, 34, 56, tzinfo=tz),
        dt.datetime(2026, 1, 1, 10, tzinfo=tz),
        dt.date(2026, 1, 1),
        Range(1, 10, "[)"),
        Multirange([Range(1, 3), Range(5, 7)]),
        uuid.UUID("8c93a5a2-0000-4000-8000-000000000000"),
        ("1", "x"),
        True,
        None,
        Vector((1.5, 2.0, 3.0)),
        SparseVector(5, (1, 3), (1.5, 2.0)),
    )
    out = json.loads(dumps(list(row)))
    assert out == [
        "1.1000",
        "NaN",
        "Infinity",
        "-Infinity",
        "NaN",
        "\\xdeadbeef",
        "\\x01",
        ["a", None],
        {"a": [1, {"b": None}]},
        "192.168.1.0/24",
        "10.0.0.1/32",
        "12:34:56+02:00",
        "2026-01-01T10:00:00+02:00",
        "2026-01-01",
        {"lower": 1, "upper": 10, "bounds": "[)", "empty": False},
        [{"lower": 1, "upper": 3, "bounds": "[)", "empty": False}, {"lower": 5, "upper": 7, "bounds": "[)", "empty": False}],
        "8c93a5a2-0000-4000-8000-000000000000",
        ["1", "x"],
        True,
        None,
        [1.5, 2.0, 3.0],
        {"dim": 5, "indices": [1, 3], "values": [1.5, 2.0]},
    ]


def test_vector_preview_bounds_long_vectors():
    v = Vector(tuple(float(i) for i in range(768)))
    assert json.loads(dumps(v, vector_preview=8)) == {"dims": 768, "preview": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]}
    assert len(json.loads(dumps(v, vector_preview=None))) == 768


def test_vector_text_forms_parse():
    assert Vector.parse("[1.5,2,3]") == Vector((1.5, 2.0, 3.0))
    assert Vector.parse("[]") == Vector(())
    assert SparseVector.parse("{1:1.5,3:2}/5") == SparseVector(5, (1, 3), (1.5, 2.0))
    assert SparseVector.parse("{}/3") == SparseVector(3, (), ())


def test_format_text_response_passes_strings_and_serialises_the_rest():
    assert _text(format_text_response("plain")) == "plain"
    assert json.loads(_text(format_text_response([{"n": Decimal("2.50"), "ok": True}]))) == [{"n": "2.50", "ok": True}]

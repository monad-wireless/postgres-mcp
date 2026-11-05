"""Acceptance tests on the docker Postgres fixture (12, 15, 16), through execute_sql in restricted mode.

Every response must parse as JSON. pgvector is not in these images: its loader is covered by unit
tests on the text form (test_query_output.py), not here.
"""

import json
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import mcp.types as types
import pytest
import pytest_asyncio

import postgres_mcp.server as server
from postgres_mcp.databases import DatabaseRegistry
from postgres_mcp.server import AccessMode
from postgres_mcp.sql import PoolSettings
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql.json_output import OutputLimits

SETUP = [
    "CREATE EXTENSION IF NOT EXISTS hstore",
    'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
    "CREATE EXTENSION IF NOT EXISTS unaccent",
    "CREATE TABLE IF NOT EXISTS entries (id int PRIMARY KEY, identifiers hstore)",
    """INSERT INTO entries VALUES (1, '"doi"=>"10.1201/x", "isbn"=>"978-0"') ON CONFLICT DO NOTHING""",
]


def _major(version: str) -> int:
    return int(version.split(":")[-1])


@contextmanager
def _restricted(url: str):
    registry = DatabaseRegistry([("t", url)], settings=PoolSettings(superuser="warn"), allowed_extensions=("*",))
    with (
        patch("postgres_mcp.server.databases", registry),
        patch("postgres_mcp.server.current_access_mode", AccessMode.RESTRICTED),
        patch("postgres_mcp.server.output_limits", OutputLimits()),
    ):
        yield registry


async def _sql(sql: str) -> dict[str, Any]:
    out = await server.execute_sql(sql=sql, database=None)
    assert isinstance(out[0], types.TextContent)
    text = out[0].text
    assert not text.startswith("Error"), text
    return json.loads(text)


class TestLive:
    @pytest_asyncio.fixture
    async def url(self, test_postgres_connection_string):
        url, version = test_postgres_connection_string
        admin = SqlDriver(engine_url=url)
        for statement in SETUP:
            await admin.execute_query(statement)  # type: ignore[arg-type]
        return url, _major(version)

    @pytest.mark.asyncio
    async def test_forced_readonly_transaction_is_read_only(self, test_postgres_connection_string):
        url, _ = test_postgres_connection_string
        rows = await SqlDriver(engine_url=url).execute_query("SHOW transaction_read_only", force_readonly=True)
        assert rows is not None
        assert rows[0].cells["transaction_read_only"] == "on"

    @pytest.mark.asyncio
    async def test_type_probe_is_json(self, url):
        url, major = url
        extra = []
        if major >= 13:
            extra.append("gen_random_uuid() AS uuid")
        if major >= 14:
            extra.append("'{[1,3),[5,7)}'::int4multirange AS mrng")
        sql = (
            "SELECT 1.10::numeric(12,4) AS num, 'NaN'::numeric AS num_nan, 'Infinity'::float8 AS inf, "
            "'\\xDEADBEEF'::bytea AS bytea, ARRAY['a',NULL]::text[] AS arr, "
            """'{"a":[1,{"b":null}]}'::jsonb AS jsonb, '<x>1</x>'::xml AS xml, """
            "'192.168.1.0/24'::cidr AS cidr, '10.0.0.1'::inet AS inet, '08:00:2b:01:02:03'::macaddr AS mac, "
            "'12:34:56+02'::timetz AS ttz, '2026-01-01 10:00+02'::timestamptz AS tstz, "
            "'[1,10)'::int4range AS rng, '(1,2)'::point AS pt, 'a fat cat'::tsvector AS tsv, B'1010'::bit(4) AS bits, "
            "'$1.50'::money AS money, ROW(1,'x') AS rec" + "".join(f", {e}" for e in extra)
        )
        with _restricted(url):
            out = await _sql(sql)
        row = dict(zip([c["name"] for c in out["columns"]], out["rows"][0], strict=True))
        assert row["num"] == "1.1000" and row["num_nan"] == "NaN" and row["inf"] == "Infinity"
        assert row["bytea"] == "\\xdeadbeef"
        assert row["arr"] == ["a", None]
        assert row["jsonb"] == {"a": [1, {"b": None}]}
        assert row["cidr"] == "192.168.1.0/24"
        assert row["rng"] == {"lower": 1, "upper": 10, "bounds": "[)", "empty": False}
        assert row["tstz"].startswith("2026-01-01T")
        assert [c["type"] for c in out["columns"]][:2] == ["numeric(12,4)", "numeric"]
        if major >= 14:
            assert len(row["mrng"]) == 2

    @pytest.mark.asyncio
    async def test_infinity_and_intervals_survive(self, url):
        url, _ = url
        with _restricted(url):
            out = await _sql("SELECT 'infinity'::date, '-infinity'::timestamptz, 'infinity'::timestamp, '1 mon'::interval, '1 year 2 days'::interval")
        assert out["rows"] == [["infinity", "-infinity", "infinity", "1 mon", "1 year 2 days"]]

    @pytest.mark.asyncio
    async def test_duplicate_column_names_are_kept(self, url):
        url, _ = url
        with _restricted(url):
            out = await _sql("SELECT 1 AS a, 2 AS a, count(*), count(*) FROM pg_class")
        assert [c["name"] for c in out["columns"]] == ["a", "a", "count", "count"]
        assert out["rows"][0][:2] == [1, 2]

    @pytest.mark.asyncio
    async def test_long_cell_is_cut(self, url):
        url, _ = url
        with _restricted(url):
            out = await _sql("SELECT repeat('z', 200000)")
        assert out["rows"][0][0] == "z" * 2000 + "…[truncated 198000 chars]"

    @pytest.mark.asyncio
    async def test_max_rows_says_more_exist(self, url):
        url, _ = url
        with _restricted(url), patch("postgres_mcp.server.output_limits", OutputLimits(max_rows=3)):
            out = await _sql("SELECT g FROM generate_series(1, 10) AS g")
        assert out["rows"] == [[1], [2], [3]]
        assert "more rows exist" in out["note"]

    @pytest.mark.asyncio
    async def test_extension_functions_and_types(self, url):
        url, _ = url
        with _restricted(url) as registry:
            uuid_out = await _sql("SELECT uuid_generate_v5(uuid_ns_url(), 'x')")
            hstore_out = await _sql("SELECT identifiers, akeys(identifiers), unaccent('Žluťoučký kůň') FROM entries LIMIT 1")
            described = json.loads(_text(await server.list_databases()))
        assert len(uuid_out["rows"][0][0]) == 36
        identifiers, keys, plain = hstore_out["rows"][0]
        assert identifiers == {"doi": "10.1201/x", "isbn": "978-0"}
        assert sorted(keys) == ["doi", "isbn"]
        assert plain == "Zlutoucky kun"
        assert described[0]["extension_functions"] > 0
        assert described[0]["superuser"] is True  # the fixture logs in as postgres
        admitted = registry.get().extension_functions
        assert admitted is not None and "akeys" in admitted


def _text(out) -> str:
    assert isinstance(out[0], types.TextContent)
    return out[0].text

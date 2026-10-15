from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql import discover_read_only_functions
from postgres_mcp.sql import parse_extension_list
from postgres_mcp.sql.extension_functions import DISCOVERY_QUERY

SPATIAL_QUERY = """
SELECT s.segment_index,
       extract(epoch FROM getTimestamp(i)) AS t,
       ST_X(getValue(i)) AS x, ST_Y(getValue(i)) AS y
FROM trajectory.segment s, unnest(instants(s.trip)) AS i
WHERE ST_Intersects(s.line_m, ST_MakeEnvelope(0, 0, 10, 10, 900010))
"""


def _driver() -> Mock:
    driver = Mock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[])
    return driver


def test_parse_extension_list():
    assert parse_extension_list(" MobilityDB, postgis ,,postgis") == ("mobilitydb", "postgis")
    assert parse_extension_list("") == ()
    assert parse_extension_list(None) == ()
    with pytest.raises(ValueError):
        parse_extension_list("postgis; DROP TABLE x")


@pytest.mark.asyncio
async def test_extension_functions_rejected_without_opt_in():
    safe = SafeSqlDriver(_driver())
    with pytest.raises(ValueError, match="Error validating query"):
        await safe.execute_query(SPATIAL_QUERY)


@pytest.mark.asyncio
async def test_extension_functions_accepted_with_opt_in_and_still_read_only():
    inner = _driver()
    extra = frozenset({"gettimestamp", "getvalue", "st_x", "st_y", "instants", "st_intersects", "st_makeenvelope"})
    safe = SafeSqlDriver(inner, extra_functions=extra)
    await safe.execute_query(SPATIAL_QUERY)
    assert inner.execute_query.await_args.kwargs["force_readonly"] is True


@pytest.mark.asyncio
async def test_opt_in_does_not_admit_other_functions_or_statements():
    safe = SafeSqlDriver(_driver(), extra_functions=frozenset({"st_x"}))
    for query in (
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT AddGeometryColumn('t', 'g', 4326, 'POINT', 2)",
        "UPDATE t SET g = ST_X(g)",
    ):
        with pytest.raises(ValueError):
            await safe.execute_query(query)


@pytest.mark.asyncio
async def test_functions_in_from_clause_are_validated():
    """RangeFunction nests FuncCall two levels deep; it used to escape validation entirely."""
    safe = SafeSqlDriver(_driver())
    for query in (
        "SELECT * FROM pg_read_file('/etc/passwd')",
        "SELECT * FROM pgr_dijkstra('SELECT 1', 1, 2)",
        "SELECT * FROM generate_series(1, 3) g, pg_ls_dir('/') d",
    ):
        with pytest.raises(ValueError, match="Error validating query"):
            await safe.execute_query(query)
    await safe.execute_query("SELECT * FROM generate_series(1, 3) AS g")


@pytest.mark.asyncio
async def test_discovery_reads_catalog_read_only_and_returns_names():
    inner = _driver()
    inner.execute_query = AsyncMock(return_value=[SqlDriver.RowResult(cells={"name": "st_x"}), SqlDriver.RowResult(cells={"name": "tdwithin"})])
    names = await discover_read_only_functions(inner, ("mobilitydb", "postgis"))
    assert names == frozenset({"st_x", "tdwithin"})
    args = inner.execute_query.await_args
    assert args.kwargs["force_readonly"] is True
    assert args.kwargs["params"] == [["mobilitydb", "postgis"]]


@pytest.mark.asyncio
async def test_discovery_without_extensions_queries_nothing():
    inner = _driver()
    assert await discover_read_only_functions(inner, ()) == frozenset()
    inner.execute_query.assert_not_awaited()


def test_discovery_query_excludes_unsafe_overloads():
    q = DISCOVERY_QUERY
    assert "provolatile = 'v'" in q and "prosecdef" in q and "prokind <> 'f'" in q
    assert "NOT EXISTS" in q  # one unsafe overload anywhere keeps the name out

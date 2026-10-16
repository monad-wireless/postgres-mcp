from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from postgres_mcp.server import execute_sql
from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver


def _rows(n: int) -> list[SqlDriver.RowResult]:
    return [SqlDriver.RowResult(cells={"i": i}) for i in range(n)]


def _driver(rows: list[SqlDriver.RowResult]) -> Mock:
    driver = Mock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=rows)
    return driver


@pytest.mark.asyncio
async def test_rejection_names_the_reason():
    safe = SafeSqlDriver(_driver([]))
    with pytest.raises(ValueError, match=r"Error validating query: .*\nReason: Function pg_read_file is not allowed"):
        await safe.execute_query("SELECT pg_read_file('/etc/passwd')")


@pytest.mark.asyncio
async def test_parse_error_names_the_reason():
    safe = SafeSqlDriver(_driver([]))
    with pytest.raises(ValueError, match=r"Failed to parse SQL statement: .+"):
        await safe.execute_query("SELEC 1")


@pytest.mark.asyncio
async def test_column_definition_lists_in_from_are_accepted():
    inner = _driver([])
    safe = SafeSqlDriver(inner)
    await safe.execute_query("""SELECT * FROM json_to_recordset('[{"a":1}]') AS t(a int)""")
    await safe.execute_query("SELECT * FROM ROWS FROM (generate_series(1,2), generate_series(3,4)) AS t(a,b)")
    assert inner.execute_query.await_count == 2


@pytest.mark.asyncio
async def test_execute_sql_returns_every_row_by_default():
    with (
        patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver(_rows(5)))),
        patch("postgres_mcp.server.max_result_rows", None),
    ):
        out = await execute_sql(sql="SELECT 1")
    assert "truncated" not in out[0].text
    assert out[0].text.count("'i'") == 5


@pytest.mark.asyncio
async def test_execute_sql_truncates_and_says_so_with_max_rows():
    with (
        patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver(_rows(5)))),
        patch("postgres_mcp.server.max_result_rows", 2),
    ):
        out = await execute_sql(sql="SELECT 1")
    text = out[0].text
    assert text.count("'i'") == 2
    assert "(truncated to 2 of 5 rows by --max-rows)" in text


@pytest.mark.asyncio
async def test_execute_sql_under_the_cap_is_unchanged():
    with (
        patch("postgres_mcp.server.get_sql_driver", AsyncMock(return_value=_driver(_rows(2)))),
        patch("postgres_mcp.server.max_result_rows", 2),
    ):
        out = await execute_sql(sql="SELECT 1")
    assert "truncated" not in out[0].text

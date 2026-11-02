import os
import re
import sys
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import mcp.types as types
import pytest

import postgres_mcp.server as server
from postgres_mcp.databases import DatabaseRegistry
from postgres_mcp.databases import UnknownDatabaseError
from postgres_mcp.databases import collect_database_urls
from postgres_mcp.databases import parse_database_spec
from postgres_mcp.server import AccessMode
from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver
from postgres_mcp.sql.sql_driver import DbConnPool

GIS = "postgresql://monad:secret@db.example:5432/monad_gis"
LAKE = "postgresql://monad:secret@db.example:5432/monad_lake"


def _text(out: server.ResponseType) -> str:
    assert isinstance(out[0], types.TextContent)
    return out[0].text


def test_parse_database_spec_splits_on_first_equals():
    assert parse_database_spec("gis=postgresql://u:p@h/db?sslmode=require") == ("gis", "postgresql://u:p@h/db?sslmode=require")


@pytest.mark.parametrize("spec", ["gis", "gis=", "=postgresql://h/db", "GIS=postgresql://h/db", "a b=postgresql://h/db"])
def test_parse_database_spec_rejects_bad_values(spec):
    with pytest.raises(ValueError):
        parse_database_spec(spec)


def test_parse_database_spec_error_hides_password():
    with pytest.raises(ValueError) as exc:
        parse_database_spec("BAD=postgresql://u:hunter2@h/db")
    assert "hunter2" not in str(exc.value)


def test_collect_orders_legacy_then_cli_then_sorted_env():
    pairs = collect_database_urls(
        legacy_url=GIS,
        cli_specs=["lake=" + LAKE],
        environ={"DATABASE_URI_ZETA": "postgresql://h/z", "DATABASE_URI_ALPHA": "postgresql://h/a", "DATABASE_URI": GIS, "OTHER": "x"},
    )
    assert pairs == [("default", GIS), ("lake", LAKE), ("alpha", "postgresql://h/a"), ("zeta", "postgresql://h/z")]


def test_collect_rejects_duplicate_names():
    with pytest.raises(ValueError, match="more than once"):
        collect_database_urls(legacy_url=None, cli_specs=["lake=" + LAKE], environ={"DATABASE_URI_LAKE": LAKE})


def test_collect_ignores_blank_env_values():
    assert collect_database_urls(legacy_url=None, cli_specs=[], environ={"DATABASE_URI_LAKE": "  "}) == []


def test_registry_default_is_first_unless_named():
    assert DatabaseRegistry([("gis", GIS), ("lake", LAKE)]).default_name == "gis"
    assert DatabaseRegistry([("gis", GIS), ("lake", LAKE)], default="lake").default_name == "lake"
    with pytest.raises(ValueError, match="not configured"):
        DatabaseRegistry([("gis", GIS)], default="lake")


def test_registry_get_selects_by_name_and_names_the_options_when_unknown():
    registry = DatabaseRegistry([("gis", GIS), ("lake", LAKE)])
    assert registry.get().name == "gis"
    assert registry.get(None).name == "gis"
    assert registry.get("lake").name == "lake"
    with pytest.raises(UnknownDatabaseError, match=re.escape("Unknown database 'nope'. Configured databases: gis, lake")):
        registry.get("nope")


def test_empty_registry_get_raises():
    with pytest.raises(UnknownDatabaseError, match="No database is configured"):
        DatabaseRegistry().get()


def test_describe_has_location_and_no_credentials():
    registry = DatabaseRegistry([("gis", GIS), ("lake", LAKE)])
    described = registry.describe()
    assert described[0] == {
        "name": "gis",
        "default": True,
        "host": "db.example",
        "port": 5432,
        "dbname": "monad_gis",
        "connected": False,
        "extension_functions": 0,
    }
    assert described[1]["default"] is False
    assert "secret" not in repr(described)


@pytest.mark.asyncio
async def test_connect_all_isolates_failures_and_discovers_per_database():
    registry = DatabaseRegistry([("gis", GIS), ("lake", LAKE)])

    async def pool_connect(self, connection_url=None):
        if self.connection_url == LAKE:
            raise ValueError("Connection attempt failed: postgresql://monad:secret@db.example/monad_lake")
        return MagicMock()

    discover = AsyncMock(return_value=frozenset({"st_area"}))
    with (
        patch.object(DbConnPool, "pool_connect", pool_connect),
        patch("postgres_mcp.databases.discover_read_only_functions", discover),
    ):
        await registry.connect_all(("postgis",))

    assert registry.get("gis").extension_functions == frozenset({"st_area"})
    assert registry.get("lake").extension_functions == frozenset()
    assert discover.await_count == 1


@pytest.mark.asyncio
async def test_connect_all_skips_discovery_without_extensions():
    registry = DatabaseRegistry([("gis", GIS)])
    discover = AsyncMock()
    with (
        patch.object(DbConnPool, "pool_connect", AsyncMock()),
        patch("postgres_mcp.databases.discover_read_only_functions", discover),
    ):
        await registry.connect_all(())
    discover.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_sql_driver_uses_the_named_database_pool_and_allowlist():
    registry = DatabaseRegistry([("gis", GIS), ("lake", LAKE)])
    registry.get("gis").extension_functions = frozenset({"st_area"})
    with (
        patch("postgres_mcp.server.databases", registry),
        patch("postgres_mcp.server.current_access_mode", AccessMode.RESTRICTED),
    ):
        gis = await server.get_sql_driver("gis")
        lake = await server.get_sql_driver("lake")
        default = await server.get_sql_driver()

    assert isinstance(gis, SafeSqlDriver) and isinstance(lake, SafeSqlDriver) and isinstance(default, SafeSqlDriver)
    assert isinstance(gis.sql_driver, SqlDriver) and gis.sql_driver.conn is registry.get("gis").pool
    assert isinstance(lake.sql_driver, SqlDriver) and lake.sql_driver.conn is registry.get("lake").pool
    assert isinstance(default.sql_driver, SqlDriver) and default.sql_driver.conn is registry.get("gis").pool
    assert gis.extra_functions == frozenset({"st_area"})
    assert lake.extra_functions == frozenset()


@pytest.mark.asyncio
async def test_execute_sql_routes_the_database_argument():
    driver = MagicMock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[SqlDriver.RowResult(cells={"x": 1})])
    get_driver = AsyncMock(return_value=driver)
    with patch("postgres_mcp.server.get_sql_driver", get_driver):
        await server.execute_sql(sql="SELECT 1", database="lake")
        await server.execute_sql(sql="SELECT 1")
    assert [c.args for c in get_driver.await_args_list] == [("lake",), (None,)]


@pytest.mark.asyncio
async def test_execute_sql_reports_an_unknown_database():
    with patch("postgres_mcp.server.databases", DatabaseRegistry([("gis", GIS)])):
        out = await server.execute_sql(sql="SELECT 1", database="lake")
    assert _text(out) == "Error: Unknown database 'lake'. Configured databases: gis"


@pytest.mark.asyncio
async def test_list_databases_tool():
    with patch("postgres_mcp.server.databases", DatabaseRegistry([("gis", GIS), ("lake", LAKE)])):
        out = await server.list_databases()
    text = _text(out)
    assert "'name': 'gis'" in text and "'name': 'lake'" in text
    assert "secret" not in text


@pytest.mark.asyncio
async def test_main_builds_the_registry_from_flags_and_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URI", raising=False)
    monkeypatch.setenv("DATABASE_URI_ARCHIVE", "postgresql://h/archive")
    monkeypatch.setattr(
        sys,
        "argv",
        ["postgres_mcp", "--database", "gis=" + GIS, "--database", "lake=" + LAKE, "--default-database", "lake"],
    )
    original = server.databases
    try:
        with (
            patch.object(DbConnPool, "pool_connect", AsyncMock()),
            patch("postgres_mcp.server.mcp.run_stdio_async", AsyncMock()),
        ):
            await server.main()
        assert server.databases.names == ["gis", "lake", "archive"]
        assert server.databases.default_name == "lake"
    finally:
        server.databases = original


@pytest.mark.asyncio
async def test_main_without_any_database_fails(monkeypatch):
    monkeypatch.delenv("DATABASE_URI", raising=False)
    for key in [k for k in list(os.environ) if k.startswith("DATABASE_URI_")]:
        monkeypatch.delenv(key)
    monkeypatch.setattr(sys, "argv", ["postgres_mcp"])
    original = server.databases
    try:
        with pytest.raises(ValueError, match="No database URL provided"):
            await server.main()
    finally:
        server.databases = original

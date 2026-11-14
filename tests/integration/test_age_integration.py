"""Apache AGE end to end on the apache/age image (AGE 1.8.0, PostgreSQL 18), through the server's tools.

Two containers: the image default (``shared_preload_libraries=age``), and the same image with
preload switched off, where the first ``cypher()`` of a session fails unless the connection setup
ran ``LOAD 'age'``. Restricted-mode tests log in as a non-superuser that holds only USAGE and
SELECT grants.
"""

import json
import time
from collections.abc import Generator
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

AGE_IMAGE = "apache/age@sha256:cd6e62c12924b0e04f69f5f23326c94a55ac13e77a66c7594da1c4f4328a397d"

SETUP = [
    "CREATE EXTENSION IF NOT EXISTS age",
    # One string: create_graph needs ag_catalog on search_path in the same session (AGE 1.8.0).
    "SET search_path = \"$user\", public, ag_catalog; SELECT * FROM ag_catalog.create_graph('people')",
    """SELECT * FROM ag_catalog.cypher('people', $$
         CREATE (a:Person {name: 'Ann', age: 30, score: 1.5})-[:KNOWS {since: 2020}]->(b:Person {name: 'Bob'}),
                (a)-[:WORKS_AT]->(:Company {name: 'FIIT'})
       $$) AS (v ag_catalog.agtype)""",
    "CREATE ROLE reader LOGIN PASSWORD 'reader'",
    "GRANT USAGE ON SCHEMA ag_catalog, people TO reader",
    "GRANT SELECT ON ALL TABLES IN SCHEMA ag_catalog, people TO reader",
]


def _container(command: list[str] | None) -> Generator[str, None, None]:
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception:
        pytest.skip("Docker is not available")
    container = client.containers.run(
        AGE_IMAGE,
        command=command,
        environment={"POSTGRES_PASSWORD": "probe", "POSTGRES_DB": "probe"},
        ports={"5432/tcp": ("127.0.0.1", 0)},
        detach=True,
    )
    try:
        deadline = time.time() + 90
        while time.time() < deadline:
            # pg_isready on the TCP socket: the image's first start runs initdb on a unix socket only.
            if container.exec_run(["pg_isready", "-h", "127.0.0.1", "-U", "postgres"]).exit_code == 0:
                break
            time.sleep(1)
        else:
            pytest.skip("AGE container did not become ready")
        container.reload()
        port = container.ports["5432/tcp"][0]["HostPort"]
        yield f"postgresql://postgres:probe@127.0.0.1:{port}/probe"
    finally:
        container.remove(force=True, v=True)


@pytest.fixture(scope="module")
def preloaded_url() -> Generator[str, None, None]:
    yield from _container(None)


@pytest.fixture(scope="module")
def unloaded_url() -> Generator[str, None, None]:
    yield from _container(["postgres", "-c", "shared_preload_libraries="])


async def _setup(url: str) -> None:
    admin = SqlDriver(engine_url=url)
    rows = await admin.execute_query("SELECT count(*) AS n FROM pg_catalog.pg_roles WHERE rolname = 'reader'")
    if rows and rows[0].cells["n"]:
        return
    for statement in SETUP:
        await admin.execute_query(statement)  # type: ignore[arg-type]


@contextmanager
def _server(url: str, mode: AccessMode, extensions: tuple[str, ...] = ()):
    registry = DatabaseRegistry([("t", url)], settings=PoolSettings(superuser="allow"), allowed_extensions=extensions)
    with (
        patch("postgres_mcp.server.databases", registry),
        patch("postgres_mcp.server.current_access_mode", mode),
        patch("postgres_mcp.server.output_limits", OutputLimits()),
    ):
        yield registry


def _text(out) -> str:
    assert isinstance(out[0], types.TextContent)
    return out[0].text


def _json(out) -> Any:
    text = _text(out)
    assert not text.startswith("Error"), text
    return json.loads(text)


async def _cypher(graph: str, query: str, columns: list[str], params: dict[str, Any] | None = None):
    return await server.execute_cypher(graph=graph, query=query, columns=columns, params=params, database=None)


def _reader(url: str) -> str:
    return url.replace("postgres:probe@", "reader:reader@")


class TestRestrictedReader:
    @pytest_asyncio.fixture
    async def url(self, preloaded_url):
        await _setup(preloaded_url)
        return _reader(preloaded_url)

    @pytest.mark.asyncio
    async def test_execute_cypher_reads_with_params_and_tags_graph_values(self, url):
        with _server(url, AccessMode.RESTRICTED, ("age",)) as registry:
            out = _json(
                await _cypher(
                    "people",
                    "MATCH p = (a:Person)-[e:KNOWS]->(b) WHERE a.name = $name RETURN a, e, p, a.score, b.name",
                    ["a", "e", "p", "score", "friend"],
                    {"name": "Ann"},
                )
            )
            described = _json(await server.list_databases())
        assert registry.get().cypher is True and described[0]["cypher"] is True
        assert [c["type"] for c in out["columns"]] == ["agtype"] * 5
        a, e, p, score, friend = out["rows"][0]
        assert a["kind"] == "vertex" and a["label"] == "Person" and a["properties"]["name"] == "Ann"
        assert e["kind"] == "edge" and e["label"] == "KNOWS" and e["start_id"] == a["id"] and e["properties"] == {"since": 2020}
        assert p["kind"] == "path" and [x["kind"] for x in p["elements"]] == ["vertex", "edge", "vertex"]
        assert score == 1.5 and friend == "Bob"

    @pytest.mark.asyncio
    async def test_execute_sql_cypher_comparison_needs_the_search_path_setup(self, url):
        with _server(url, AccessMode.RESTRICTED, ("age",)):
            out = _json(await server.execute_sql(sql="SELECT * FROM cypher('people', $$ MATCH (n:Person) WHERE n.age = 30 RETURN n.name $$) AS (v agtype)", database=None))
        assert out["rows"] == [["Ann"]]

    @pytest.mark.asyncio
    async def test_writes_and_sql_functions_are_refused(self, url):
        with _server(url, AccessMode.RESTRICTED, ("age",)):
            create = _text(await _cypher("people", "CREATE (x:Ghost) RETURN x", ["x"]))
            backend = _text(await _cypher("people", "RETURN pg_catalog.pg_backend_pid()", ["v"]))
            graph = _text(await server.execute_sql(sql="SELECT * FROM ag_catalog.create_graph('other')", database=None))
            count = _json(await _cypher("people", "MATCH (n) RETURN count(n)", ["n"]))
        assert create.startswith("Error") and "CREATE" in create
        assert backend.startswith("Error") and "schema-qualified" in backend
        assert graph.startswith("Error")
        assert count["rows"] == [[3]]

    @pytest.mark.asyncio
    async def test_read_only_transaction_blocks_cypher_writes_without_the_guard(self, url):
        """The second barrier: AGE's writes fail in the forced read-only transaction."""
        driver = SqlDriver(engine_url=url.replace("reader:reader@", "postgres:probe@"))
        with pytest.raises(Exception, match="read-only transaction"):
            await driver.execute_query(
                "SELECT * FROM ag_catalog.cypher('people', $$ CREATE (x:Ghost) RETURN x $$) AS (v ag_catalog.agtype)", force_readonly=True
            )

    @pytest.mark.asyncio
    async def test_cypher_refused_without_age_opt_in(self, url):
        with _server(url, AccessMode.RESTRICTED, ("pg_trgm",)):
            out = _text(await _cypher("people", "MATCH (n) RETURN n", ["n"]))
        assert out.startswith("Error") and "allow-extension-functions age" in out

    @pytest.mark.asyncio
    async def test_list_graphs_and_schema(self, url):
        with _server(url, AccessMode.RESTRICTED, ("age",)):
            graphs = _json(await server.list_graphs(database=None))
            schema = _json(await server.get_graph_schema(graph="people", sample_size=50, database=None))
            missing = _text(await server.get_graph_schema(graph="nope", sample_size=50, database=None))
        assert graphs == [{"graph": "people", "schema": "people", "vertex_labels": 2, "edge_labels": 2}]
        vertices = {v["label"]: v for v in schema["vertex_labels"]}
        edges = {e["label"]: e for e in schema["edge_labels"]}
        assert set(vertices) == {"Person", "Company"} and set(edges) == {"KNOWS", "WORKS_AT"}
        assert vertices["Person"]["sampled"] == 2
        assert vertices["Person"]["properties"] == {"age": ["integer"], "name": ["string"], "score": ["float"]}
        assert edges["KNOWS"]["endpoints"] == [{"from": "Person", "to": "Person", "sampled": 1}]
        assert edges["WORKS_AT"]["endpoints"] == [{"from": "Person", "to": "Company", "sampled": 1}]
        assert missing.startswith("Error") and "does not exist" in missing


class TestUnrestricted:
    @pytest.mark.asyncio
    async def test_execute_cypher_writes_with_params(self, preloaded_url):
        await _setup(preloaded_url)
        with _server(preloaded_url, AccessMode.UNRESTRICTED):
            created = _json(await _cypher("people", "CREATE (c:City {name: $name, pct: 10 % 3}) RETURN c", ["c"], {"name": "Bratislava"}))
            found = _json(await _cypher("people", "MATCH (c:City) RETURN c.name, c.pct", ["name", "pct"]))
        assert created["rows"][0][0]["label"] == "City"
        assert found["rows"] == [["Bratislava", 1]]

    @pytest.mark.asyncio
    async def test_first_cypher_call_works_without_preload(self, unloaded_url):
        await _setup(unloaded_url)
        # A fresh registry means a fresh pool, so this is the first cypher() of its session.
        with _server(unloaded_url, AccessMode.UNRESTRICTED):
            out = _json(await _cypher("people", "MATCH (n:Person) WHERE n.age = 30 RETURN n.name", ["name"]))
        assert out["rows"] == [["Ann"]]

    @pytest.mark.asyncio
    async def test_execute_cypher_without_age_says_so(self, unloaded_url):
        url = unloaded_url.rsplit("/", 1)[0] + "/postgres"  # the maintenance database has no AGE
        with _server(url, AccessMode.UNRESTRICTED):
            out = _text(await _cypher("people", "RETURN 1", ["v"]))
        assert out.startswith("Error") and "not installed" in out

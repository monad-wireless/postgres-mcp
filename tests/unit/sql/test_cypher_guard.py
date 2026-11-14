"""The restricted-mode Cypher guard, and how SafeSqlDriver applies it to cypher() calls.

The accepted queries all ran on AGE 1.8.0; the refused ones cover the escape routes found there:
writes, schema-qualified SQL functions, and bare names that resolve through search_path.
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from postgres_mcp.graph import build_cypher_statement
from postgres_mcp.sql import CypherGuard
from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver

# A slice of the age_* names that discovery admits on AGE 1.8.0.
AGE_FUNCTIONS = frozenset(
    {"age_toupper", "age_tolower", "age_collect", "age_avg", "age_sum", "age_min", "age_max", "age_stdev", "age_id"}
    | {"age_labels", "age_type", "age_properties", "age_keys", "age_startnode", "age_endnode", "age_length", "age_nodes"}
    | {"age_relationships", "age_exists", "age_tostring", "age_tointeger", "age_size", "age_range", "age_label", "age_percentilecont"}
)

ACCEPTED = [
    "MATCH (n) RETURN count(n)",
    "MATCH (n:Person) WHERE n.age = 30 RETURN n.name",
    "MATCH (n)-[e]->(m) RETURN id(n), labels(n), type(e), properties(n), keys(n), startNode(e), endNode(e)",
    "MATCH p=(n)-[e]->(m) RETURN length(p), nodes(p), relationships(p)",
    "MATCH (n) RETURN avg(n.age), sum(n.age), min(n.age), max(n.age), stDev(n.age), collect(n.name)",
    "RETURN coalesce(null, 1), rand() < 2, toUpper('a'), size([1])",
    "RETURN [x IN [1,2,3] WHERE x > 1 | x * 2], any(x IN [1] WHERE x = 1), all(x IN [1] WHERE x = 1)",
    "RETURN none(x IN [1] WHERE x = 2), single(x IN [1] WHERE x = 1), reduce(acc = 0, x IN [1,2,3] | acc + x)",
    "MATCH (n) WHERE EXISTS((n)-[:KNOWS]->()) RETURN n.name",
    "MATCH (n) WHERE NOT (n)-[:KNOWS]->() AND (n.age > 1 OR n.age IN [2]) RETURN n",
    "MATCH (a)-[*1..2]->(b) RETURN count(*)",
    "MATCH (n) RETURN n.name ORDER BY n.name SKIP 0 LIMIT 1",
    "OPTIONAL MATCH (n:Nope) RETURN n",
    "RETURN 1 UNION RETURN 2",
    "MATCH (n) WITH n UNWIND [1,2] AS i RETURN DISTINCT label(n)",
    "MATCH (n) RETURN CASE WHEN n.age > 1 THEN 'x' ELSE 'y' END, n {.name}",
    "MATCH (n:`Person`) RETURN n.`name`",
    "MATCH (n) WHERE n.name = 'CREATE (x) SET y' RETURN n.set, n.create",  # keywords in strings and as property keys
    "MATCH (n:Delete)-[:SET]->(m) RETURN m",  # keywords as label and relationship type
    "MATCH (n) RETURN n.name /* comment CREATE */ // line SET\n",
    "MATCH (n:Person) WHERE n.name = $name RETURN n.age",
    "EXPLAIN MATCH (n) RETURN n",
    "MATCH (n) RETURN {`set`: 1}",
]

REFUSED = [
    ("CREATE (x:Ghost {k: 1}) RETURN x", "CREATE"),
    ("MATCH (n) SET n.flag = 1 RETURN n", "SET"),
    ("MATCH (n) DETACH DELETE n", "DETACH"),
    ("MATCH (n) DELETE n", "DELETE"),
    ("MERGE (n:Ghost2 {k: 2}) RETURN n", "MERGE"),
    ("MATCH (n) REMOVE n.name RETURN n", "REMOVE"),
    ("CALL public.evil2(1) YIELD x RETURN x", "CALL"),
    ("MATCH (n) create (m) RETURN m", "CREATE"),
    ("EXPLAIN ANALYZE MATCH (n) RETURN n", "ANALYZE"),
    ("RETURN pg_catalog.pg_backend_pid()", "schema-qualified"),
    ("RETURN `pg_catalog`.`pg_backend_pid`()", "schema-qualified"),
    ("RETURN pg_catalog . pg_backend_pid ( )", "schema-qualified"),
    ("RETURN evil2(1)", "evil2"),
    ("RETURN pg_terminate_backend(1)", "pg_terminate_backend"),
    ("RETURN `evil2`(1)", "evil2"),
    ("RETURN prepare_cypher('g', 'q')", "prepare_cypher"),
    ("MATCH (n) RETURN {set: 1}", "SET"),
    ("RETURN 'unterminated", "unterminated"),
    ("RETURN 1 /* open", "unterminated"),
]


@pytest.mark.parametrize("query", ACCEPTED)
def test_guard_accepts_read_only_cypher(query):
    CypherGuard(AGE_FUNCTIONS).check(query)


@pytest.mark.parametrize(("query", "reason"), REFUSED)
def test_guard_refuses(query, reason):
    with pytest.raises(ValueError, match=reason):
        CypherGuard(AGE_FUNCTIONS).check(query)


def test_age_function_needs_the_allowlist():
    with pytest.raises(ValueError, match="toUpper"):
        CypherGuard(frozenset()).check("RETURN toUpper('a')")


def _driver() -> Mock:
    driver = Mock(spec=SqlDriver)
    driver.execute_query = AsyncMock(return_value=[])
    driver.execute_table = AsyncMock(return_value=None)
    return driver


def _safe(cypher: bool = True) -> SafeSqlDriver:
    return SafeSqlDriver(_driver(), extra_functions=AGE_FUNCTIONS, cypher_guard=CypherGuard(AGE_FUNCTIONS) if cypher else None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM cypher('probe_g', $$ MATCH (n) RETURN n $$) AS (n agtype)",
        "SELECT * FROM ag_catalog.cypher('probe_g', $$ MATCH (n) RETURN n $$) AS (n ag_catalog.agtype)",
        "SELECT ag_catalog.age_toupper(v) FROM cypher('probe_g', $q$ RETURN 'a' $q$) AS (v agtype)",
        "PREPARE q(agtype) AS SELECT * FROM cypher('probe_g', $$ MATCH (n) WHERE n.name = $name RETURN n $$, $1) AS (v agtype)",
    ],
)
async def test_restricted_sql_accepts_read_only_cypher(sql):
    safe = _safe()
    await safe.execute_query(sql)
    assert safe.sql_driver.execute_query.await_args.kwargs["force_readonly"] is True  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("SELECT * FROM cypher('probe_g', $$ CREATE (n) RETURN n $$) AS (n agtype)", "CREATE"),
        ("SELECT * FROM cypher('probe_g', $$ RETURN pg_catalog.pg_backend_pid() $$) AS (n agtype)", "schema-qualified"),
        ("SELECT * FROM cypher(NULL, NULL) AS (n agtype)", "string constants"),
        ("SELECT * FROM cypher('probe_g', 'MATCH (n) RETURN n' || '') AS (n agtype)", "string constants"),
        ("SELECT * FROM cypher('probe_g', $$ RETURN 1 $$, '{}') AS (n agtype)", "bind parameter"),
        ("SELECT * FROM cypher('probe_g') AS (n agtype)", "graph name, a query"),
        ("SELECT * FROM public.cypher('probe_g', $$ RETURN 1 $$) AS (n agtype)", "not allowed"),
        ("SELECT ag_catalog.age_prepare_cypher('probe_g', 'CREATE (n)')", "not allowed"),
        ("SELECT * FROM ag_catalog.create_graph('x')", "not allowed"),
        ("SELECT a.b.age_toupper(v) FROM t", "not allowed"),
    ],
)
async def test_restricted_sql_refuses(sql, reason):
    with pytest.raises(ValueError, match=reason):
        await _safe().execute_query(sql)


@pytest.mark.asyncio
async def test_cypher_refused_without_age_opt_in():
    with pytest.raises(ValueError, match="allow-extension-functions age"):
        await _safe(cypher=False).execute_query("SELECT * FROM cypher('probe_g', $$ RETURN 1 $$) AS (n agtype)")


@pytest.mark.asyncio
async def test_qualified_core_function_outside_pg_catalog_stays_refused():
    with pytest.raises(ValueError, match="public.count"):
        await _safe().execute_query("SELECT public.count(*) FROM t")


def test_build_statement_without_params():
    s = build_cypher_statement("probe_g", "MATCH (n) WHERE n.x % 2 = 0 RETURN n", ["n"])
    assert s.sql == s.validate_as
    assert s.sql == 'SELECT * FROM ag_catalog.cypher(\'probe_g\', $cypher$MATCH (n) WHERE n.x % 2 = 0 RETURN n$cypher$) AS ("n" ag_catalog.agtype)'
    assert s.params == []


def test_build_statement_with_params_doubles_percent_and_binds_json():
    s = build_cypher_statement("probe_g", "MATCH (n) WHERE n.name = $name AND n.x % 2 = 0 RETURN n.age, n.name", ["age", "name"], {"name": "Ann"})
    assert "$name AND n.x %% 2 = 0" in s.sql and s.sql.endswith(', %s) AS ("age" ag_catalog.agtype, "name" ag_catalog.agtype)')
    assert "n.x % 2 = 0" in s.validate_as and ", $1) AS (" in s.validate_as
    assert s.params == ['{"name": "Ann"}']


def test_build_statement_picks_a_dollar_tag_absent_from_the_query():
    s = build_cypher_statement("probe_g", "RETURN '$cypher$'", ["v"])
    assert "$cypher_$RETURN '$cypher$'$cypher_$" in s.sql


@pytest.mark.parametrize(
    ("graph", "columns", "params"),
    [
        ("bad'name", ["v"], None),
        ("probe_g", [], None),
        ("probe_g", ['v" agtype) --'], None),
        ("probe_g", ["v", "v"], None),
        ("probe_g", ["v"], ["not", "a", "map"]),
    ],
)
def test_build_statement_rejects_bad_input(graph, columns, params):
    with pytest.raises(ValueError):
        build_cypher_statement(graph, "RETURN 1", columns, params)


@pytest.mark.asyncio
async def test_built_statement_passes_the_restricted_validator():
    safe = _safe()
    s = build_cypher_statement("probe_g", "MATCH (n) WHERE n.name = $name RETURN n", ["n"], {"name": "x"})
    await safe.execute_bound_table(s.sql, s.params, validate_as=s.validate_as)  # type: ignore[arg-type]
    call = safe.sql_driver.execute_table.await_args  # type: ignore[attr-defined]
    assert call.kwargs["force_readonly"] is True and call.kwargs["params"] == s.params
    assert "%s" in call.args[0]

"""Opt-in allowlisting of read-only functions owned by installed extensions (restricted mode).

Restricted mode accepts a function call only if its name is in ``SafeSqlDriver.ALLOWED_FUNCTIONS``,
a fixed list of core PostgreSQL functions. That rejects every PostGIS or MobilityDB function, so a
spatial or temporal SELECT cannot run at all.

``--allow-extension-functions postgis,mobilitydb`` reads the functions of the named extensions from
``pg_catalog`` once at startup and admits a function name only when every function in the whole
database with that name is safe:

- IMMUTABLE or STABLE (``provolatile`` in ``i``/``s``). VOLATILE functions are excluded, which drops
  writers such as ``AddGeometryColumn`` or ``UpdateGeometrySRID``.
- not SECURITY DEFINER (``prosecdef``), so the call runs with the caller's own privileges.
- a plain function (``prokind = 'f'``) or an aggregate (``'a'``), not a procedure or window
  function. An aggregate passes only if every support function it calls (transition, final,
  combine, serialise, moving-aggregate) passes the same volatility and SECURITY DEFINER test:
  pgvector's ``avg(vector)`` qualifies, an aggregate with a volatile transition function does not.

``*`` admits every extension installed in that database that is also in
``SafeSqlDriver.ALLOWED_EXTENSIONS``; it is resolved per database at discovery, so each database
gets its own set. Extension names may contain hyphens (``uuid-ossp``); they reach the query as a
bound parameter.

The check is by name across all schemas because the validator matches unqualified names: if any
overload anywhere is unsafe, the name stays out. Extensions whose functions execute SQL passed as text
(pgRouting's ``pgr_*`` take an inner query) must not be named; their VOLATILE marking excludes them
anyway, and the inner query would bypass the parser.

The transaction stays read-only regardless (``SafeSqlDriver.execute_query`` forces it); this only
widens which read functions the parser accepts.
"""

from __future__ import annotations

import logging
import re

from typing_extensions import LiteralString

from .safe_sql import SafeSqlDriver
from .sql_driver import SqlDriver

logger = logging.getLogger(__name__)

_EXTENSION_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALL_INSTALLED = "*"

DISCOVERY_QUERY: LiteralString = """
WITH ext_funcs AS (
    SELECT DISTINCT lower(p.proname) AS name
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_depend d
      ON d.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
     AND d.objid = p.oid
     AND d.deptype = 'e'
    JOIN pg_catalog.pg_extension e ON e.oid = d.refobjid
    WHERE e.extname = ANY(%s)
)
SELECT f.name
FROM ext_funcs f
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_proc q
    LEFT JOIN pg_catalog.pg_aggregate a ON a.aggfnoid = q.oid
    WHERE lower(q.proname) = f.name
      AND (
        q.provolatile = 'v' OR q.prosecdef OR q.prokind NOT IN ('f', 'a')
        OR (q.prokind = 'a' AND EXISTS (
              SELECT 1 FROM pg_catalog.pg_proc s
              WHERE s.oid IN (a.aggtransfn::oid, a.aggfinalfn::oid, a.aggcombinefn::oid, a.aggserialfn::oid,
                              a.aggdeserialfn::oid, a.aggmtransfn::oid, a.aggminvtransfn::oid, a.aggmfinalfn::oid)
                AND (s.provolatile = 'v' OR s.prosecdef)))
      )
)
ORDER BY f.name
"""

INSTALLED_QUERY: LiteralString = "SELECT extname FROM pg_catalog.pg_extension WHERE extname = ANY(%s) ORDER BY extname"


def parse_extension_list(value: str | None) -> tuple[str, ...]:
    """Split ``postgis,uuid-ossp`` into validated, lower-case extension names; ``*`` stands alone."""
    if not value:
        return ()
    names = tuple(sorted({part.strip().lower() for part in value.split(",") if part.strip()}))
    if ALL_INSTALLED in names:
        if len(names) > 1:
            raise ValueError("'*' admits every installed extension and cannot be combined with names")
        return (ALL_INSTALLED,)
    for name in names:
        if not _EXTENSION_NAME.match(name):
            raise ValueError(f"invalid extension name: {name!r}")
    return names


async def resolve_extensions(sql_driver: SqlDriver, extensions: tuple[str, ...]) -> tuple[str, ...]:
    """``("*",)`` becomes the installed extensions that ``SafeSqlDriver.ALLOWED_EXTENSIONS`` lists."""
    if extensions != (ALL_INSTALLED,):
        return extensions
    rows = await sql_driver.execute_query(INSTALLED_QUERY, params=[sorted(SafeSqlDriver.ALLOWED_EXTENSIONS)], force_readonly=True)
    return tuple(str(row.cells["extname"]) for row in rows or [])


async def discover_read_only_functions(sql_driver: SqlDriver, extensions: tuple[str, ...]) -> frozenset[str]:
    """Names of the safe functions owned by ``extensions``; empty when none are named or installed."""
    resolved = await resolve_extensions(sql_driver, extensions)
    if not resolved:
        return frozenset()
    rows = await sql_driver.execute_query(DISCOVERY_QUERY, params=[list(resolved)], force_readonly=True)
    names = frozenset(str(row.cells["name"]) for row in rows or [])
    logger.info(f"Allowing {len(names)} read-only functions from extensions: {', '.join(resolved)}")
    return names

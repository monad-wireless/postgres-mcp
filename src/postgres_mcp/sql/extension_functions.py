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
- a plain function (``prokind = 'f'``), not a procedure.

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

from .sql_driver import SqlDriver

logger = logging.getLogger(__name__)

_EXTENSION_NAME = re.compile(r"^[a-z0-9_]+$")

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
    SELECT 1 FROM pg_catalog.pg_proc q
    WHERE lower(q.proname) = f.name
      AND (q.provolatile = 'v' OR q.prosecdef OR q.prokind <> 'f')
)
ORDER BY f.name
"""


def parse_extension_list(value: str | None) -> tuple[str, ...]:
    """Split ``postgis,mobilitydb`` into validated, lower-case extension names."""
    if not value:
        return ()
    names = tuple(sorted({part.strip().lower() for part in value.split(",") if part.strip()}))
    for name in names:
        if not _EXTENSION_NAME.match(name):
            raise ValueError(f"invalid extension name: {name!r}")
    return names


async def discover_read_only_functions(sql_driver: SqlDriver, extensions: tuple[str, ...]) -> frozenset[str]:
    """Names of the safe functions owned by ``extensions``; empty when none are named."""
    if not extensions:
        return frozenset()
    rows = await sql_driver.execute_query(DISCOVERY_QUERY, params=[list(extensions)], force_readonly=True)
    names = frozenset(str(row.cells["name"]) for row in rows or [])
    logger.info(f"Allowing {len(names)} read-only functions from extensions: {', '.join(extensions)}")
    return names

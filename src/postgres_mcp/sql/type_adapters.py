"""Per-connection type loaders: values psycopg's defaults lose, abort on, or leave unparsed.

The pool calls ``configure_connection`` once for every new connection (psycopg_pool ``configure=``).
It changes three things relative to psycopg's defaults:

1. **Infinite dates and timestamps.** The default loaders raise ``DataError`` ("date too large
   (after year 10K)") on ``'infinity'`` and ``'-infinity'``, which fails the whole result. The
   loaders here return the strings ``"infinity"`` / ``"-infinity"`` instead; finite values load as
   before.
2. **Intervals.** The default loader returns ``timedelta``, which has no months, so ``'1 mon'``
   became 30 days and the original value was lost. Intervals load as the text PostgreSQL prints
   (``"1 mon 2 days 03:00:00"``, subject to the server's ``IntervalStyle``).
3. **Extension types, when the extension is installed in that database.** ``hstore`` loads as a
   ``dict`` (psycopg's own adapter); pgvector's ``vector`` and ``halfvec`` load as ``Vector`` and
   ``sparsevec`` as ``SparseVector``, parsed from their text form, so no pgvector client library is
   needed. Apache AGE's ``agtype`` loads through ``agtype.parse_agtype`` (vertices, edges and
   paths as dataclasses, ``::numeric`` as ``Decimal``). Other extension types (``citext``,
   ``ltree``) already load as ``str``.
4. **Apache AGE session setup, when ``agtype`` exists.** ``prepare_age_session`` runs
   ``LOAD 'age'`` (then ``LOAD '$libdir/plugins/age'``) and appends ``ag_catalog`` to
   ``search_path``. Probes on AGE 1.8.0 showed why: without ``shared_preload_libraries=age`` the
   first ``cypher()`` call of a session fails ("unhandled cypher(cstring) function call") unless
   the library was loaded; without ``ag_catalog`` on the path, ``create_graph`` and Cypher
   comparisons (``WHERE n.age = 30``) fail. A non-superuser may not ``LOAD 'age'`` even when it
   is preloaded, so a refused ``LOAD`` is normal and only logged. ``ag_catalog`` goes last, so
   the caller's own schemas still win every name lookup.

The extension lookup is one catalogue query per pool, cached on the pool, so a connection that
psycopg_pool recycles after ``max_idle`` does not repeat it. The pool also remembers which
``LOAD`` form worked, so a refused one is tried once per pool, not once per connection.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from typing import Any

from psycopg import AsyncConnection
from psycopg import pq
from psycopg.adapt import Loader
from psycopg.types import TypeInfo
from psycopg.types.datetime import DateBinaryLoader
from psycopg.types.datetime import DateLoader
from psycopg.types.datetime import TimestampBinaryLoader
from psycopg.types.datetime import TimestampLoader
from psycopg.types.datetime import TimestamptzBinaryLoader
from psycopg.types.datetime import TimestamptzLoader
from psycopg.types.hstore import register_hstore
from typing_extensions import LiteralString

from .agtype import AgtypeSyntaxError
from .agtype import parse_agtype

logger = logging.getLogger(__name__)

EXTENSION_TYPES: tuple[str, ...] = ("hstore", "vector", "halfvec", "sparsevec", "agtype")

AGE_LOAD_STATEMENTS: tuple[LiteralString, ...] = ()
AGE_SEARCH_PATH: LiteralString = """
SELECT pg_catalog.set_config(
    'search_path',
    CASE WHEN pg_catalog.current_setting('search_path') = '' THEN 'ag_catalog'
         ELSE pg_catalog.current_setting('search_path') || ', ag_catalog' END,
    false)
WHERE 'ag_catalog' <> ALL (pg_catalog.current_schemas(false))
"""

_EXTENSION_TYPE_QUERY = """
SELECT t.typname, t.oid::int, t.typarray::int
FROM pg_catalog.pg_type t
WHERE t.typname = ANY(%s) AND t.typtype = 'b'
"""

# PostgreSQL's binary encodings of +/-infinity: int32 days for date, int64 microseconds for timestamps.
_DATE_INF = {struct.pack("!i", 0x7FFFFFFF): "infinity", struct.pack("!i", -0x80000000): "-infinity"}
_TS_INF = {struct.pack("!q", 0x7FFFFFFFFFFFFFFF): "infinity", struct.pack("!q", -0x8000000000000000): "-infinity"}
_TEXT_INF = {b"infinity": "infinity", b"-infinity": "-infinity"}


@dataclass(frozen=True, slots=True)
class Vector:
    """A pgvector ``vector`` or ``halfvec`` value."""

    values: tuple[float, ...]

    @classmethod
    def parse(cls, text: str) -> Vector:
        body = text.strip()[1:-1]
        return cls(tuple(float(x) for x in body.split(",")) if body else ())


@dataclass(frozen=True, slots=True)
class SparseVector:
    """A pgvector ``sparsevec`` value; ``indices`` are 1-based as PostgreSQL prints them."""

    dim: int
    indices: tuple[int, ...]
    values: tuple[float, ...]

    @classmethod
    def parse(cls, text: str) -> SparseVector:
        pairs, _, dim = text.strip().rpartition("/")
        body = pairs.strip()[1:-1]
        items = [item.split(":") for item in body.split(",")] if body else []
        return cls(int(dim), tuple(int(i) for i, _ in items), tuple(float(v) for _, v in items))


class _InfDateLoader(DateLoader):
    def load(self, data: Any) -> Any:
        return _TEXT_INF.get(bytes(data)) or super().load(data)


class _InfDateBinaryLoader(DateBinaryLoader):
    def load(self, data: Any) -> Any:
        return _DATE_INF.get(bytes(data)) or super().load(data)


class _InfTimestampLoader(TimestampLoader):
    def load(self, data: Any) -> Any:
        return _TEXT_INF.get(bytes(data)) or super().load(data)


class _InfTimestampBinaryLoader(TimestampBinaryLoader):
    def load(self, data: Any) -> Any:
        return _TS_INF.get(bytes(data)) or super().load(data)


class _InfTimestamptzLoader(TimestamptzLoader):
    def load(self, data: Any) -> Any:
        return _TEXT_INF.get(bytes(data)) or super().load(data)


class _InfTimestamptzBinaryLoader(TimestamptzBinaryLoader):
    def load(self, data: Any) -> Any:
        return _TS_INF.get(bytes(data)) or super().load(data)


class _TextLoader(Loader):
    """Load a value as the text PostgreSQL printed."""

    def load(self, data: Any) -> str:
        return bytes(data).decode()


class _VectorLoader(Loader):
    def load(self, data: Any) -> Vector:
        return Vector.parse(bytes(data).decode())


class _SparseVectorLoader(Loader):
    def load(self, data: Any) -> SparseVector:
        return SparseVector.parse(bytes(data).decode())


class _AgtypeLoader(Loader):
    def load(self, data: Any) -> Any:
        text = bytes(data).decode()
        try:
            return parse_agtype(text)
        except AgtypeSyntaxError as e:
            logger.debug(f"agtype value kept as text ({e}): {text[:200]}")
            return text


def register_core_loaders(conn: AsyncConnection[Any]) -> None:
    """Infinity-safe date/timestamp loaders and verbatim intervals (no query, no transaction)."""
    adapters = conn.adapters
    for typ, text_loader, binary_loader in (
        ("date", _InfDateLoader, _InfDateBinaryLoader),
        ("timestamp", _InfTimestampLoader, _InfTimestampBinaryLoader),
        ("timestamptz", _InfTimestamptzLoader, _InfTimestamptzBinaryLoader),
    ):
        adapters.register_loader(typ, text_loader)
        adapters.register_loader(typ, binary_loader)
    adapters.register_loader("interval", _TextLoader)


class ExtensionTypeCache:
    """The extension types installed in one database, fetched on the pool's first connection."""

    def __init__(self) -> None:
        self.types: dict[str, TypeInfo] | None = None
        # AGE: candidate LOAD statements still worth trying in this pool; narrowed to the one that worked.
        self.age_loads: tuple[LiteralString, ...] = AGE_LOAD_STATEMENTS

    async def fetch(self, conn: AsyncConnection[Any]) -> dict[str, TypeInfo]:
        if self.types is None:
            async with conn.cursor() as cur:
                await cur.execute(_EXTENSION_TYPE_QUERY, [list(EXTENSION_TYPES)])
                rows = await cur.fetchall()
            self.types = {name: TypeInfo(name, oid, array_oid) for name, oid, array_oid in rows}
            if self.types:
                logger.info(f"Extension types with native loaders: {', '.join(sorted(self.types))}")
        return self.types


def register_extension_loaders(conn: AsyncConnection[Any], types: dict[str, TypeInfo]) -> None:
    if (info := types.get("hstore")) is not None:
        register_hstore(info, conn)
    for name, loader in (("vector", _VectorLoader), ("halfvec", _VectorLoader), ("sparsevec", _SparseVectorLoader), ("agtype", _AgtypeLoader)):
        if (info := types.get(name)) is not None:
            info.register(conn)  # so column type names resolve ("vector", not the OID)
            conn.adapters.register_loader(info.oid, loader)


async def prepare_age_session(conn: AsyncConnection[Any], cache: ExtensionTypeCache) -> None:
    """Load the AGE library if this role may, then append ``ag_catalog`` to ``search_path`` (committed)."""
    for statement in cache.age_loads:
        try:
            await conn.execute(statement)
            await conn.commit()
            cache.age_loads = (statement,)
            break
        except Exception as e:
            await conn.rollback()
            logger.debug(f"{statement} refused: {e}")
    else:
        if cache.age_loads:
            logger.info(
                "This role may not LOAD the AGE library; cypher() works when age is in shared_preload_libraries "
                "or session_preload_libraries, otherwise the first call of each session fails"
            )
        cache.age_loads = ()
    # set_config(..., false) is session-wide but transactional: the commit keeps it past configure's rollback.
    await conn.execute(AGE_SEARCH_PATH)
    await conn.commit()


async def configure_connection(conn: AsyncConnection[Any], cache: ExtensionTypeCache) -> None:
    """psycopg_pool ``configure=`` body: leaves the connection idle, as the pool requires."""
    register_core_loaders(conn)
    try:
        types = await cache.fetch(conn)
        register_extension_loaders(conn, types)
    except Exception as e:
        # A failed lookup must not cost the connection: extension values then load as text.
        logger.warning(f"Extension type lookup failed; those values stay text: {e}")
        types = {}
    if "agtype" in types:
        try:
            await prepare_age_session(conn, cache)
        except Exception as e:
            logger.warning(f"AGE session setup failed; cypher() may need ag_catalog qualified names: {e}")
            await conn.rollback()
    if conn.info.transaction_status != pq.TransactionStatus.IDLE:
        await conn.rollback()

"""Named database connections: one server process, several PostgreSQL databases.

A database is configured under a short name and gets its own connection pool and its own
restricted-mode extension-function allowlist (``--allow-extension-functions`` is discovered per
database, because one database may have PostGIS installed and the next may not).

Sources, in the order they are registered:

1. The legacy single URL: ``DATABASE_URI`` or the positional ``database_url`` argument, under the
   name ``default``. Existing single-database configurations keep working unchanged.
2. ``--database NAME=URI``, repeatable, in command-line order.
3. ``DATABASE_URI_<NAME>`` environment variables, sorted by name. ``<NAME>`` is lower-cased.

The default database (used when a tool call names none) is ``--default-database`` when given,
otherwise the first registered one. A name registered twice is a configuration error.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from urllib.parse import urlparse

from .sql import DbConnPool
from .sql import SqlDriver
from .sql import discover_read_only_functions
from .sql import obfuscate_password

logger = logging.getLogger(__name__)

LEGACY_NAME = "default"
ENV_PREFIX = "DATABASE_URI_"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class UnknownDatabaseError(ValueError):
    """A tool call named a database that is not configured."""


def validate_name(name: str) -> str:
    """Return ``name`` if it is a valid database name, else raise ``ValueError``."""
    if not _NAME.match(name):
        raise ValueError(f"invalid database name {name!r}: use lower-case letters, digits, '_' or '-', starting with a letter or digit")
    return name


def parse_database_spec(spec: str) -> tuple[str, str]:
    """Split ``gis=postgresql://...`` into ``("gis", "postgresql://...")``."""
    name, sep, url = spec.partition("=")
    if not sep or not url.strip():
        raise ValueError(f"invalid --database value {obfuscate_password(spec)!r}: expected NAME=URI")
    return validate_name(name.strip()), url.strip()


def collect_database_urls(
    legacy_url: str | None,
    cli_specs: Sequence[str],
    environ: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Ordered ``(name, url)`` pairs from the three configuration sources (see module docstring)."""
    pairs: list[tuple[str, str]] = []
    if legacy_url:
        pairs.append((LEGACY_NAME, legacy_url))
    pairs.extend(parse_database_spec(spec) for spec in cli_specs)
    env_pairs = [
        (validate_name(key[len(ENV_PREFIX) :].lower()), value) for key, value in environ.items() if key.startswith(ENV_PREFIX) and value.strip()
    ]
    pairs.extend(sorted(env_pairs))

    seen: set[str] = set()
    for name, _ in pairs:
        if name in seen:
            raise ValueError(f"database {name!r} is configured more than once")
        seen.add(name)
    return pairs


@dataclass
class Database:
    """One configured database: its pool and its restricted-mode function allowlist."""

    name: str
    url: str
    pool: DbConnPool
    extension_functions: frozenset[str] = field(default_factory=frozenset)

    def describe(self, is_default: bool) -> dict[str, Any]:
        """Name, location and state, without credentials."""
        parsed = urlparse(self.url)
        return {
            "name": self.name,
            "default": is_default,
            "host": parsed.hostname,
            "port": parsed.port,
            "dbname": parsed.path.lstrip("/") or None,
            "connected": self.pool.is_valid,
            "extension_functions": len(self.extension_functions),
        }


class DatabaseRegistry:
    """The databases this server can reach, by name, with one default."""

    def __init__(self, pairs: Sequence[tuple[str, str]] = (), default: str | None = None):
        self._databases: dict[str, Database] = {name: Database(name=name, url=url, pool=DbConnPool(url)) for name, url in pairs}
        if default is not None and default not in self._databases:
            raise ValueError(f"--default-database {default!r} is not configured; configured: {', '.join(self._databases) or 'none'}")
        self.default_name: str | None = default if default is not None else next(iter(self._databases), None)

    def __iter__(self) -> Iterator[Database]:
        return iter(self._databases.values())

    def __len__(self) -> int:
        return len(self._databases)

    @property
    def names(self) -> list[str]:
        return list(self._databases)

    def get(self, name: str | None = None) -> Database:
        """The named database, or the default one when ``name`` is ``None`` or empty."""
        key = name or self.default_name
        if key is None:
            raise UnknownDatabaseError("No database is configured.")
        try:
            return self._databases[key]
        except KeyError:
            raise UnknownDatabaseError(f"Unknown database {key!r}. Configured databases: {', '.join(self._databases)}") from None

    def describe(self) -> list[dict[str, Any]]:
        return [db.describe(db.name == self.default_name) for db in self]

    async def connect_all(self, allowed_extensions: tuple[str, ...]) -> None:
        """Open every pool concurrently; discover extension functions where the pool opened.

        A database that cannot be reached is logged and skipped: the server still starts and the
        other databases work. Its pool reconnects lazily on the next query, but its extension
        functions stay undiscovered until the server restarts (same as the single-database server).
        """
        await asyncio.gather(*(self._connect_one(db, allowed_extensions) for db in self))

    async def _connect_one(self, db: Database, allowed_extensions: tuple[str, ...]) -> None:
        try:
            await db.pool.pool_connect()
            logger.info(f"Connected to database {db.name!r}")
            if allowed_extensions:
                db.extension_functions = await discover_read_only_functions(SqlDriver(conn=db.pool), allowed_extensions)
        except Exception as e:
            logger.warning(f"Could not connect to database {db.name!r}: {obfuscate_password(str(e))}")
            logger.warning(f"The MCP server will start but operations on {db.name!r} will fail until a valid connection is established.")

    async def close_all(self) -> None:
        for db in self:
            try:
                await db.pool.close()
            except Exception as e:
                logger.error(f"Error closing database {db.name!r}: {e}")

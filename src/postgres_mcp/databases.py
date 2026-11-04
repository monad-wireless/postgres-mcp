"""Named database connections: one server process, several PostgreSQL databases.

A database is configured under a short name and gets its own connection pool and its own
restricted-mode extension-function allowlist (``--allow-extension-functions`` is discovered per
database, because one database may have PostGIS installed and the next may not). Pools open on
first use and share one ``PoolBudget``, so many configured databases cost connections only while
they are in use.

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
from .sql import PoolBudget
from .sql import PoolSettings
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
    """One configured database: its lazily opened pool and its restricted-mode function allowlist.

    ``extension_functions`` is ``None`` until discovery succeeds. It lives here and not on the pool,
    so a pool that the budget evicts and later reopens does not rerun the catalogue query.
    """

    name: str
    url: str
    pool: DbConnPool
    extension_functions: frozenset[str] | None = None
    _discovery_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False, compare=False)

    async def ensure_discovered(self, allowed_extensions: tuple[str, ...]) -> None:
        """Discover the extension-function allowlist once; on failure stay undiscovered and retry next call.

        Until discovery succeeds the allowlist is empty, which is the strictest validator setting.
        """
        if self.extension_functions is not None:
            return
        if not allowed_extensions:
            self.extension_functions = frozenset()
            return
        async with self._discovery_lock:
            if self.extension_functions is not None:
                return
            try:
                self.extension_functions = await discover_read_only_functions(SqlDriver(conn=self.pool), allowed_extensions)
            except Exception as e:
                logger.warning(f"Extension discovery on {self.name!r} failed, will retry: {obfuscate_password(str(e))}")

    def describe(self, is_default: bool) -> dict[str, Any]:
        """Name, location and pool state, without credentials."""
        parsed = urlparse(self.url)
        return {
            "name": self.name,
            "default": is_default,
            "host": parsed.hostname,
            "port": parsed.port,
            "dbname": parsed.path.lstrip("/") or None,
            **self.pool.describe(),
            "extension_functions": None if self.extension_functions is None else len(self.extension_functions),
        }


class DatabaseRegistry:
    """The databases this server can reach, by name, with one default.

    Every database gets a ``DbConnPool`` at construction, but no pool opens until a tool call needs
    it. All pools share one ``PoolBudget``, which caps how many are open at once and retires the
    least recently used idle one (see ``PoolBudget`` for the invariants).
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]] = (),
        default: str | None = None,
        *,
        settings: PoolSettings | None = None,
        budget: PoolBudget | None = None,
        allowed_extensions: tuple[str, ...] = (),
    ):
        self.budget = budget or PoolBudget()
        self.allowed_extensions = allowed_extensions
        self._databases: dict[str, Database] = {
            name: Database(name=name, url=url, pool=DbConnPool(url, settings=settings, budget=self.budget)) for name, url in pairs
        }
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

    async def ready(self, name: str | None = None) -> Database:
        """The named (or default) database with its allowlist discovered. The pool opens on first query."""
        db = self.get(name)
        await db.ensure_discovered(self.allowed_extensions)
        return db

    async def open_default(self) -> None:
        """Open the default database at startup, so a wrong URL shows in the log before the first tool call.

        A failure is logged, not raised: the server starts and the pool retries on the next query.
        The other databases open on first use.
        """
        if self.default_name is None:
            return
        db = self._databases[self.default_name]
        try:
            await db.pool.pool_connect()
            logger.info(f"Connected to default database {db.name!r}")
            await db.ensure_discovered(self.allowed_extensions)
        except Exception as e:
            logger.warning(f"Could not connect to database {db.name!r}: {obfuscate_password(str(e))}")
            logger.warning(f"The MCP server will start; {db.name!r} will retry on the next query.")

    async def close_all(self) -> None:
        for db in self:
            try:
                await db.pool.close()
            except Exception as e:
                logger.error(f"Error closing database {db.name!r}: {e}")

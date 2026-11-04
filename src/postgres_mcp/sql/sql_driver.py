"""SQL driver adapter for PostgreSQL connections."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from urllib.parse import urlparse
from urllib.parse import urlunparse

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from psycopg_pool import PoolTimeout
from typing_extensions import LiteralString

logger = logging.getLogger(__name__)


def obfuscate_password(text: str | None) -> str | None:
    """
    Obfuscate password in any text containing connection information.
    Works on connection URLs, error messages, and other strings.
    """
    if text is None:
        return None

    if not text:
        return text

    # Try first as a proper URL
    try:
        parsed = urlparse(text)
        if parsed.scheme and parsed.netloc and parsed.password:
            # Replace password with asterisks in proper URL
            netloc = parsed.netloc.replace(parsed.password, "****")
            return urlunparse(parsed._replace(netloc=netloc))
    except Exception:
        pass

    # Handle strings that contain connection strings but aren't proper URLs
    # Match postgres://user:password@host:port/dbname pattern
    url_pattern = re.compile(r"(postgres(?:ql)?:\/\/[^:]+:)([^@]+)(@[^\/\s]+)")
    text = re.sub(url_pattern, r"\1****\3", text)

    # Match connection string parameters (password=xxx)
    # This simpler pattern captures password without quotes
    param_pattern = re.compile(r'(password=)([^\s&;"\']+)', re.IGNORECASE)
    text = re.sub(param_pattern, r"\1****", text)

    # Match password in DSN format with single quotes
    dsn_single_quote = re.compile(r"(password\s*=\s*')([^']+)(')", re.IGNORECASE)
    text = re.sub(dsn_single_quote, r"\1****\3", text)

    # Match password in DSN format with double quotes
    dsn_double_quote = re.compile(r'(password\s*=\s*")([^"]+)(")', re.IGNORECASE)
    text = re.sub(dsn_double_quote, r"\1****\3", text)

    return text


PoolState = Literal["closed", "open", "failed"]


@dataclass(frozen=True, slots=True)
class PoolSettings:
    """Sizing and timeouts for one database's psycopg pool.

    ``min_size=0`` is the default on purpose: psycopg_pool closes a connection that has been idle
    for ``max_idle`` seconds once the pool holds more than ``min_size``, so an idle pool decays to
    zero server connections while the pool object (and its warm state) stays cached.
    """

    min_size: int = 0
    max_size: int = 5
    max_idle: float = 300.0
    max_lifetime: float = 3600.0
    connect_timeout: float = 10.0
    acquire_timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.min_size < 0 or self.max_size < 1 or self.min_size > self.max_size:
            raise ValueError(f"invalid pool size: min_size={self.min_size}, max_size={self.max_size}")
        if min(self.max_idle, self.max_lifetime, self.connect_timeout, self.acquire_timeout) <= 0:
            raise ValueError("pool timeouts must be positive")


class PoolBudget:
    """A bound on the number of open pools, shared by every ``DbConnPool`` of one server.

    Policy: least-recently-used replacement over pool objects, plus idle expiry. A pool is "used"
    when a lease (``DbConnPool.connection()``) starts or ends.

    Invariants (one asyncio event loop, so every synchronous block below is atomic):

    I1. A pool with a lease in flight is never retired. ``DbConnPool.connection()`` increments its
        lease count before its first ``await`` and ``retire_if_idle()`` checks it synchronously.
    I2. After ``admit()``, open pools <= ``max_open_pools`` unless every other open pool holds a
        lease. The cap is soft in that case: a tool call is never blocked on the cap, the overflow
        is logged, and the next admission or sweep restores the bound.
    I3. Retirement order is least recently used among idle pools.
    I4. A pool idle for longer than ``idle_ttl`` seconds is retired at the next lease or admission
        on any pool. There is no background task, so the expiry is lazy and deterministic.

    Consequence: server connections held by this process <= ``max_open_pools * max_size`` outside
    the soft-cap case. Per-connection idle reaping, health checks and connection lifetime stay with
    psycopg_pool (``max_idle``, ``check``, ``max_lifetime``); this class only manages pool objects.
    """

    def __init__(
        self,
        max_open_pools: int | None = None,
        idle_ttl: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_open_pools is not None and max_open_pools < 1:
            raise ValueError("max_open_pools must be at least 1")
        if idle_ttl is not None and idle_ttl <= 0:
            raise ValueError("idle_ttl must be positive")
        self.max_open_pools = max_open_pools
        self.idle_ttl = idle_ttl
        self.clock = clock
        # Open pools, least recently used first.
        self._lru: OrderedDict[DbConnPool, None] = OrderedDict()

    @property
    def open_pools(self) -> list[DbConnPool]:
        """Open pools, least recently used first."""
        return list(self._lru)

    def touch(self, pool: DbConnPool) -> None:
        if pool in self._lru:
            self._lru.move_to_end(pool)

    def forget(self, pool: DbConnPool) -> None:
        self._lru.pop(pool, None)

    async def admit(self, pool: DbConnPool) -> None:
        """Register a newly opened pool, then retire expired and least-recently-used idle pools (I2-I4)."""
        self._lru[pool] = None
        self._lru.move_to_end(pool)
        victims = self._expired(keep=pool)
        if self.max_open_pools is not None:
            for candidate in list(self._lru):
                if len(self._lru) - len(victims) <= self.max_open_pools:
                    break
                if candidate is not pool and candidate not in victims and candidate.leases == 0:
                    victims.append(candidate)
            overflow = len(self._lru) - len(victims) - self.max_open_pools
            if overflow > 0:
                logger.warning(f"{len(self._lru) - len(victims)} pools open, cap {self.max_open_pools}: every other pool is busy")
        await self._retire(victims)

    async def sweep(self, keep: DbConnPool) -> None:
        """Retire pools idle for longer than ``idle_ttl`` (I4), never ``keep``."""
        await self._retire(self._expired(keep=keep))

    def _expired(self, keep: DbConnPool) -> list[DbConnPool]:
        if self.idle_ttl is None:
            return []
        now = self.clock()
        return [p for p in self._lru if p is not keep and p.leases == 0 and now - p.last_used > self.idle_ttl]

    async def _retire(self, victims: list[DbConnPool]) -> None:
        # Detach every victim synchronously first (I1), then await the closes.
        detached = [(victim, raw) for victim in victims if (raw := victim.retire_if_idle()) is not None]
        for victim, raw in detached:
            logger.info(f"Closing idle pool {obfuscate_password(victim.connection_url)}")
            await _close_quietly(raw)


async def _close_quietly(pool: AsyncConnectionPool[Any]) -> None:
    try:
        await pool.close()
    except Exception as e:
        logger.warning(f"Error closing connection pool: {obfuscate_password(str(e))}")


class DbConnPool:
    """A lazily opened psycopg pool for one database.

    The pool opens on the first ``connection()`` (or an explicit ``pool_connect()``), at most once
    at a time per instance. A failed open leaves the state ``failed`` and the next call tries again.
    A query error never closes the pool: psycopg_pool's ``check`` drops dead connections at
    checkout and opens new ones on demand, so only an unreachable server shows up as a failure.
    """

    def __init__(
        self,
        connection_url: Optional[str] = None,
        settings: PoolSettings | None = None,
        budget: PoolBudget | None = None,
    ):
        self.connection_url = connection_url
        self.settings = settings or PoolSettings()
        self.budget = budget
        self.pool: AsyncConnectionPool[Any] | None = None
        self.state: PoolState = "closed"
        self.leases = 0
        self.last_used: float = self._now()
        self._last_error: str | None = None
        self._open_lock = asyncio.Lock()

    def _now(self) -> float:
        return self.budget.clock() if self.budget is not None else time.monotonic()

    async def pool_connect(self, connection_url: Optional[str] = None) -> AsyncConnectionPool[Any]:
        """Return the open pool, opening (and probing) it first if needed."""
        if self.pool is not None and self.state == "open":
            return self.pool
        async with self._open_lock:
            # A concurrent caller may have opened it while this one waited on the lock.
            if self.pool is not None and self.state == "open":
                return self.pool
            return await self._open(connection_url or self.connection_url)

    async def _open(self, url: str | None) -> AsyncConnectionPool[Any]:
        self.connection_url = url
        if not url:
            self.state = "failed"
            self._last_error = "Database connection URL not provided"
            raise ValueError(self._last_error)

        s = self.settings
        pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
            conninfo=url,
            min_size=s.min_size,
            max_size=s.max_size,
            max_idle=s.max_idle,
            max_lifetime=s.max_lifetime,
            timeout=s.acquire_timeout,
            kwargs={"connect_timeout": max(1, round(s.connect_timeout))},
            check=AsyncConnectionPool.check_connection,
            open=False,
        )
        try:
            await pool.open(wait=True, timeout=s.connect_timeout)
            # Probe: with min_size=0 the open succeeds without a connection, so ask for one.
            async with pool.connection(timeout=s.connect_timeout) as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("SELECT 1")
        except Exception as e:
            self.state = "failed"
            self._last_error = obfuscate_password(str(e))
            await _close_quietly(pool)
            raise ValueError(f"Connection attempt failed: {obfuscate_password(str(e))}") from e

        self.pool = pool
        self.state = "open"
        self._last_error = None
        self.last_used = self._now()
        if self.budget is not None:
            await self.budget.admit(self)
        return pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        """Lease one connection; the lease pins this pool against eviction until it ends."""
        self.leases += 1  # before the first await (PoolBudget invariant I1)
        self.last_used = self._now()
        try:
            if self.budget is not None:
                self.budget.touch(self)
                await self.budget.sweep(keep=self)
            pool = await self.pool_connect()
            try:
                async with pool.connection() as conn:
                    yield conn
            except PoolTimeout as e:
                self._last_error = obfuscate_password(str(e))
                raise
        finally:
            self.leases -= 1
            self.last_used = self._now()
            if self.budget is not None:
                self.budget.touch(self)

    def retire_if_idle(self) -> AsyncConnectionPool[Any] | None:
        """Detach and return the psycopg pool if no lease is in flight; the caller closes it."""
        if self.leases or self.pool is None:
            return None
        pool, self.pool, self.state = self.pool, None, "closed"
        if self.budget is not None:
            self.budget.forget(self)
        return pool

    async def close(self) -> None:
        """Close the pool now, leases or not (shutdown path)."""
        pool, self.pool = self.pool, None
        self.state = "closed"
        if self.budget is not None:
            self.budget.forget(self)
        if pool is not None:
            await _close_quietly(pool)

    @property
    def is_valid(self) -> bool:
        """True while the pool is open."""
        return self.state == "open"

    @property
    def last_error(self) -> Optional[str]:
        """The last open or acquire failure, with any password masked."""
        return self._last_error

    def describe(self) -> dict[str, Any]:
        """State for list_databases: no URL, no credentials."""
        return {
            "state": self.state,
            "in_flight": self.leases,
            "idle_seconds": None if self.leases or self.state != "open" else round(self._now() - self.last_used, 1),
            "last_error": self._last_error,
        }


class SqlDriver:
    """Adapter class that wraps a PostgreSQL connection with the interface expected by DTA."""

    @dataclass
    class RowResult:
        """Simple class to match the Griptape RowResult interface."""

        cells: Dict[str, Any]

    def __init__(
        self,
        conn: Any = None,
        engine_url: str | None = None,
    ):
        """
        Initialize with a PostgreSQL connection or pool.

        Args:
            conn: PostgreSQL connection object or pool
            engine_url: Connection URL string as an alternative to providing a connection
        """
        if conn:
            self.conn = conn
            # Check if this is a connection pool
            self.is_pool = isinstance(conn, DbConnPool)
        elif engine_url:
            # Don't connect here since we need async connection
            self.engine_url = engine_url
            self.conn = None
            self.is_pool = False
        else:
            raise ValueError("Either conn or engine_url must be provided")

    def connect(self):
        if self.conn is not None:
            return self.conn
        if self.engine_url:
            self.conn = DbConnPool(self.engine_url)
            self.is_pool = True
            return self.conn
        else:
            raise ValueError("Connection not established. Either conn or engine_url must be provided")

    async def execute_query(
        self,
        query: LiteralString,
        params: list[Any] | None = None,
        force_readonly: bool = False,
    ) -> Optional[List[RowResult]]:
        """
        Execute a query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters
            force_readonly: Whether to enforce read-only mode

        Returns:
            List of RowResult objects or None on error
        """
        try:
            if self.conn is None:
                self.connect()
                if self.conn is None:
                    raise ValueError("Connection not established")

            if self.is_pool:
                # A lease pins the pool against eviction for the duration of the query. A query
                # error leaves the pool open: psycopg_pool drops a broken connection at checkout.
                async with self.conn.connection() as connection:
                    return await self._execute_with_connection(connection, query, params, force_readonly=force_readonly)
            else:
                return await self._execute_with_connection(self.conn, query, params, force_readonly=force_readonly)
        except Exception:
            if self.conn and not self.is_pool:
                self.conn = None
            raise

    async def _execute_with_connection(self, connection, query, params, force_readonly) -> Optional[List[RowResult]]:
        """Execute query with the given connection."""
        transaction_started = False
        try:
            async with connection.cursor(row_factory=dict_row) as cursor:
                # Start read-only transaction
                if force_readonly:
                    await cursor.execute("BEGIN TRANSACTION READ ONLY")
                    transaction_started = True

                if params:
                    await cursor.execute(query, params)
                else:
                    await cursor.execute(query)

                # For multiple statements, move to the last statement's results
                while cursor.nextset():
                    pass

                if cursor.description is None:  # No results (like DDL statements)
                    if not force_readonly:
                        await cursor.execute("COMMIT")
                    elif transaction_started:
                        await cursor.execute("ROLLBACK")
                        transaction_started = False
                    return None

                # Get results from the last statement only
                rows = await cursor.fetchall()

                # End the transaction appropriately
                if not force_readonly:
                    await cursor.execute("COMMIT")
                elif transaction_started:
                    await cursor.execute("ROLLBACK")
                    transaction_started = False

                return [SqlDriver.RowResult(cells=dict(row)) for row in rows]

        except Exception as e:
            # Try to roll back the transaction if it's still active
            if transaction_started:
                try:
                    await connection.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error rolling back transaction: {rollback_error}")

            logger.error(f"Error executing query ({query}): {e}")
            raise e

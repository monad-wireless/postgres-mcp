"""PoolBudget and lazy DbConnPool: eviction order, lease pinning, idle expiry, single open.

No database: ``AsyncConnectionPool`` is replaced by ``FakeRawPool``, which records what it was
built with and whether it was opened or closed. Time comes from ``FakeClock``.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from typing import ClassVar
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from postgres_mcp.databases import DatabaseRegistry
from postgres_mcp.sql import DbConnPool
from postgres_mcp.sql import PoolBudget
from postgres_mcp.sql import PoolSettings


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class FakeRawPool:
    """Stands in for psycopg_pool.AsyncConnectionPool."""

    built: ClassVar[list["FakeRawPool"]] = []

    def __init__(self, conninfo: str = "", **kwargs: Any) -> None:
        self.conninfo = conninfo
        self.kwargs = kwargs
        self.opened = False
        self.closed = False
        FakeRawPool.built.append(self)

    @staticmethod
    async def check_connection(conn: Any) -> None:
        return None

    async def open(self, wait: bool = False, timeout: float = 30.0) -> None:
        await asyncio.sleep(0)  # yield, so concurrent first uses really interleave
        self.opened = True

    async def close(self, timeout: float = 5.0) -> None:
        self.closed = True

    @asynccontextmanager
    async def connection(self, timeout: float | None = None):
        conn = MagicMock()
        cursor = AsyncMock()

        @asynccontextmanager
        async def cursor_cm(*args: Any, **kwargs: Any):
            yield cursor

        conn.cursor = cursor_cm
        yield conn


@pytest.fixture(autouse=True)
def fake_pool_class():
    FakeRawPool.built = []
    with patch("postgres_mcp.sql.sql_driver.AsyncConnectionPool", FakeRawPool):
        yield


def _pools(budget: PoolBudget, *names: str) -> list[DbConnPool]:
    return [DbConnPool(f"postgresql://u:pw@h/{n}", budget=budget) for n in names]


async def _use(pool: DbConnPool) -> None:
    async with pool.connection():
        pass


def test_settings_reject_impossible_sizes():
    with pytest.raises(ValueError):
        PoolSettings(min_size=3, max_size=2)
    with pytest.raises(ValueError):
        PoolSettings(connect_timeout=0)
    with pytest.raises(ValueError):
        PoolBudget(max_open_pools=0)


@pytest.mark.asyncio
async def test_pool_is_built_with_the_settings_and_a_health_check():
    pool = DbConnPool("postgresql://u:pw@h/db", settings=PoolSettings(max_size=3, max_idle=42.0, connect_timeout=7.0))
    assert FakeRawPool.built == []  # nothing opens at construction
    await _use(pool)
    (raw,) = FakeRawPool.built
    assert raw.kwargs["min_size"] == 0
    assert raw.kwargs["max_size"] == 3
    assert raw.kwargs["max_idle"] == 42.0
    assert raw.kwargs["kwargs"] == {"connect_timeout": 7}
    assert raw.kwargs["check"] is FakeRawPool.check_connection
    assert raw.kwargs["open"] is False and raw.opened


@pytest.mark.asyncio
async def test_concurrent_first_use_opens_one_pool():
    (pool,) = _pools(PoolBudget(), "a")
    await asyncio.gather(*(_use(pool) for _ in range(8)))
    assert len(FakeRawPool.built) == 1
    assert pool.leases == 0


@pytest.mark.asyncio
async def test_cap_evicts_the_least_recently_used_idle_pool():
    clock = FakeClock()
    budget = PoolBudget(max_open_pools=2, clock=clock)
    a, b, c = _pools(budget, "a", "b", "c")
    await _use(a)
    clock.now += 1
    await _use(b)
    clock.now += 1
    await _use(a)  # a is now more recent than b
    clock.now += 1
    await _use(c)

    assert budget.open_pools == [a, c]
    assert b.state == "closed" and b.pool is None
    assert FakeRawPool.built[1].closed  # b's raw pool
    await _use(b)  # an evicted pool reopens on demand and evicts the next LRU (a)
    assert budget.open_pools == [c, b]
    assert len(FakeRawPool.built) == 4


@pytest.mark.asyncio
async def test_a_leased_pool_is_never_evicted_and_the_cap_is_soft_meanwhile():
    budget = PoolBudget(max_open_pools=1)
    a, b, c = _pools(budget, "a", "b", "c")
    async with a.connection():
        await _use(b)  # a holds a lease: b opens anyway, over the cap
        assert a.state == "open" and b.state == "open"
        assert len(budget.open_pools) == 2
    await _use(c)  # now a and b are idle: the cap is restored
    assert budget.open_pools == [c]


@pytest.mark.asyncio
async def test_idle_ttl_retires_pools_at_the_next_lease_anywhere():
    clock = FakeClock()
    budget = PoolBudget(idle_ttl=60, clock=clock)
    a, b = _pools(budget, "a", "b")
    await _use(a)
    await _use(b)
    clock.now += 30
    await _use(b)
    assert a.state == "open"  # 30 s idle, under the TTL
    clock.now += 31
    await _use(b)
    assert a.state == "closed" and b.state == "open"


@pytest.mark.asyncio
async def test_failed_open_is_retried_on_the_next_use():
    (pool,) = _pools(PoolBudget(), "a")
    with patch.object(FakeRawPool, "open", AsyncMock(side_effect=OSError("no route to host pw"))):
        with pytest.raises(ValueError, match="Connection attempt failed"):
            await _use(pool)
    assert pool.state == "failed" and pool.leases == 0
    assert FakeRawPool.built[0].closed
    await _use(pool)
    assert pool.state == "open"


@pytest.mark.asyncio
async def test_discovery_survives_eviction_and_reopen():
    budget = PoolBudget(max_open_pools=1)
    registry = DatabaseRegistry(
        [("gis", "postgresql://u:pw@h/gis"), ("lake", "postgresql://u:pw@h/lake")],
        budget=budget,
        allowed_extensions=("postgis",),
    )
    discover = AsyncMock(return_value=frozenset({"st_area"}))
    with patch("postgres_mcp.databases.discover_read_only_functions", discover):
        gis = await registry.ready("gis")
        await _use(gis.pool)
        await _use(registry.get("lake").pool)  # evicts gis
        assert gis.pool.state == "closed"
        gis = await registry.ready("gis")
    assert gis.extension_functions == frozenset({"st_area"})
    assert discover.await_count == 1  # once for gis, not once per open of its pool


@pytest.mark.asyncio
async def test_close_all_closes_every_open_pool():
    registry = DatabaseRegistry([("gis", "postgresql://u:pw@h/gis"), ("lake", "postgresql://u:pw@h/lake")])
    for db in registry:
        await _use(db.pool)
    await registry.close_all()
    assert all(raw.closed for raw in FakeRawPool.built)
    assert registry.budget.open_pools == []

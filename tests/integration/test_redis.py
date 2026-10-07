from __future__ import annotations

import pytest

from vikingrag.infrastructure.cache.redis import RedisClient

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_redis_ping(redis_client: RedisClient) -> None:
    assert await redis_client.ping() is True

"""Redis connectivity adapter - no caching policy in Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

from redis.asyncio import Redis

from vikingrag.domain.errors import InfrastructureError
from vikingrag.settings.config import RedisSettings


@dataclass(slots=True)
class RedisClient:
    client: Redis[str]

    async def ping(self) -> bool:
        try:
            result = await self.client.ping()
            return bool(result)
        except Exception as exc:
            raise InfrastructureError(f"Redis ping failed: {exc}") from exc

    async def close(self) -> None:
        # redis-py 5+ prefers aclose; stubs may lag behind the runtime API.
        close = getattr(self.client, "aclose", None) or self.client.close
        await close()


def create_redis_client(settings: RedisSettings) -> RedisClient:
    client: Redis[str] = Redis.from_url(settings.url, decode_responses=True)
    return RedisClient(client=client)

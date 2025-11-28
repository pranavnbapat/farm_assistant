# app/utils/response_cache.py

from __future__ import annotations

import hashlib
import json
import os
import redis
import time

from typing import Any, Iterable, Optional

from redis.exceptions import RedisError


_redis = None  # lazy-initialised client

def _r() -> redis.Redis:
    global _redis
    if _redis is None:
        url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
        _redis = redis.from_url(url, decode_responses=True)
    return _redis

def _norm(s: str) -> str:
    return " ".join((s or "").strip().split())

def hash_contexts(contexts: Optional[Iterable[str]]) -> str:
    h = hashlib.sha256()
    if contexts:
        for c in contexts:
            h.update(c.encode("utf-8"))
            h.update(b"\n---\n")
    return h.hexdigest()

def make_key(
    user_q: str,
    system_tag: str,
    model_id: str,
    params: dict[str, Any],
    ctx_hash: str = "",
    user_scope: str = "",
) -> str:
    payload = "\n".join([
        "P:" + _norm(user_q),
        "S:" + system_tag,
        "M:" + model_id,
        "J:" + json.dumps(params, sort_keys=True, separators=(",", ":")),
        "C:" + ctx_hash,
        "U:" + user_scope,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def get_cached(key: str) -> Optional[dict]:
    if os.getenv("CACHE_ENABLED", "true").lower() != "true":
        return None
    try:
        val = _r().get(f"cache:{key}")
    except RedisError:
        return None
    return json.loads(val) if val else None

def set_cached(key: str, answer: str, meta: dict[str, Any]) -> None:
    if os.getenv("CACHE_ENABLED", "true").lower() != "true":
        return
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "86400"))
    payload = {"answer": answer, "meta": meta, "created_at": int(time.time())}
    try:
        _r().set(f"cache:{key}", json.dumps(payload), ex=ttl)
    except RedisError:
        # swallow error – just means no caching
        return

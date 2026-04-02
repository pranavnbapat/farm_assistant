# Redis Deployment

This app uses Redis only for response caching.

What Redis adds:
- faster repeated answers
- lower repeated LLM load
- shared cache across users asking the same question

What Redis does not do here:
- chat persistence
- auth/session storage
- user profile storage

## Local Docker

The repo `docker-compose.yml` now includes:
- `farm_assistant`
- `redis`

Default cache wiring:
- `REDIS_URL=redis://redis:6379/0`
- `CACHE_ENABLED=true`
- `CACHE_TTL_SECONDS=86400`

Start locally with:

```bash
docker compose up -d
```

## Online Deployment

For a server that pulls `ghcr.io/pranavnbapat/farm_assistant:latest`, add a Redis service to the server-side compose file.

Example:

```yaml
services:
  farm_assistant:
    container_name: farm_assistant
    image: ghcr.io/pranavnbapat/farm_assistant:latest
    restart: always
    env_file:
      - .env
    depends_on:
      - redis
    networks:
      - traefik-net
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.farm-assistant.rule=Host(`farm-assistant.nexavion.com`)"
      - "traefik.http.routers.farm-assistant.entrypoints=websecure"
      - "traefik.http.routers.farm-assistant.tls=true"
      - "traefik.http.routers.farm-assistant.tls.certresolver=letsencrypt"
      - "traefik.http.services.farm-assistant.loadbalancer.server.port=8000"

  redis:
    container_name: farm_assistant_redis
    image: redis:7-alpine
    restart: always
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:

networks:
  traefik-net:
    external: true
```

## Online `.env`

Set:

```env
REDIS_URL=redis://redis:6379/0
CACHE_ENABLED=true
CACHE_TTL_SECONDS=86400
```

Because both containers are in the same compose project, `redis` works as the hostname.

## Deploy

After updating the server compose file:

```bash
docker compose pull farm_assistant
docker compose up -d redis farm_assistant
docker compose ps
docker logs -f farm_assistant
```

## Verify

From inside the app container or host:

```bash
docker exec -it farm_assistant_redis redis-cli ping
```

Expected:

```text
PONG
```

If Redis is unavailable, the app still works. It just skips caching.

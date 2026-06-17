# Farm Assistant — Arena Multi-Model Deployment

How the chatbot arena runs several LLMs as blind, side-by-side cards, all sharing
the **same** `farm_assistant` RAG pipeline (Standard routed RAG, Method 1).

See `CHATBOT_ARENA_EXPERIMENT_PLAN.md` for the experiment design (the 1 × 5
configuration set).

## Core idea

One **model card** in the arena = one running `farm_assistant` instance pointed at
one LLM. Every instance runs the **same image** (`ghcr.io/pranavnbapat/farm_assistant:latest`)
and the **same code** — they differ only by their `.env` (which LLM) and their
public URL. Retrieval, prompts, citation rules, and generation limits stay
identical across cards, so the only variable is the model.

```
                 eu-farmbook-frontend  (arena page; routes each card to a URL)
                          │
   ┌──────────────┬───────┴────────┬──────────────────┐
   ▼              ▼                ▼                  ▼
 Qwen          EuroLLM          OpenAI             Anthropic
 farm-assistant eurollm.…       gpt.…              claude.…
 .nexavion.com  nexavion        nexavion           nexavion
   │              │               │                  │
   ▼              ▼               ▼                  ▼
 RunPod #1     RunPod #2       api.openai.com    api.anthropic.com
 (Qwen vLLM)   (EuroLLM vLLM)  (no GPU)           (no GPU)
```

## Provider routing (additive, Qwen-safe)

Generation is provider-gated in `app/clients/vllm_client.py`:

- `LLM_PROVIDER=vllm` (default) → existing OpenAI-compatible `/v1/chat/completions`
  path. Used by Qwen, EuroLLM, and OpenAI (OpenAI *is* this API). **Unchanged.**
- `LLM_PROVIDER=anthropic` → Anthropic Messages API. The `anthropic` SDK is
  **lazy-imported** inside this branch only, so non-Anthropic instances never
  import it and pay no cost.

Settings live in `app/config.py`: `LLM_PROVIDER`, `ANTHROPIC_API_KEY`,
`ANTHROPIC_MODEL`, `ANTHROPIC_MAX_TOKENS` — all defaulted, so existing instances
that don't set them behave exactly as before. The Qwen instance, with no
`LLM_PROVIDER` in its `.env`, is byte-for-byte the same code path as today.

## Instances

| Card | Subdomain | `.env` file | `LLM_PROVIDER` | LLM endpoint | GPU |
|------|-----------|-------------|----------------|--------------|-----|
| Qwen3 | `farm-assistant.nexavion.com` (existing) | `.env` | (unset) | RunPod #1 | yes |
| EuroLLM | `eurollm.farm-assistant.nexavion.com` | `.env.eurollm` | `vllm` | RunPod #2 | yes |
| OpenAI | `gpt.farm-assistant.nexavion.com` | `.env.openai` | `vllm` | `api.openai.com` | no |
| Anthropic | `claude.farm-assistant.nexavion.com` | `.env.anthropic` | `anthropic` | `api.anthropic.com` | no |

The OpenAI / Anthropic / EuroLLM **app** containers are lightweight and run on the
existing nexavion host; only the model serving (Qwen, EuroLLM) needs a GPU/RunPod.

## `.env` files

Three per-instance env files live in this repo (git- and docker-ignored, so they
are **never** committed or baked into the image — copy them onto the server
manually, like `.env`):

- `.env.anthropic` — fill `ANTHROPIC_API_KEY`. `ANTHROPIC_MODEL=claude-haiku-4-5`.
- `.env.openai` — fill `VLLM_API_KEY` (OpenAI key). Keep `RUNPOD_VLLM_HOST` empty
  so it doesn't override `VLLM_URL=https://api.openai.com`. `VLLM_MODEL=gpt-5.4-mini`.
- `.env.eurollm` — fill `RUNPOD_VLLM_HOST`, `VLLM_MODEL`, `VLLM_API_KEY` from the
  **second** RunPod that serves EuroLLM.

Everything else in each file is copied from `.env` (OpenSearch creds, `FA_ENV=prd`
which auto-sets the prd `CHAT_BACKEND_URL`, generation limits) so the pipeline and
auth are identical across cards.

## DNS

Add A-records (GoDaddy), all pointing at the nexavion server IP `144.91.69.255`:

- `gpt.farm-assistant` → 144.91.69.255
- `eurollm.farm-assistant` → 144.91.69.255
- `claude.farm-assistant` → 144.91.69.255

Traefik + letsencrypt issue TLS automatically on first request.

## docker-compose

Add three services to the online `docker-compose.yml` (same image, distinct
`env_file`, distinct Traefik router/service name + `Host`, internal port 8000):

```yaml
  farm_assistant_anthropic:
    container_name: farm_assistant_anthropic
    image: ghcr.io/pranavnbapat/farm_assistant:latest
    restart: always
    env_file: [.env.anthropic]
    networks: [traefik-net]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fa-anthropic.rule=Host(`claude.farm-assistant.nexavion.com`)"
      - "traefik.http.routers.fa-anthropic.entrypoints=websecure"
      - "traefik.http.routers.fa-anthropic.tls=true"
      - "traefik.http.routers.fa-anthropic.tls.certresolver=letsencrypt"
      - "traefik.http.services.fa-anthropic.loadbalancer.server.port=8000"

  farm_assistant_openai:
    container_name: farm_assistant_openai
    image: ghcr.io/pranavnbapat/farm_assistant:latest
    restart: always
    env_file: [.env.openai]
    networks: [traefik-net]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fa-openai.rule=Host(`gpt.farm-assistant.nexavion.com`)"
      - "traefik.http.routers.fa-openai.entrypoints=websecure"
      - "traefik.http.routers.fa-openai.tls=true"
      - "traefik.http.routers.fa-openai.tls.certresolver=letsencrypt"
      - "traefik.http.services.fa-openai.loadbalancer.server.port=8000"

  farm_assistant_eurollm:
    container_name: farm_assistant_eurollm
    image: ghcr.io/pranavnbapat/farm_assistant:latest
    restart: always
    env_file: [.env.eurollm]
    networks: [traefik-net]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fa-eurollm.rule=Host(`eurollm.farm-assistant.nexavion.com`)"
      - "traefik.http.routers.fa-eurollm.entrypoints=websecure"
      - "traefik.http.routers.fa-eurollm.tls=true"
      - "traefik.http.routers.fa-eurollm.tls.certresolver=letsencrypt"
      - "traefik.http.services.fa-eurollm.loadbalancer.server.port=8000"
```

## Frontend wiring (`eu-farmbook-frontend`)

Code is already wired: `server.ts` resolves backend keys `openai_gpt` /
`anthropic_claude` / `eurollm` to URLs; `compare.ts` has metadata branches.
Set these env vars (local: `.env.local`; also in **production**):

```ini
FARM_ASSISTANT_ANTHROPIC_URL=https://claude.farm-assistant.nexavion.com
FARM_ASSISTANT_OPENAI_URL=https://gpt.farm-assistant.nexavion.com
FARM_ASSISTANT_EUROLLM_URL=https://eurollm.farm-assistant.nexavion.com
FARM_ASSISTANT_ARENA_VARIANTS=[{"id":"qwen3","backend":"um_qwen3","weight":1,"enabled":true},{"id":"eurollm","backend":"eurollm","weight":1,"enabled":true},{"id":"openai","backend":"openai_gpt","weight":1,"enabled":true},{"id":"anthropic","backend":"anthropic_claude","weight":1,"enabled":true}]
```

`FARM_ASSISTANT_ARENA_VARIANTS` must be a **single line**. All `enabled` variants
run as blind, shuffled A/B/C cards; `weight` is unused in the arena.

## Deploy steps

1. Fill the placeholders in `.env.anthropic`, `.env.openai`, `.env.eurollm`.
2. Stand up the second RunPod for EuroLLM; fill `.env.eurollm`.
3. Build & push the image (now includes `anthropic==0.109.2`).
4. On the server: add the 3 compose services, copy the 3 `.env` files, then
   `docker compose pull && docker compose up -d`.
   - The Qwen container pulls the same image but, with no `LLM_PROVIDER`, is
     unchanged. No downtime expected for it beyond the normal restart.
5. Set the four frontend vars in production and redeploy the frontend.
6. Smoke test: open the arena, ask one question. The only new code path is
   **Anthropic** — verify that card answers and cites sources like the others.

## Open items

- **`gpt-5.4-mini` params:** GPT-5-class models on Chat Completions may require
  `max_completion_tokens` instead of `max_tokens`, and may reject custom
  `temperature`/`top_p`. If the OpenAI instance 400s on first call, that's why —
  it needs an OpenAI-aware payload tweak in `vllm_client.py` (the Anthropic
  adapter is unaffected). Confirm the exact model string too.
- **Mistral:** the planned 5th family ("external, already there") is not yet
  wired — add its backend key to `FARM_ASSISTANT_ARENA_VARIANTS` and, if it runs
  its own RAG rather than this pipeline, note that as a methodology caveat.
- **Pin reproducibility:** `anthropic==0.109.2` is pinned. Freeze the exact
  OpenAI / EuroLLM / Claude model identifiers before the experiment starts, per
  the experiment plan.

## Provider gotchas (learned in production)

- **OpenAI GPT-5 family** (`gpt-5*`, incl. `gpt-5.4-mini`): the Chat Completions
  API rejects `max_tokens` (needs `max_completion_tokens`) and custom
  `temperature`/`top_p` (400). `build_gen_payload` auto-detects `gpt-5*` and
  switches param style; force it elsewhere with `OPENAI_GPT5_PARAM_STYLE=true`.
- **Anthropic model name**: the pipeline passes its configured vLLM model name
  (e.g. `qwen3-30b-a3b-awq`) into generation calls; Anthropic 404s on it. The
  adapter ignores any non-`claude-*` model and uses `ANTHROPIC_MODEL`.
- **Streaming error visibility**: `stream_generate` reads the error body while
  the stream is open and raises `Upstream LLM returned HTTP <code>: <body>`.
  Before this fix, a non-200 mid-stream surfaced only as the misleading
  "Attempted to access streaming response content, without having called
  `read()`." If you see that message, you're on an old image.

## Arena card visibility switch (admin)

A **Django superuser** can show/hide the closed-source cards (OpenAI, Anthropic)
for everyone, at runtime, from the arena page. State persists on
`ChatExperimentVariant.enabled` (no DB migration).

**Spans three repos:**

| Repo | Change |
|---|---|
| `django_euf_admin` | Views `arena_variants` (GET, flags + `is_admin`) and `arena_variant_toggle` (POST, **superuser-only**, 403 otherwise) in `euf/views/fastapi/UserChatV.py`; exported in `__init__.py`; URLs `chat/experiments/arena/variants/` and `.../toggle/`. |
| `farm_assistant` | Proxy routes `/chatbot/api/experiments/arena/variants` (GET) and `.../variants/toggle` (POST) in `app/main.py`. |
| `eu-farmbook-frontend` | `getArenaVariantFlags`/`applyArenaVisibility` filter the lineup in the compare run + stream routes (graceful fallback: visible on error); `/api/farm-assistant/arena/settings` proxy; admin panel in `FarmAssistantCompareShell.tsx`. |

**Authorization:** enforced server-side by `is_superuser` on the toggle endpoint.
The frontend only hides the panel for non-admins; it is not the security gate.

**Deploy order:** `django_euf_admin` (no migration) → `farm_assistant`
(build/push/pull) → `eu-farmbook-frontend` (commit/redeploy).

**Prerequisite:** the operator's account must be `is_superuser=True` in
`django_euf_admin`, or the panel won't appear and toggles return 403.

**Behaviour:** toggling a card off drops it from every user's shuffled lineup on
the next question; the other cards (Qwen/EuroLLM/Mistral) are never affected.

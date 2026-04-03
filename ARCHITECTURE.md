# Architecture Documentation

This document describes the current architecture of the Farm Assistant application as implemented in this repository.

## System Overview

Farm Assistant is a FastAPI-based RAG application with a browser chat UI. It uses:

- FastAPI for the web app and public API surface
- OpenSearch for document retrieval
- vLLM for answer generation
- a Django backend for authentication, chat/session persistence, user profiles, facts, and attachment records
- browser-native speech APIs for voice input and read-aloud

```text
Browser
  ├── Login page
  ├── Chat UI
  └── SSE chat stream
          │
          ▼
FastAPI App
  ├── HTML routes: /, /chat, /c/{session_id}
  ├── Chat stream + chat API routes
  ├── File upload routes
  └── Proxy/wrapper routes for Django-backed operations
          │
          ├── OpenSearch
          ├── vLLM
          └── Django backend
```

## Main Runtime Components

### FastAPI App (`app/main.py`)

`app/main.py` is the application entry point. It:

- creates the FastAPI app
- configures docs exposure
- mounts static assets and templates
- includes the ask and files routers
- serves the login and chat HTML routes
- exposes a public `/chatbot/api/*` wrapper surface
- proxies session, profile, fact, feedback, login, and logout operations to Django

### Ask Router (`app/routers/ask.py`)

This is the main chat execution path. It:

- accepts streaming chat requests
- extracts user identity from the auth token when present
- loads chat history and user profile context
- routes turns into the appropriate prompt mode
- retrieves OpenSearch context when needed
- streams generated output from vLLM
- extracts cited sources for display
- triggers profile/fact updates asynchronously after a completed turn

### Files Router (`app/routers/files.py`)

This handles PDF upload and deletion. It:

- accepts PDF uploads
- extracts text
- generates a summary
- stores parsed content in the in-memory document store
- upserts attachment metadata to the Django backend when auth and session context are available

### Search / Context / Prompt Services

The services layer is split by concern:

- `app/services/search_service.py`
  - OpenSearch request orchestration
  - result normalization and pagination handling
- `app/services/context_service.py`
  - document chunk scoring
  - context block formatting
  - source tracking for citations
- `app/services/prompt_service.py`
  - prompt construction for retrieval-backed turns
  - prompt construction for history-only recap turns
  - prompt construction for conversation-only turns
  - prompt construction for assistant-capability turns
  - title generation

### Chat History Service (`app/services/chat_history.py`)

This loads prior messages for a session from the Django backend and formats them for prompt inclusion. The current implementation is stateless on the FastAPI side; there is no server-side KV-cache reuse layer.

### User Profile Service (`app/services/user_profile_service.py`)

This service reads and updates personalization data through Django-backed APIs. It is used to:

- fetch stored user profile attributes
- fetch stored user facts
- build prompt context from profile/fact data
- infer and write updated profile/fact data from completed conversation turns

### Clients

- `app/clients/vllm_client.py`
  - OpenAI-compatible chat completion calls to vLLM
  - token streaming for SSE responses
- `app/clients/opensearch_client.py`
  - HTTP requests to OpenSearch
  - optional auth and SSL verification controls

## Request Flow

### Chat Turn

```text
User submits message
  → FastAPI receives /ask/stream request
  → auth token decoded if present
  → session history loaded from Django
  → current client-visible history merged where applicable
  → user profile/facts loaded from Django
  → turn mode resolved
  → if retrieval-backed: OpenSearch queried and context built
  → prompt built for the chosen mode
  → vLLM response streamed back over SSE
  → citations extracted and sources filtered
  → completed turn can be logged to Django
  → profile/fact update dispatched asynchronously
```

### Supported Turn Modes

The current chat flow uses dynamic turn modes rather than a single fixed prompt path:

- `normal`
  - retrieval-backed domain answer flow
- `history_only`
  - recap/history-summary flow using stored conversation only
- `conversation_only`
  - simple conversational continuation without retrieval
- `assistant_capabilities`
  - product/self-description flow without retrieval

This distinction exists to improve chat quality while preserving a single user-facing chat experience.

## Streaming Architecture

The chat UI uses Server-Sent Events.

Typical event flow:

- `status`
- `token`
- `sources`
- `stats`
- `timing`
- `done`
- `error`

The browser progressively renders the answer, then renders cited references and timing details when the stream completes.

## Browser UI Architecture

### Login / Auth

The browser login flow posts credentials to FastAPI, which forwards them to the Django backend. Tokens and identity metadata are stored in `localStorage` and reused for chat, session, and profile operations.

### Chat UI (`static/js/chat.js`)

The chat frontend is responsible for:

- session list rendering and grouping
- URL-based chat selection (`/c/{session_id}`)
- opening and restoring chat history
- starting SSE requests
- rendering streamed assistant messages
- rendering citations and markdown-like formatting
- inline assistant message actions:
  - copy
  - like
  - regenerate
  - read aloud
- sending message feedback
- sending completed-turn persistence payloads

### Voice (`static/js/voice.js`)

Voice behavior is browser-native:

- speech recognition fills the text input
- speech synthesis reads assistant answers aloud

There is no active server-side TTS router in the current implementation.

## Public API Surface

The app exposes a documented `/chatbot/api/*` surface that wraps the same underlying logic used by the web UI.

Current documented groups:

- Authentication
- Chats
- User Profile
- Files

The UI page routes `/`, `/chat`, and `/c/{session_id}` are intentionally hidden from OpenAPI.

See [API_ENDPOINT_SURFACE.md](./API_ENDPOINT_SURFACE.md) for the endpoint inventory.

## State and Persistence Model

### FastAPI Side

FastAPI is largely stateless across requests. It holds:

- in-memory PDF document data for uploaded files
- concurrency primitives
- transient request state during streaming

### Django Side

Django is the durable system of record for:

- authentication
- chat sessions
- chat messages
- message feedback
- user profiles
- user facts
- attachment records

## Configuration Architecture

Settings are defined in `app/config.py` via Pydantic Settings.

Key settings groups:

- environment selection: `FA_ENV`
- app/docs flags: `LOG_LEVEL`, `APP_TITLE`, `APP_VERSION`, `ENABLE_DOCS`, `ENABLE_REDOC`
- backend routing: `CHAT_BACKEND_URL`
- OpenSearch config
- vLLM config
- generation limits and defaults

Environment selection influences which Django backend base URL is used when an explicit `CHAT_BACKEND_URL` override is not provided.

## Security Notes

- JWTs are stored in browser `localStorage`
- the streaming endpoint currently accepts auth via query parameter for the SSE flow
- auth/session/profile operations are proxied through FastAPI to avoid frontend CORS issues
- SSL verification for OpenSearch is configurable

## Performance Characteristics

The current codebase optimizes for responsiveness through:

1. SSE token streaming
2. HTTP connection reuse via `httpx`
3. bounded generation concurrency
4. prompt/context truncation
5. direct browser rendering without a heavy frontend framework

There is no Redis-backed response cache in the current implementation.

## Deployment Model

### Local

- `./run.sh local`
- `./run.sh local dev`
- `./run.sh local prd`
- `./run.sh docker`

### Docker

The repo `docker-compose.yml` currently runs a single `farm_assistant` service built from this repository and configured through `.env`.

### External Dependencies

| Service | Purpose | Required |
|---------|---------|----------|
| vLLM | LLM inference | Yes |
| OpenSearch | Retrieval | Yes |
| Django backend | Auth, sessions, profile, feedback, attachments | Yes |

## Extension Points

Practical extension points in the current code:

1. add or swap a different retrieval backend in `search_service.py`
2. change ranking/context construction in `context_service.py`
3. refine prompt behavior in `prompt_service.py`
4. add more public wrapper endpoints in `app/main.py`
5. expand client-side message actions in `static/js/chat.js`
6. improve profile extraction heuristics or model-driven extraction in `user_profile_service.py`

## Monitoring and Debugging

Operational visibility currently comes from:

- `GET /health`
- FastAPI logs
- timing events in the SSE stream
- browser-side inspection of streaming/status behavior

## Personalization Model

User personalization is additive rather than mandatory. The assistant can still answer without a stored profile, but when profile/fact data exists it influences:

- explanation depth
- terminology level
- relevance of examples
- continuity across ongoing user-specific discussions

See [USER_DATA_CAPTURE_AND_USAGE.md](./USER_DATA_CAPTURE_AND_USAGE.md) for the data inventory.

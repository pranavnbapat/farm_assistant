# Farm Assistant RAG

An agricultural assistant that combines OpenSearch retrieval, vLLM generation, and a FastAPI web UI to provide source-cited answers for EU-FarmBook-related use cases.

## Overview

Farm Assistant is a conversational RAG application for agriculture, farming, and rural development topics. It serves a browser-based chat UI, proxies authentication and session storage to a Django backend, retrieves relevant source material from OpenSearch, and generates answers through a vLLM-compatible chat completion API.

## Features

- Streaming chat over Server-Sent Events
- Retrieval-Augmented Generation with source citations
- Multi-turn conversation continuity
- Session management with stored chat history and generated titles
- Dynamic turn handling for retrieval, recap/history-only, conversation-only, and capability questions
- User profile and fact extraction via the Django backend
- PDF upload and reuse as chat context
- Browser-based speech-to-text and read-aloud support
- JWT-based authentication proxied to Django
- Swagger/OpenAPI docs for the public chatbot API surface

## Tech Stack

- Backend: FastAPI on Python 3.11
- LLM inference: vLLM via OpenAI-compatible API
- Search: OpenSearch
- Frontend: vanilla JavaScript, HTML templates, SSE
- Voice: browser Web Speech APIs
- Session/auth/profile persistence: Django backend
- Containerization: Docker and Docker Compose

## Quick Start

### Prerequisites

- Python 3.11
- Docker and Docker Compose if you want containerized local runs
- Access to a vLLM endpoint
- Access to an OpenSearch instance with indexed agricultural content
- Access to the Django auth/chat backend for login, sessions, and profile data

### Environment Setup

1. Copy the sample file:

```bash
cp .env.sample .env
```

2. Set the required values in `.env`:

```bash
FA_ENV=local

OPENSEARCH_API_URL=https://your-opensearch-instance.com
OPENSEARCH_API_USR=your_username
OPENSEARCH_API_PWD=your_password
VERIFY_SSL=true

VLLM_URL=https://your-vllm-instance.com
VLLM_MODEL=qwen3-30b-a3b-awq
VLLM_API_KEY=your-api-key
```

### Running With Docker

```bash
docker compose up --build
```

The app will be available at `http://localhost:8000`.

### Running Locally

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` intentionally contains only the app's direct dependencies. `pip` resolves the transitive set.

3. Start the app:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running With `run.sh`

```bash
./run.sh local
```

Supported modes:

```bash
./run.sh local
./run.sh local dev
./run.sh local prd
./run.sh docker
```

- `local`: runs the FastAPI UI locally and uses `.env` backend values
- `local dev`: runs the FastAPI UI locally and points auth/chat traffic at the dev Django backend
- `local prd`: runs the FastAPI UI locally and points auth/chat traffic at the prd Django backend
- `docker`: runs the repo Docker Compose setup

## Configuration

All runtime configuration is read from environment variables via `app/config.py`.

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `FA_ENV` | Environment selector | `local` |
| `LOG_LEVEL` | Application log level | `INFO` |
| `APP_TITLE` | FastAPI app title | `Farm Assistant RAG` |
| `APP_VERSION` | FastAPI app version | `0.1.0` |
| `ENABLE_DOCS` | Enable Swagger UI | `false` |
| `ENABLE_REDOC` | Enable ReDoc | `false` |
| `CHAT_BACKEND_URL` | Explicit Django backend base URL override | empty |

### OpenSearch

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENSEARCH_API_URL` | OpenSearch base URL | required |
| `OPENSEARCH_API_USR` | Username | empty |
| `OPENSEARCH_API_PWD` | Password | empty |
| `VERIFY_SSL` | Verify SSL certificates | `true` |
| `OS_API_PATH` | Search API path | `/neural_search_relevant` |

### vLLM

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM base URL | `http://localhost:8000` |
| `VLLM_MODEL` | Model identifier | `qwen3-30b-a3b-awq` |
| `VLLM_API_KEY` | Optional API key | empty |
| `RUNPOD_VLLM_HOST` | Optional vLLM host override | empty |

### Generation

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_TOKENS` | Max tokens per answer | `768` |
| `TEMPERATURE` | Sampling temperature | `0.4` |
| `NUM_CTX` | Context window target | `4096` |
| `TOP_K` | Default retrieval depth | `5` |
| `MAX_CONTEXT_CHARS` | Prompt context cap | `24000` |

### Files / Voice

| Variable | Description | Default |
|----------|-------------|---------|
| `PIPER_MODELS_DIR` | Local model directory path used by the repo | `/app/models` |

## API Surface

Public API documentation is exposed through Swagger when docs are enabled:

- `GET /docs`
- `GET /redoc`

The documented public surface includes:

- `POST /chatbot/api/auth/login`
- `POST /chatbot/api/auth/logout`
- `GET /chatbot/api/chats`
- `POST /chatbot/api/chats`
- `GET /chatbot/api/chats/{session_id}`
- `PATCH /chatbot/api/chats/{session_id}`
- `DELETE /chatbot/api/chats/{session_id}`
- `POST /chatbot/api/chats/{session_id}/message/{message_id}/feedback`
- `GET /chatbot/api/users/me/profile`
- `PATCH /chatbot/api/users/me/profile`
- `GET /chatbot/api/users/me/facts`
- `POST /chatbot/api/users/me/facts`
- `POST /chatbot/api/users/me/profile/build`
- `POST /chatbot/api/files/pdf`
- `DELETE /chatbot/api/files/pdf/{doc_id}`

The browser UI routes `/`, `/chat`, and `/c/{session_id}` are intentionally hidden from OpenAPI.

See [API_ENDPOINT_SURFACE.md](./API_ENDPOINT_SURFACE.md) for the endpoint inventory.

## Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── logging_conf.py
│   ├── clients/
│   │   ├── opensearch_client.py
│   │   └── vllm_client.py
│   ├── routers/
│   │   ├── ask.py
│   │   └── files.py
│   └── services/
│       ├── chat_history.py
│       ├── context_service.py
│       ├── pdf_service.py
│       ├── prompt_service.py
│       ├── search_service.py
│       ├── user_profile_service.py
│       └── user_profile_service_v2.py
├── static/
│   ├── css/custom.css
│   └── js/
│       ├── auth.js
│       ├── chat.js
│       ├── login.js
│       └── voice.js
├── templates/
│   ├── ask_stream.html
│   └── login.html
├── testing/
├── models/
├── Dockerfile
├── docker-compose.yml
├── build_and_push.sh
├── pull_and_restart.sh
├── requirements.txt
└── run.sh
```

## Chat Behavior

The application does not rely on a single hardcoded chat path. The backend can route a turn into different prompt strategies depending on the request and available context:

- retrieval-backed answers for factual/domain questions
- history-only recap answers for conversation-summary turns
- conversation-only handling for simple follow-ups and acknowledgements
- assistant-capabilities answers for product/self-description questions

This keeps the chat flow closer to expected assistant behavior without exposing separate user-facing endpoints for each mode.

## Voice Behavior

Voice input and playback are browser-driven:

- speech recognition is handled in `static/js/voice.js`
- read-aloud uses browser speech synthesis from the chat UI

There is no active server-side TTS router in the current code.

## Development Notes

### Changing the Model

Set `VLLM_MODEL` to a model served by your vLLM endpoint.

### Customizing Prompts

Prompt construction lives in `app/services/prompt_service.py`.

### Extending Search

Search orchestration lives in `app/services/search_service.py` and context formatting in `app/services/context_service.py`.

### Session / Profile Backend

The FastAPI app depends on the Django backend for:

- authentication
- chat session persistence
- chat feedback persistence
- user profile storage
- user fact storage
- attachment upsert/fetch

## Additional Docs

- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [USER_DATA_CAPTURE_AND_USAGE.md](./USER_DATA_CAPTURE_AND_USAGE.md)
- [API_ENDPOINT_SURFACE.md](./API_ENDPOINT_SURFACE.md)

## License

This project is part of the EU-FarmBook initiative.

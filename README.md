# Farm Assistant RAG

An AI-powered agricultural assistant that uses Retrieval-Augmented Generation (RAG) to provide accurate, source-cited answers from EU-FarmBook's knowledge base.

## Overview

Farm Assistant is a conversational AI designed to help farmers, researchers, and agricultural professionals access relevant information from a curated collection of agricultural resources. The system combines semantic search with large language models to deliver context-aware responses with proper citations.

## Features

- **Streaming Chat Interface**: Real-time response streaming with Server-Sent Events (SSE)
- **Retrieval-Augmented Generation (RAG)**: Searches a knowledge base before generating answers
- **Source Citations**: All answers include inline citations linking to original sources
- **Multi-turn Conversations**: Maintains conversation context across multiple exchanges
- **Session Management**: Save, load, and manage chat sessions with automatic title generation
- **Intent Routing**: Automatically routes queries to RAG or direct LLM based on intent classification
- **Domain Restriction**: Strictly limited to agriculture, farming, and rural development topics
- **Multi-language Support**: Responds in the same language as the user's question (all 24 EU languages)
- **User Personalization**: Builds user profiles from conversations to tailor responses
- **Voice Support**: Speech-to-Text (STT) and Text-to-Speech (TTS) with 24 EU languages
- **Response Caching**: Redis-based caching for improved performance
- **Authentication**: Secure JWT-based authentication integrated with Django backend

## Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **LLM Inference**: vLLM with OpenAI-compatible API (primary), Ollama (legacy fallback)
- **Search Engine**: OpenSearch (neural search with vector embeddings)
- **Frontend**: Vanilla JavaScript with Server-Sent Events
- **Text-to-Speech**: Browser Web Speech API with Piper TTS (server-side option)
- **Caching**: Redis
- **Containerization**: Docker & Docker Compose

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Access to a vLLM instance (RunPod, self-hosted, or cloud provider)
- Access to an OpenSearch instance with indexed agricultural content
- Redis instance (optional, for caching)

### Environment Setup

1. Copy the sample environment file:
   ```bash
   cp .env.sample .env
   ```

2. Configure the required variables in `.env`:
   ```bash
   # OpenSearch Configuration
   OPENSEARCH_API_URL=https://your-opensearch-instance.com
   OPENSEARCH_API_USR=your_username
   OPENSEARCH_API_PWD=your_password
   VERIFY_SSL=true

   # vLLM Configuration (Primary)
   VLLM_URL=https://your-vllm-instance.com
   VLLM_MODEL=qwen3-30b-a3b-awq
   VLLM_API_KEY=your-api-key
   VLLM_MAX_MODEL_LEN=131072

   # Environment (local/dev/prd)
   FA_ENV=local
   ```

### Running with Docker

```bash
docker-compose up --build
```

The application will be available at `http://localhost:8000`.

### Running Locally (Development)

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Configuration

All configuration is managed through environment variables or the `.env` file:

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `FA_ENV` | Environment selector (local/dev/prd) | `local` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `APP_TITLE` | Application title | `Farm Assistant RAG` |
| `APP_VERSION` | Application version | `0.1.0` |

### vLLM Configuration (Primary)

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_URL` | vLLM API endpoint | `http://localhost:8000` |
| `VLLM_MODEL` | Model identifier | `qwen3-30b-a3b-awq` |
| `VLLM_API_KEY` | API key (if required) | `None` |
| `VLLM_MAX_MODEL_LEN` | Max context length (tokens) | `131072` |
| `RUNPOD_VLLM_HOST` | Alternative vLLM host | `None` |

### Legacy Ollama Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama API endpoint | `http://ollama:11434` |
| `LLM_MODEL` | Default LLM model | `deepseek-llm:7b-chat-q5_K_M` |

### LLM Generation Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `MAX_TOKENS` | Maximum tokens per generation | `-1` (unlimited) |
| `TEMPERATURE` | Sampling temperature | `0.4` |
| `NUM_CTX` | Context window size | `4096` |
| `TOP_K` | Number of sources to include | `5` |
| `MAX_CONTEXT_CHARS` | Max context size in characters | `24000` |

### OpenSearch Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENSEARCH_API_URL` | OpenSearch endpoint | Required |
| `OPENSEARCH_API_USR` | OpenSearch username | `None` |
| `OPENSEARCH_API_PWD` | OpenSearch password | `None` |
| `VERIFY_SSL` | Verify SSL certificates | `true` |
| `OS_API_PATH` | API path for neural search | `/neural_search_relevant` |

### Search Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `SEARCH_MODEL` | Search model name | `msmarco` |
| `SEARCH_INCLUDE_FULLTEXT` | Include full text in search | `true` |
| `SEARCH_SORT_BY` | Default sort order | `score_desc` |

### Intent Router

| Variable | Description | Default |
|----------|-------------|---------|
| `INTENT_ROUTER_URL` | Intent classification service | `https://intent-router.nexavion.com/intent-router` |

### Redis Cache

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | `redis://127.0.0.1:6379/0` |
| `CACHE_ENABLED` | Enable response caching | `true` |
| `CACHE_TTL_SECONDS` | Cache TTL in seconds | `86400` |
| `MAX_ACTIVE_GENERATIONS` | Max concurrent LLM requests | `3` |

### Piper TTS

| Variable | Description | Default |
|----------|-------------|---------|
| `PIPER_MODELS_DIR` | Directory for voice models | `/app/models` |

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Authentication
- `POST /api/login` - Authenticate user and obtain JWT tokens

### Chat
- `GET /ask/stream` - Streaming chat endpoint (SSE)
  - Query params: `q`, `session_id`, `model`, `max_tokens`, `temperature`, `page`, `k`, `top_k`, `auth_token`

### Summarization
- `POST /summarise` - Summarize text with custom prompt

### Text-to-Speech
- `POST /tts/stream` - Stream audio from text (NDJSON input)

### Session Management (Proxy to Django)
- `GET /proxy/chat/sessions/` - List user sessions
- `POST /proxy/chat/sessions/` - Create new session
- `GET /proxy/chat/sessions/{session_id}/` - Get session details
- `DELETE /proxy/chat/sessions/{session_id}/` - Delete session
- `POST /proxy/chat/log-turn/` - Log chat turn

### User Profile (Proxy to Django)
- `GET /proxy/chat/user/profile/` - Get user profile
- `PATCH /proxy/chat/user/profile/` - Update user profile
- `GET /proxy/chat/user/facts/` - Get user facts
- `POST /proxy/chat/user/facts/` - Add user fact

### Logout
- `POST /proxy/logout/` - Logout and invalidate tokens

### Documentation
- `GET /docs` - OpenAPI/Swagger documentation (when enabled)
- `GET /redoc` - ReDoc documentation (when enabled)

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Settings management (Pydantic)
│   ├── schemas.py              # Pydantic models for API
│   ├── logging_conf.py         # Logging configuration
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ask.py              # Chat/streaming endpoints
│   │   └── tts.py              # Text-to-speech endpoints
│   ├── clients/
│   │   ├── __init__.py
│   │   ├── vllm_client.py      # vLLM (OpenAI-compatible) client
│   │   ├── ollama_client.py    # Ollama LLM client (legacy)
│   │   ├── opensearch_client.py # OpenSearch client
│   │   └── hf_local_client.py  # HuggingFace local model client
│   ├── services/
│   │   ├── __init__.py
│   │   ├── chat_history.py     # Session management
│   │   ├── context_service.py  # Context building and ranking
│   │   ├── prompt_service.py   # Prompt templates
│   │   ├── search_service.py   # Search orchestration
│   │   └── user_profile_service.py  # User personalization
│   └── utils/
│       └── response_cache.py   # Redis caching utilities
├── templates/
│   ├── login.html              # Login page
│   └── ask_stream.html         # Chat interface
├── static/
│   ├── css/
│   │   └── custom.css          # Styles
│   └── js/
│       ├── auth.js             # Authentication handling
│       ├── login.js            # Login page logic
│       ├── chat.js             # Chat UI and streaming
│       ├── tts.js              # TTS callbacks (legacy)
│       └── voice.js            # Voice controls (TTS/STT)
├── models/                     # Piper TTS voice models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── run.sh
└── .env.sample
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architectural documentation.

## Key Features

### Domain Restriction
The assistant is strictly limited to agriculture-related topics. If a user asks about non-agricultural topics (programming, sports, politics, etc.), the assistant politely declines and offers to help with farming-related questions instead.

### Multi-language Support
The assistant detects the language of the user's question and responds in the same language. All 24 EU languages are supported for both text and voice interactions.

### User Personalization
The system builds user profiles from conversation history, including:
- Expertise level (beginner/expert)
- Farm type (organic, conventional, dairy, etc.)
- Region and crops
- Communication preferences

This information is used to personalize responses (e.g., simpler explanations for beginners, technical details for experts).

### Voice Support
- **Speech-to-Text**: Click the microphone button to speak your question
- **Text-to-Speech**: Click the 🔊 button on any assistant response to hear it
- **Language Selection**: Choose your preferred language for voice recognition
- **24 EU Languages**: Full support for all EU languages

## Development

### Adding New LLM Models

Models can be configured through the `VLLM_MODEL` environment variable. The application uses vLLM's OpenAI-compatible API, so any model served by your vLLM instance can be used.

### Customizing Prompts

System prompts and templates are defined in `app/services/prompt_service.py`. Modify these to change the assistant's behavior and response style.

### Extending Search

The search service in `app/services/search_service.py` can be extended to support additional search backends or ranking algorithms.

### Adding New TTS Voices

Voice models can be added to the `models/` directory and registered in `app/routers/tts.py`.

## License

This project is part of the EU-FarmBook initiative.

## Contributing

Contributions are welcome. Please ensure your code follows the existing patterns and includes appropriate tests.

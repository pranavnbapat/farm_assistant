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
- **Intent Routing**: Automatically routes queries to RAG or direct LLM based on intent
- **Text-to-Speech**: Built-in TTS support using Piper voice models
- **Response Caching**: Redis-based caching for improved performance
- **Authentication**: Secure JWT-based authentication integrated with Django backend

## Tech Stack

- **Backend**: FastAPI (Python 3.11)
- **LLM Inference**: vLLM with OpenAI-compatible API (RunPod/self-hosted)
- **Search Engine**: OpenSearch (neural search with vector embeddings)
- **Frontend**: Vanilla JavaScript with Server-Sent Events
- **Text-to-Speech**: Piper TTS
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

   # vLLM Configuration
   VLLM_URL=https://your-vllm-instance.com
   VLLM_MODEL=qwen3-30b-a3b-awq
   VLLM_API_KEY=your-api-key

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

| Variable | Description | Default |
|----------|-------------|---------|
| `FA_ENV` | Environment selector (local/dev/prd) | `local` |
| `VLLM_URL` | vLLM API endpoint | Required |
| `VLLM_MODEL` | Model identifier | `qwen3-30b-a3b-awq` |
| `VLLM_API_KEY` | API key (if required) | `None` |
| `VLLM_MAX_MODEL_LEN` | Max context length | `131072` |

Legacy Ollama settings (for backward compatibility):

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_URL` | Ollama API endpoint | `http://ollama:11434` |
| `LLM_MODEL` | Default LLM model | `deepseek-llm:7b-chat-q5_K_M` |
| `MAX_TOKENS` | Maximum tokens per generation | `-1` (unlimited) |
| `TEMPERATURE` | Sampling temperature | `0.4` |
| `NUM_CTX` | Context window size | `4096` |
| `OPENSEARCH_API_URL` | OpenSearch endpoint | Required |
| `OPENSEARCH_API_USR` | OpenSearch username | Optional |
| `OPENSEARCH_API_PWD` | OpenSearch password | Optional |
| `VERIFY_SSL` | Verify SSL certificates | `true` |
| `INTENT_ROUTER_URL` | Intent classification service | Required |
| `REDIS_URL` | Redis connection string | `redis://127.0.0.1:6379/0` |
| `CACHE_ENABLED` | Enable response caching | `true` |
| `LOG_LEVEL` | Logging level | `INFO` |

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Authentication
- `POST /api/login` - Authenticate user and obtain JWT tokens

### Chat
- `GET /ask/stream` - Streaming chat endpoint (SSE)
  - Query params: `q`, `session_id`, `model`, `max_tokens`, `temperature`, `page`, `k`, `top_k`

### Summarization
- `POST /summarise` - Summarize text with custom prompt

### Text-to-Speech
- `POST /tts/stream` - Stream audio from text (NDJSON input)

### Documentation
- `GET /docs` - OpenAPI/Swagger documentation (when enabled)
- `GET /redoc` - ReDoc documentation (when enabled)

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Settings management
│   ├── schemas.py           # Pydantic models
│   ├── logging_conf.py      # Logging configuration
│   ├── routers/
│   │   ├── ask.py           # Chat/streaming endpoints
│   │   └── tts.py           # Text-to-speech endpoints
│   ├── clients/
│   │   ├── ollama_client.py # Ollama LLM client
│   │   ├── opensearch_client.py # OpenSearch client
│   │   └── hf_local_client.py   # HuggingFace local model client
│   ├── services/
│   │   ├── chat_history.py  # Session management
│   │   ├── context_service.py   # Context building
│   │   ├── prompt_service.py    # Prompt templates
│   │   └── search_service.py    # Search orchestration
│   └── utils/
│       └── response_cache.py    # Redis caching utilities
├── templates/
│   ├── login.html           # Login page
│   └── ask_stream.html      # Chat interface
├── static/
│   ├── css/custom.css       # Styles
│   └── js/                  # Frontend scripts
│       ├── auth.js
│       ├── chat.js
│       ├── login.js
│       └── tts.js
├── models/                  # Piper TTS voice models
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.sample
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architectural documentation.

## Development

### Adding New LLM Models

Models can be configured through the `VLLM_MODEL` environment variable. The application uses vLLM's OpenAI-compatible API, so any model served by your vLLM instance can be used.

### Customizing Prompts

System prompts and templates are defined in `app/services/prompt_service.py`. Modify these to change the assistant's behavior and response style.

### Extending Search

The search service in `app/services/search_service.py` can be extended to support additional search backends or ranking algorithms.

## License

This project is part of the EU-FarmBook initiative.

## Contributing

Contributions are welcome. Please ensure your code follows the existing patterns and includes appropriate tests.

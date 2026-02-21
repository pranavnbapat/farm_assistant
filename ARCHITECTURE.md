# Architecture Documentation

This document describes the architecture of the Farm Assistant RAG application.

## System Overview

The Farm Assistant is a conversational AI system built on a **Retrieval-Augmented Generation (RAG)** architecture. It combines semantic search over an agricultural knowledge base with large language model inference to provide accurate, source-cited answers.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client (Browser)                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  Login Page │───▶│  Chat UI    │───▶│  SSE Stream (text + audio)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Farm Assistant (FastAPI)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │   Auth      │    │   Router    │    │    Streaming Handler        │ │
│  │  (/login)   │    │ (Intent)    │    │      (/ask/stream)          │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
│         │                  │                         │                  │
│         ▼                  ▼                         ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Processing Pipeline                         │   │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │   │
│  │  │  Search  │──▶│ Context  │──▶│  Prompt  │──▶│  Generation  │  │   │
│  │  │  (OS)    │   │  Build   │   │  Build   │   │   (Ollama)   │  │   │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  OpenSearch   │          │    Ollama     │          │    Redis      │
│  (Documents)  │          │    (LLM)      │          │   (Cache)     │
└───────────────┘          └───────────────┘          └───────────────┘
```

## Core Components

### 1. FastAPI Application (`app/main.py`)

The main entry point that:
- Initializes the FastAPI application with lifespan management
- Configures CORS middleware
- Mounts static files and templates
- Registers API routers
- Manages concurrency with semaphores

Key features:
- **Lifespan management**: Initializes resources (semaphores) on startup
- **Environment-based routing**: Different backend URLs for local/dev/production
- **JWT authentication**: Proxies authentication to Django backend

### 2. Request Flow

#### Chat Request Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Intent Router  │ ────► Routes to RAG or LLM-only
└─────────────────┘
    │
    ▼ (RAG Path)
┌─────────────────┐
│  OpenSearch     │ ────► Retrieve relevant documents
│  (Neural Search)│
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Context Service │ ────► Rank, filter, format documents
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Prompt Service  │ ────► Build structured prompt with context
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Ollama LLM    │ ────► Generate streaming response
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Response with  │ ────► Citations extracted and linked
│  Source Links   │
└─────────────────┘
```

#### Streaming Architecture

The application uses **Server-Sent Events (SSE)** for real-time streaming:

1. Client establishes SSE connection to `/ask/stream`
2. Server emits events:
   - `status`: Processing stage updates
   - `token`: Individual response tokens
   - `sources`: Retrieved source documents
   - `timing`: Performance metrics
   - `done`: Completion signal
   - `error`: Error information

### 3. Services Layer

#### Search Service (`app/services/search_service.py`)

Orchestrates document retrieval:
- Builds search payloads for OpenSearch
- Handles pagination
- Collects and deduplicates results
- Provides hit probing for intent classification

#### Context Service (`app/services/context_service.py`)

Transforms search results into LLM context:
- **Paragraph splitting**: Divides documents into semantic chunks
- **Ranking**: Scores paragraphs by query term overlap and keyword boosts
- **Context building**: Formats documents with metadata (title, source, license)
- **Source tracking**: Maintains citation indices for answer attribution

Key algorithm:
```python
# Token overlap scoring
q_tokens = extract_tokens(question)
for paragraph in document:
    p_tokens = extract_tokens(paragraph)
    score = overlap(q_tokens, p_tokens) * 10 + 
            boost_overlap(p_tokens, keywords) * 4 +
            position_bonus
```

#### Prompt Service (`app/services/prompt_service.py`)

Manages prompt templates:
- **RAG Prompt**: Includes sources, citation instructions, formatting rules
- **Generic Prompt**: For LLM-only queries without retrieval
- **Summary Prompt**: For text summarization tasks
- **Title Prompt**: For automatic session title generation

#### Chat History Service (`app/services/chat_history.py`)

Manages conversation state:
- Loads session history from Django backend
- Formats conversation for context window
- Saves LLM KV cache for efficient multi-turn
- Truncates history to fit context limits

### 4. Client Layer

#### vLLM Client (`app/clients/vllm_client.py`)

Interfaces with vLLM's OpenAI-compatible API:
- **Streaming generation**: Yields tokens via SSE as they're produced
- **Stateless API**: No KV cache reuse between turns (unlike Ollama)
- **Configurable options**: Temperature, max_tokens, top_p

Key parameters:
```python
{
    "temperature": 0.4,      # Balanced creativity
    "max_tokens": -1,        # Auto (vLLM decides)
    "top_p": 0.9,            # Nucleus sampling
    "model": "qwen3-30b"     # Model identifier
}
```

**Note**: vLLM uses the OpenAI chat completions format (`/v1/chat/completions`) rather than Ollama's generate endpoint.

#### OpenSearch Client (`app/clients/opensearch_client.py`)

Simple HTTP client for OpenSearch:
- Basic authentication support
- SSL verification configuration
- Request timeout handling

### 5. Routers

#### Ask Router (`app/routers/ask.py`)

Main chat endpoint implementing:

**Intent Routing**:
```python
intent = classify_intent(query)
if intent == "LLM_ONLY":
    # Direct LLM without search
else:
    # Full RAG pipeline
```

**Caching**:
- Cache key based on: question, context hash, model, parameters, history
- Redis-backed with configurable TTL
- Returns cached responses as simulated streams

**Concurrency Control**:
- Semaphore-based limiting (default: 3 concurrent generations)
- Queue status messages when at capacity

**Citation Processing**:
- Extracts citation patterns: `[1]`, `[1, 2]`, `(source 1)`
- Filters sources to only include cited documents
- Falls back to top-5 if no citations found

#### TTS Router (`app/routers/tts.py`)

Text-to-speech pipeline:
1. Accepts NDJSON text stream
2. Feeds to Piper TTS (PCM output)
3. Transcodes to WebM/Opus via FFmpeg
4. Streams audio chunks to client

### 6. Frontend Architecture

#### Authentication Flow (`static/js/auth.js`, `static/js/login.js`)

```
┌─────────┐     ┌─────────────┐     ┌────────────────┐
│  User   │────▶│  /api/login │────▶│ Django Backend │
└─────────┘     └─────────────┘     └────────────────┘
                                              │
┌─────────┐     ┌─────────────┐              ▼
│  Chat   │◀────│  JWT Store  │◀───── Access/Refresh Tokens
│  Page   │     │ (localStorage)│
└─────────┘     └─────────────┘
```

Token management:
- Access token stored in `localStorage`
- Refresh token for automatic renewal
- Automatic logout on token expiration

#### Chat Interface (`static/js/chat.js`)

Features:
- **Session management**: Create, load, delete chat sessions
- **Streaming display**: Real-time token rendering
- **Citation rendering**: Converts `[1]` to superscript links
- **TTS integration**: Browser speech synthesis with 🔊 button
- **Auto-focus**: `/` key focuses input from anywhere

Event handling:
```javascript
// SSE event types
es.addEventListener('status', handleStatus);
es.addEventListener('token', handleToken);
es.addEventListener('sources', handleSources);
es.addEventListener('timing', handleTiming);
es.addEventListener('done', handleDone);
```

## Data Flow

### Request Lifecycle

```
1. Request Received
   ├── Validate authentication (JWT in localStorage)
   ├── Check cache (Redis)
   └── Initialize streaming response

2. Intent Classification
   ├── Call intent router service
   └── Determine: RAG or LLM-only

3. Document Retrieval (RAG path)
   ├── Build search query
   ├── Query OpenSearch
   └── Receive ranked documents

4. Context Preparation
   ├── Split documents into paragraphs
   ├── Rank by relevance
   ├── Select top-k chunks
   └── Format with metadata

5. Prompt Construction
   ├── Load conversation history
   ├── Inject context blocks
   ├── Apply system instructions
   └── Build final prompt

6. Generation
   ├── Stream to Ollama
   ├── Yield tokens via SSE
   └── Capture KV cache

7. Post-Processing
   ├── Extract citations
   ├── Filter source list
   ├── Update session title (first turn)
   └── Cache response

8. Response Complete
   ├── Emit sources
   ├── Emit timing metrics
   └── Close SSE connection
```

## Configuration Architecture

Settings are managed via Pydantic Settings with environment variable support:

```python
class Settings(BaseSettings):
    # Environment
    FA_ENV: str  # local/dev/prd
    
    # OpenSearch
    OPENSEARCH_API_URL: str
    OPENSEARCH_API_USR: str | None
    OPENSEARCH_API_PWD: str | None
    
    # Ollama
    OLLAMA_URL: str
    LLM_MODEL: str
    TEMPERATURE: float
    NUM_CTX: int
    
    # Feature flags
    ENABLE_DOCS: bool
    ENABLE_REDOC: bool
```

Environment-specific backends:
| Environment | Django Backend | Auth Endpoint |
|-------------|----------------|---------------|
| local | http://127.0.0.1:8000 | /fastapi/login/ |
| dev | https://backend-admin.dev.farmbook.ugent.be | /fastapi/login/ |
| prd | https://backend-admin.prd.farmbook.ugent.be | /fastapi/login/ |

## Security Considerations

1. **Authentication**: JWT tokens with automatic refresh
2. **CORS**: Configured for all origins (adjust for production)
3. **SSL**: Configurable verification for external APIs
4. **Input Sanitization**: HTML escaping in frontend
5. **Rate Limiting**: Concurrency semaphore on generation

## Performance Optimizations

1. **Response Caching**: Redis-based with SHA-256 keys
2. **KV Cache Reuse**: Ollama context preserved across turns
3. **Streaming**: Immediate token delivery via SSE
4. **Connection Pooling**: HTTPX async clients with keep-alive
5. **Concurrent Search**: Parallel document fetching
6. **Lazy Loading**: Models loaded on first request

## Deployment

### Docker Architecture

```dockerfile
# Multi-stage build
FROM python:3.11-slim

# System dependencies: tini, build tools, OpenBLAS, FFmpeg
# Python dependencies from requirements.txt
# Piper voice models downloaded at build time

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### External Dependencies

| Service | Purpose | Required |
|---------|---------|----------|
| Ollama | LLM inference | Yes |
| OpenSearch | Document search | Yes |
| Django Backend | Auth & sessions | Yes |
| Redis | Response caching | No |
| Intent Router | Query classification | Yes |

## Extension Points

1. **New LLM Backends**: Implement `generate_once()` and `stream_generate()` interface
2. **Custom Search**: Extend `SearchService` with additional backends
3. **Additional TTS Voices**: Add entries to `VOICES` registry
4. **Custom Prompts**: Modify templates in `prompt_service.py`
5. **New Routers**: Add API endpoints in `app/routers/`

## Monitoring & Debugging

- **Health Check**: `GET /health`
- **API Docs**: `/docs` (when enabled)
- **Logs**: Structured logging with configurable levels
- **Timing Metrics**: Included in SSE `timing` events
- **Cache Stats**: Redis key inspection

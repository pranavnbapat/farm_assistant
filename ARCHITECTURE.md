# Architecture Documentation

This document describes the architecture of the Farm Assistant RAG application.

## System Overview

The Farm Assistant is a conversational AI system built on a **Retrieval-Augmented Generation (RAG)** architecture. It combines semantic search over an agricultural knowledge base with large language model inference to provide accurate, source-cited answers.

```
┌────────────────────────────────────────────────────────────────────────┐
│                           Client (Browser)                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐ │
│  │  Login Page │───▶│  Chat UI    │───▶│  SSE Stream (text + voice)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Farm Assistant (FastAPI)                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────────┐  │
│  │   Auth      │    │   Router    │    │    Streaming Handler        │  │
│  │  (/login)   │    │ (Intent)    │    │      (/ask/stream)          │  │
│  └─────────────┘    └─────────────┘    └─────────────────────────────┘  │
│         │                  │                         │                  │
│         ▼                  ▼                         ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      Processing Pipeline                        │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐  │    │
│  │  │  Search  │──▶│ Context  │──▶│  Prompt  │──▶│  Generation  │  │    │
│  │  │  (OS)    │   │  Build   │   │  Build   │   │   (vLLM)     │  │    │
│  │  └──────────┘   └──────────┘   └──────────┘   └──────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐
│  OpenSearch   │          │     vLLM      │
│  (Documents)  │          │    (LLM)      │
└───────────────┘          └───────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Django Backend (Auth/Sessions)                     │
│  ┌─────────────┐   ┌─────────────┐   ┌────────────────────────────────┐ │
│  │  Auth/JWT   │   │  Sessions   │   │      User Profile Service      │ │
│  └─────────────┘   └─────────────┘   └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. FastAPI Application (`app/main.py`)

The main entry point that:
- Initializes the FastAPI application with lifespan management
- Configures CORS middleware
- Mounts static files and templates
- Registers API routers
- Manages concurrency with semaphores
- Provides proxy endpoints for Django backend (avoids CORS issues)

Key features:
- **Lifespan management**: Initializes resources (semaphores) on startup
- **Environment-based routing**: Different backend URLs for local/dev/production
- **JWT authentication**: Proxies authentication to Django backend
- **Session proxying**: Proxies session management calls to Django backend

### 2. Request Flow

#### Chat Request Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Domain Check   │ ────► Validates question is agriculture-related
└─────────────────┘
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
│  Load Profile   │ ────► Get user expertise/farm type/preferences
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Prompt Service  │ ────► Build structured prompt with context + profile
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   vLLM LLM      │ ────► Generate streaming response
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Update Profile  │ ────► Extract facts, update user profile
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
   - `status`: Processing stage updates (Intent, Search, Context, LLM)
   - `token`: Individual response tokens
   - `sources`: Retrieved source documents (filtered by citations)
   - `stats`: LLM generation statistics
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
- **Context building**: Formats documents with metadata (title, source, license, project)
- **Source tracking**: Maintains citation indices for answer attribution
- **Retrieval quality estimation**: Simple token overlap scoring

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

Simple, natural prompt building:
- **Main Prompt**: Natural conversation format with sources, history, and profile
- **Summary Prompt**: For text summarization tasks
- **Title Prompt**: For automatic session title generation

Key features:
- **Natural conversation**: Full conversation history provided to LLM
- **No hardcoded rules**: LLM handles context, language, and relevance naturally
- **Personalization**: User profile included when available

#### User Profile Service (`app/services/user_profile_service.py`)

Manages user personalization:
- **Smart extraction**: Only processes substantive messages (skips greetings, meta-instructions)
- **Profile extraction**: Extracts user attributes from meaningful conversations
- **Fact storage**: Stores user facts in Django backend
- **Profile context building**: Creates context string for prompts

Extracted attributes:
- Expertise level (beginner/expert)
- Farm type (organic, conventional, dairy, etc.)
- Region and crops
- Communication preferences
- Topics of interest

Skipped messages (not profiled):
- Greetings ("Hi", "How are you?")
- Meta-instructions ("Answer in English")
- Short acknowledgments ("OK", "Thanks")
- Personal questions ("Who created you?")

#### Chat History Service (`app/services/chat_history.py`)

Manages conversation state:
- Loads session history from Django backend
- Formats conversation for context window
- Saves LLM KV cache (legacy, for Ollama)
- Truncates history to fit context limits

### 4. Client Layer

#### vLLM Client (`app/clients/vllm_client.py`)

Primary LLM interface using OpenAI-compatible API:
- **Streaming generation**: Yields tokens via SSE as they're produced
- **Stateless API**: No KV cache reuse between turns (unlike Ollama)
- **Configurable options**: Temperature, max_tokens, top_p
- **API key support**: For authenticated vLLM endpoints

Key parameters:
```python
{
    "temperature": 0.4,      # Balanced creativity
    "max_tokens": -1,        # Auto (vLLM decides)
    "top_p": 0.9,            # Nucleus sampling
    "model": "qwen3-30b"     # Model identifier
}
```

**Note**: vLLM uses the OpenAI chat completions format (`/v1/chat/completions`).

#### OpenSearch Client (`app/clients/opensearch_client.py`)

Simple HTTP client for OpenSearch:
- Basic authentication support
- SSL verification configuration
- Request timeout handling

### 5. Routers

#### Ask Router (`app/routers/ask.py`)

Main chat endpoint implementing:

**Request Processing**:
```python
# Natural conversation flow - no hardcoded logic
# 1. Always search (LLM decides if results are useful)
# 2. Send full conversation history to LLM
# 3. LLM naturally handles context and continuity
```

**User Extraction**:
```python
user_uuid = _extract_user_uuid_from_token(auth_token)
# Extracts UUID from JWT for profile loading
```

**Natural Conversation Flow**:
```python
# No separate intent classification
# No hardcoded "ambiguous response" detection
# Full history + search results → LLM handles naturally
prompt = build_prompt(contexts, question, history, profile)
response = await stream_generate(prompt)
```

**Concurrency Control**:
- Semaphore-based limiting (default: 3 concurrent generations)
- Queue status messages when at capacity

**Citation Processing**:
- Extracts citation patterns: `[1]`, `[1, 2]`, `(source 1)`, `source 2`
- Filters sources to only include cited documents
- Falls back to top-5 if no citations found

**Profile Update** (fire-and-forget):
```python
asyncio.create_task(
    UserProfileService.process_conversation_turn(
        user_uuid, session_id, user_q, full_text, auth_token
    )
)
```

#### TTS Router (`app/routers/tts.py`)

Server-side text-to-speech pipeline:
1. Accepts NDJSON text stream
2. Feeds to Piper TTS (PCM output)
3. Transcodes to WebM/Opus via FFmpeg
4. Streams audio chunks to client

Voice registry supports:
- `en-gb-male` (Alan)
- `en-gb-female` (Alba)
- `en-gb-neutral` (Alan)

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
- Session management via Django backend proxy

#### Chat Interface (`static/js/chat.js`)

Features:
- **Session management**: Create, load, delete chat sessions
- **Streaming display**: Real-time token rendering
- **Citation rendering**: Converts `[1]` to superscript links
- **Voice integration**: Browser speech synthesis with 🔊 button
- **Auto-focus**: `/` key or any printable key focuses input from anywhere
- **Think time display**: Shows server processing duration

Event handling:
```javascript
// SSE event types
es.addEventListener('status', handleStatus);
es.addEventListener('token', handleToken);
es.addEventListener('sources', handleSources);
es.addEventListener('timing', handleTiming);
es.addEventListener('done', handleDone);
```

#### Voice Support (`static/js/voice.js`)

Comprehensive voice support for 24 EU languages:

**TTS Manager**:
- Browser Web Speech API integration
- Automatic language detection from text
- Manual language override
- Play/pause/stop controls
- Best voice selection per language

**STT Manager**:
- Web Speech Recognition API
- Continuous listening mode
- Interim results display
- 24 EU language support
- Touch-friendly mobile interface

## Data Flow

### Request Lifecycle

```
1. Request Received
   ├── Validate authentication (JWT in localStorage/query param)
   ├── Extract user UUID from token
   └── Initialize streaming response

2. Domain Validation
   ├── Check for non-agriculture keywords
   └── Log result for analytics

3. Intent Classification
   ├── Call intent router service
   └── Determine: RAG or LLM-only

4. User Profile Loading (async)
   ├── Fetch profile from Django
   └── Build profile context string

5. Document Retrieval (RAG path)
   ├── Build search query
   ├── Query OpenSearch
   └── Receive ranked documents

6. Context Preparation
   ├── Split documents into paragraphs
   ├── Rank by relevance
   ├── Select top-k chunks
   └── Format with metadata

7. Prompt Construction
   ├── Load conversation history
   ├── Inject profile context
   ├── Inject context blocks
   ├── Apply domain restriction rules
   ├── Apply language rules
   └── Build final prompt

8. Generation
   ├── Stream to vLLM
   ├── Yield tokens via SSE
   └── Capture statistics

9. Post-Processing
   ├── Extract citations
   ├── Filter source list
   ├── Update session title (first turn)
   ├── Update user profile (fire-and-forget)
   └── Cache response

10. Response Complete
    ├── Emit sources (cited only)
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
    
    # vLLM (Primary)
    VLLM_URL: str
    VLLM_MODEL: str
    VLLM_API_KEY: str | None

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

1. **Authentication**: JWT tokens with automatic refresh via Django backend
2. **CORS**: Configured for all origins (adjust for production)
3. **SSL**: Configurable verification for external APIs
4. **Input Sanitization**: HTML escaping in frontend
5. **Rate Limiting**: Concurrency semaphore on generation
6. **Token Storage**: Tokens stored in localStorage (consider httpOnly cookies for enhanced security)

## Performance Optimizations

1. **Streaming**: Immediate token delivery via SSE
2. **Connection Pooling**: HTTPX async clients with keep-alive
3. **Concurrent Search**: Parallel document fetching
4. **Lazy Loading**: Models loaded on first request
5. **Context Truncation**: History limited to fit context window
6. **Profile Caching**: User profiles cached per request

## Deployment

### Docker Architecture

```dockerfile
# Multi-stage build
FROM python:3.11-slim

# System deps: tini + build toolchain + OpenBLAS runtime + FFmpeg
# Python dependencies from requirements.txt
# Piper voice models downloaded at build time

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", 
     "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-keep-alive", "120"]
```

### External Dependencies

| Service | Purpose | Required |
|---------|---------|----------|
| vLLM | LLM inference | Yes (primary) |
| OpenSearch | Document search | Yes |
| Django Backend | Auth, sessions, profiles | Yes |
| Intent Router | Query classification | No (falls back to RAG) |
| Ollama | LLM inference (legacy) | No |

## Extension Points

1. **New LLM Backends**: Implement `generate_once()` and `stream_generate()` interface in `app/clients/`
2. **Custom Search**: Extend `SearchService` with additional backends
3. **Additional TTS Voices**: Add entries to `VOICES` registry in `app/routers/tts.py`
4. **Custom Prompts**: Modify templates in `prompt_service.py`
5. **New Routers**: Add API endpoints in `app/routers/`
6. **Enhanced Profile Extraction**: Implement LLM-based extraction in `user_profile_service.py`

## Monitoring & Debugging

- **Health Check**: `GET /health`
- **API Docs**: `/docs` (when enabled)
- **Logs**: Structured logging with configurable levels
- **Timing Metrics**: Included in SSE `timing` events
- **Profile Logging**: User profile updates logged

## Multi-language Architecture

The system supports all 24 EU languages through:

1. **Language Detection**: LLM instructed to respond in user's question language
2. **Translation**: Sources may be in different language; LLM translates content
3. **Voice Support**: 
   - STT: Web Speech Recognition with 24 EU language codes
   - TTS: Browser voices with automatic language detection
4. **Prompt Rules**: Explicit instructions in prompts to maintain language consistency

## Personalization Architecture

User profiles enhance responses without sending full chat history:

```
Conversation → Extraction → Profile Store → Context Building → Prompt Injection
     │              │              │                │                  │
     ▼              ▼              ▼                ▼                  ▼
User asks    Keywords/LLM    Django API      Format as string    "User Profile:
question     extract facts    (PostgreSQL)    for prompt          - Expertise: expert
                                                              - Farm: organic dairy"
```

Profile attributes influence:
- Explanation depth (beginner vs expert)
- Technical terminology usage
- Organic vs conventional recommendations
- Regional considerations

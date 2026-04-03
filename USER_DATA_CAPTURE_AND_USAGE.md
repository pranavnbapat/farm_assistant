# User Data Capture And Usage (Code-Based Inventory)

This document summarizes what user information is captured in this codebase and how it is used, based on repository inspection.

## Scope And Caveat

- This FastAPI app proxies many operations to a Django backend (`/chat/*`, `/fastapi/*`).
- This document covers data handling visible in this repository.
- The Django backend may capture/store additional data not visible here.

## 1) Authentication Data

### Data Captured

- `email`
- `password`

### Where Captured

- Login form inputs:
  - `templates/login.html`
  - `#login-email`, `#login-password`
- Frontend submit logic:
  - `static/js/login.js`
  - `static/js/auth.js`

### How It Is Used

- Sent to FastAPI `POST /api/login`.
- Forwarded to upstream auth backend (`/fastapi/login/` on Django backend).
- Response is expected to include tokens and user UUID.

### Storage/Retention In This Repo

- Credentials are not persisted locally by this app.
- Email is logged on login attempt/success/failure in backend logs (`app/main.py`).

## 2) Tokens And User Identity

### Data Captured/Derived

- `access_token` (JWT)
- `refresh_token`
- `user_uuid`
- `user_email` (and legacy `fa_email`)
- Display name derived from JWT claims:
  - `first_name` or `given_name` or first token of `name`

### Where Captured/Stored

- Browser `localStorage`:
  - `fa_access_token`
  - `fa_refresh_token`
  - `fa_user_uuid`
  - `fa_user_email`
  - `fa_email` (legacy)

### How It Is Used

- Auth headers to backend proxy/session endpoints:
  - `Authorization: Bearer <token>`
  - `X-Refresh-Token: <refresh_token>`
- UI greeting:
  - Welcome name + email shown in chat header.
- Server-side user scoping:
  - JWT is decoded to extract `uuid`/`user_id`/`sub` for profile and file ownership.
- Logout:
  - `/proxy/logout/` called with email, then local auth keys removed.

## 3) Chat Content, Sessions, And Turn Metadata

### Data Captured

- User question text (`q`)
- Assistant response text
- Session UUID and session title
- Optional follow-up hint
- Model selection
- Latency timing
- Attachment metadata per turn (`doc_id`, `filename`)

### Where Captured

- Frontend chat flow:
  - `static/js/chat.js`
- SSE request:
  - `/ask/stream` query params include `q`, `session_id`, `doc_ids`, `followup_hint`, `auth_token`.
- Turn logging:
  - `POST /proxy/chat/log-turn/` with:
    - `session_uuid`
    - `user_message`
    - `assistant_message`
    - `meta` (model, latency, attachments)

### How It Is Used

- Retrieval query construction and OpenSearch requests.
- Prompt assembly with:
  - relevant sources,
  - previous conversation history,
  - user profile context (if available).
- LLM inference via vLLM (`/v1/chat/completions`).
- Session continuity:
  - list/create/open/rename/delete sessions via proxied backend endpoints.

## 4) Conversation History

### Data Accessed

- Prior session messages (`role`, `content`, optional metadata from backend payload).

### Where Accessed

- `app/services/chat_history.py`
- Fetch from Django backend:
  - `GET /chat/sessions/{session_id}/`

### How It Is Used

- Formatted into compact text and appended to LLM prompt as previous conversation.

## 5) User Profile And Inferred Personalization Data

### Data Captured/Derived From User Messages

- `expertise_level`
- `farm_type`
- `region`
- `crops_list`
- `common_topics`
- Extracted facts:
  - categories like `issue`, `preference`, `tool`, `experience`, `location`, `farm_type`
  - `fact_text`
  - `confidence_score`
  - `source_session_uuid`

### Where Implemented

- Active import path in ask router:
  - `from app.services.user_profile_service import UserProfileService`
- Note:
  - `user_profile_service_v2.py` exists and aliases `UserProfileService = UserProfileServiceV2`,
    but `ask.py` currently imports from `user_profile_service.py`.

### How It Is Used

- Read profile and facts from backend:
  - `/chat/user/profile/`
  - `/chat/user/facts/`
- Build a profile context snippet and inject into prompt.
- After each turn, parse user message and update profile/facts (fire-and-forget).

## 6) Uploaded PDF Data

### Data Captured

- Raw uploaded PDF bytes
- Filename
- Owner identifier (`user_uuid` from token; fallback `anonymous`)
- Generated `doc_id`
- Extracted text from PDF
- LLM-generated summary
- Processing status/error

### Where Captured/Stored

- Upload endpoint:
  - `POST /files/pdf`
- Temporary disk write:
  - `/tmp/farm_assistant_uploads`
- In-memory document store:
  - `_DOC_STORE` in `app/services/pdf_service.py`
- After processing, binary file is deleted from disk and `path` cleared.

### How It Is Used

- PDF text/summary is converted into retrieval context for answers.
- When session + auth are available, attachment record is upserted to backend with:
  - `attachment_uuid`, `session_uuid`, `filename`, `summary`,
  - truncated `extracted_text` (up to 20,000 chars),
  - extraction metadata.
- Existing session attachments can be re-fetched and reused as context.

## 7) Voice Data (Browser Speech Recognition/TTS)

### Data Captured

- Microphone speech transcript (browser STT result text).

### Where Captured

- `static/js/voice.js` (`SpeechRecognition` / `webkitSpeechRecognition`)

### How It Is Used

- Transcript text is inserted into the chat input field.
- It is only sent to backend when user submits chat.
- TTS plays assistant text locally in browser; no server TTS payload in this path.

## 8) Data Sent To External/Internal Services

### Django Backend (proxied)

- Login credentials to `/fastapi/login/`.
- Session management requests (`/chat/sessions/*`).
- Turn logs (`/chat/log-turn/`) with message content and metadata.
- Profile/facts requests (`/chat/user/profile/`, `/chat/user/facts/`).
- Attachment upsert/fetch (`/chat/attachments/*`).

### OpenSearch API

- Search payload includes `search_term` (user question or normalized variant).

### vLLM API

- Prompt content sent as chat message payload includes:
  - user question,
  - selected source context,
  - previous conversation snippet,
  - user profile context (if present).

## 9) Notable Privacy/Security Observations From Code

- Tokens are stored in `localStorage` (XSS-sensitive storage location).
- SSE auth token is passed via URL query parameter (`auth_token`) for `/ask/stream`.
- Login email is written to application logs.
- PDF extracted text is retained in memory and may be persisted to backend attachments.
- This repo does not show explicit consent capture, data deletion workflows for chat/profile, or retention limits on backend-stored chat/profile data.

# API Endpoint Surface

This document lists the additive API endpoints exposed for the current Farm Assistant flow. These endpoints are wrappers or aliases around the same logic already used by the web UI.

## Authentication

- `POST /chatbot/api/auth/login`
  - Same login flow as the existing UI.
  - Body: `email`, `password`

- `POST /chatbot/api/auth/logout`
  - Same backend logout flow.
  - Body: `email`

## Chats

- `GET /chatbot/api/chats`
  - List chats/sessions.

- `POST /chatbot/api/chats`
  - Create a chat/session.
  - Body: `title`, `metadata`

- `GET /chatbot/api/chats/{session_id}`
  - Get one chat/session with messages.

- `PATCH /chatbot/api/chats/{session_id}`
  - Update a chat/session.
  - Body: `title`, `metadata`

- `DELETE /chatbot/api/chats/{session_id}`
  - Delete a chat/session.

- `POST /chatbot/api/chats/message`
  - Send a message without an existing session.
  - Body matches current chat params:
    - `q`
    - `page`
    - `k`
    - `top_k`
    - `max_tokens`
    - `temperature`
    - `model`
    - `followup_hint`
    - `doc_ids`

- `POST /chatbot/api/chats/{session_id}/message`
  - Send a message to an existing session.
  - Same body as above.
  - Returns the same SSE response flow as the current chat path.

- `GET /chatbot/api/chats/message/stream`
  - Query-param alias of the original streaming endpoint for non-session use.

- `GET /chatbot/api/chats/{session_id}/message/stream`
  - Query-param alias of the original streaming endpoint for existing sessions.

- `POST /chatbot/api/chats/log-turn`
  - Persist a completed chat turn.
  - Body:
    - `session_uuid`
    - `user_message`
    - `assistant_message`
    - `meta`

- `POST /chatbot/api/chats/{session_id}/log-turn`
  - Same as above, with session in the path.

## User Profile

- `GET /chatbot/api/users/me/profile`
  - Fetch current user profile.

- `PATCH /chatbot/api/users/me/profile`
  - Update current user profile.
  - Body may include:
    - `expertise_level`
    - `farm_type`
    - `region`
    - `preferred_language`
    - `communication_style`
    - `crops_list`
    - `common_topics`
    - `query_languages`
    - `total_queries`

- `POST /chatbot/api/users/me/profile/build`
  - Run the same profile-building logic used after chat turns.
  - Body:
    - `session_uuid`
    - `user_message`
    - `assistant_message`

- `GET /chatbot/api/users/me/facts`
  - Fetch stored profile facts.
  - Query params:
    - `category`
    - `limit`

- `POST /chatbot/api/users/me/facts`
  - Add a fact manually.
  - Body:
    - `fact_category`
    - `fact_text`
    - `confidence_score`
    - `source_session_uuid`

## Files

- `POST /chatbot/api/files/pdf`
  - Upload a PDF for chat context.

- `DELETE /chatbot/api/files/pdf/{doc_id}`
  - Delete an uploaded PDF.

## Notes

- Existing UI routes and proxy routes are still present and unchanged in behavior.
- The old internal paths are hidden from OpenAPI where a cleaner documented alias now exists.
- The generated docs are available through the existing FastAPI docs path:
  - `/docs`

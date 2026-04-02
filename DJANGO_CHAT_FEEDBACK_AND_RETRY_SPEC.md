# Django Chat Feedback And Retry Spec

This document defines the recommended Django-side implementation for the Farm Assistant chat actions that should be persisted and measured.

Scope:
- keep `like`
- keep `try_again`
- do not treat `copy`, `read aloud`, or other utility actions as analytics
- do not add denormalized counter fields

## Recommended Analytics Model

Use raw events or raw feedback records, not counters.

Why:
- counters drift
- counters are harder to audit
- raw events can always be aggregated later
- product definitions change over time

For the current scope:
- `like` should be stored as message feedback
- `try_again` should be stored as turn metadata or as an event record

## Existing Models

Based on the Django models you shared:

- `ChatSession`
- `ChatMessage`
- `ChatAttachment`
- `ChatFeedback`
- `ChatUserProfile`
- `ChatUserFact`

These are sufficient to implement `like` professionally without adding counter columns.

## Like Handling

### Recommended storage

Use `ChatFeedback` as the source of truth for likes.

Suggested semantic mapping:
- `rating = 1` means liked
- no row means no like

Because `ChatFeedback.message` is a `OneToOneField`, the design currently supports one feedback record per message.

That is acceptable if:
- the chatbot message is evaluated only by the owning user
- you do not need multiple reactions from multiple users to the same message

### Endpoint

Implement:

- `POST /chat/sessions/{session_uuid}/message/{message_id}/feedback/`

Request body:

```json
{
  "feedback": "up",
  "meta": {
    "action": "like"
  }
}
```

Accepted values:
- `"up"`
- `"none"`

If you still want backward compatibility with future negative feedback, you may also accept `"down"`, but for the current UI it is not needed.

### Service behavior

Validation:
- session must exist
- message must exist
- message must belong to session
- message role should be `assistant`
- request user should match the session owner unless staff/admin override exists

Persistence:
- on `"up"`:
  - create or update `ChatFeedback`
  - set `rating = 1`
  - optionally keep a small `comment` or `null`
- on `"none"`:
  - delete the `ChatFeedback` row, or set `rating = null`

Recommendation:
- delete the row on `"none"`
- this keeps the meaning of an existing row clear: explicit positive feedback

### Query examples

Likes for one message:

```python
ChatFeedback.objects.filter(message_id=message_id, rating=1).count()
```

Likes for one session:

```python
ChatFeedback.objects.filter(
    message__session__session_uuid=session_uuid,
    rating=1,
).count()
```

Liked assistant messages for one user:

```python
ChatFeedback.objects.filter(
    user=user,
    rating=1,
    message__role="assistant",
)
```

## Try Again Handling

### Recommended storage

Do not create a counter field.

Use one of these two options:

### Option A: use `ChatMessage.extra`

This is the minimal-change option and is acceptable for the current scope.

When a user clicks `Try again`, the next generated assistant message should be logged as a normal new assistant message, with metadata such as:

```json
{
  "action": "retry",
  "retry_of_message_id": 123,
  "model": "qwen3-30b-a3b-awq",
  "latency_ms": 2310
}
```

This gives you:
- exact retry linkage
- no schema migration
- queryable metadata

Tradeoff:
- analytics queries on JSON are less clean than a dedicated event table

### Option B: add a `ChatEvent` model

This is the better long-term architecture if you expect more event analytics later.

Recommended model:

```python
class ChatEvent(TimeStampedSoftDeleteModel):
    class Meta:
        db_table = "chat_event"
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["event_type", "created_at"]),
            models.Index(fields=["message", "created_at"]),
        ]

    EVENT_TYPE_CHOICES = (
        ("retry", "Retry"),
        ("like", "Like"),
    )

    id = models.AutoField(primary_key=True)

    session = models.ForeignKey(
        ChatSession,
        on_delete=models.CASCADE,
        related_name="events",
    )

    message = models.ForeignKey(
        ChatMessage,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="events",
    )

    user = models.ForeignKey(
        DefaultAuthUserExtend,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="chat_events",
    )

    event_type = models.CharField(max_length=20, choices=EVENT_TYPE_CHOICES)
    extra = models.JSONField(default=dict, blank=True)
```

This is cleaner if you expect to expand analytics later.

### Recommended choice

For the current project stage:
- keep `like` in `ChatFeedback`
- keep `try_again` in `ChatMessage.extra`

That is the best cost/benefit choice.

## Retry Semantics

Important:
- `Try again` should not mutate the previous assistant message in storage
- it should create a new assistant message tied to the same user prompt or same turn context

Why:
- preserves auditability
- avoids hidden rewriting of conversation history
- aligns better with append-only chat logs

If the UI visually replaces the last answer later, that should still map to a new stored assistant message underneath, unless you deliberately design versioned messages.

## Recommended Django Service Functions

### Feedback service

```python
def set_message_feedback(*, session_uuid, message_id, user, feedback_value):
    session = ChatSession.objects.get(session_uuid=session_uuid, is_deleted=False)
    message = ChatMessage.objects.get(
        id=message_id,
        session=session,
        role="assistant",
        is_deleted=False,
    )

    if session.user_id and session.user_id != user.id:
        raise PermissionDenied("Not allowed to rate this message")

    if feedback_value == "up":
        feedback, _ = ChatFeedback.objects.update_or_create(
            message=message,
            defaults={
                "user": user,
                "rating": 1,
                "comment": None,
            },
        )
        return feedback

    if feedback_value == "none":
        ChatFeedback.objects.filter(message=message, user=user).delete()
        return None

    raise ValidationError("Unsupported feedback value")
```

### Retry metadata helper

```python
def mark_retry_metadata(*, assistant_message, retry_of_message_id):
    extra = dict(assistant_message.extra or {})
    extra["action"] = "retry"
    extra["retry_of_message_id"] = retry_of_message_id
    assistant_message.extra = extra
    assistant_message.save(update_fields=["extra", "updated_at"])
```

## Recommended Endpoint Contract

### 1. Message feedback

`POST /chat/sessions/{session_uuid}/message/{message_id}/feedback/`

Request:

```json
{
  "feedback": "up",
  "meta": {
    "action": "like"
  }
}
```

Success response:

```json
{
  "status": "success",
  "session_uuid": "....",
  "message_id": 123,
  "feedback": "up"
}
```

### 2. Retry tracking

No new endpoint is required if the existing log-turn flow already persists message `extra` or turn metadata.

On retry, ensure the backend persists:

```json
{
  "action": "retry",
  "retry_of_message_id": 123
}
```

If the existing log-turn endpoint currently ignores this metadata, update the Django service handling `POST /chat/log-turn/` so that the assistant-side `ChatMessage.extra` stores these keys.

## Reporting Queries

### Count likes per session

```python
ChatFeedback.objects.filter(
    message__session__session_uuid=session_uuid,
    rating=1,
).count()
```

### Count retries per session from `ChatMessage.extra`

```python
ChatMessage.objects.filter(
    session__session_uuid=session_uuid,
    role="assistant",
    extra__action="retry",
).count()
```

### Count retries per user

```python
ChatMessage.objects.filter(
    user=user,
    role="assistant",
    extra__action="retry",
).count()
```

## Operational Recommendation

Implement in this order:

1. Django feedback endpoint backed by `ChatFeedback`
2. Django log-turn persistence of retry metadata into `ChatMessage.extra`
3. Reporting queries or admin views
4. Optional future `ChatEvent` model if analytics needs expand

## Decision Summary

- `like`: use existing `ChatFeedback`, no counter field needed
- `try_again`: store as raw metadata, no counter field needed
- `copy` and `read aloud`: do not persist as analytics
- no denormalized MySQL counter columns should be added at this stage

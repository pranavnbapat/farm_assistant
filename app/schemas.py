# app/schemas.py

from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


class AskIn(BaseModel):
    question: str
    page: Optional[int] = Field(default=None, examples=[1])
    k: Optional[int] = Field(default=None, examples=[5])
    model: Optional[str] = None
    include_fulltext: Optional[bool] = Field(default=None, examples=[True])
    sort_by: Optional[str] = Field(default=None, examples=["score_desc"])
    dev: Optional[bool] = Field(default=False, examples=[False])
    max_tokens: Optional[int] = Field(default=None, examples=[2000])
    temperature: Optional[float] = Field(default=None, examples=[0.4])
    top_k: Optional[int] = Field(default=None, examples=[4])

class SourceItem(BaseModel):
    id: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None
    score: Optional[float] = None
    subtitle: Optional[str] = None
    description: Optional[str] = None
    project: Optional[str] = None
    license: Optional[str] = None
    keywords: Optional[list[str]] = None
    topics: Optional[list[str]] = None
    themes: Optional[list[str]] = None
    languages: Optional[list[str]] = None
    creators: Optional[list[str]] = None
    date_of_completion: Optional[str] = None
    display_url: Optional[str] = None
    sid: str | None = None

class AskOut(BaseModel):
    answer: str
    used_context: List[str]
    sources: List[SourceItem] = []
    meta: Dict[str, Any] = {}

class SummariseIn(BaseModel):
    prompt: str
    text: str
    max_tokens: Optional[int] = Field(default=None, examples=[-1])
    temperature: Optional[float] = Field(default=None, examples=[0.4])

class SummariseOut(BaseModel):
    summary: str
    meta: Dict[str, Any] = {}


class ChatSessionCreateIn(BaseModel):
    title: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatSessionPatchIn(BaseModel):
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatTurnLogIn(BaseModel):
    session_uuid: Optional[str] = None
    user_message: str
    assistant_message: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class LogoutIn(BaseModel):
    email: str


class ChatMessageStreamIn(BaseModel):
    q: str
    page: int = Field(default=1, examples=[1])
    k: Optional[int] = Field(default=None, examples=[5])
    top_k: Optional[int] = Field(default=None, examples=[4])
    max_tokens: Optional[int] = Field(default=None, examples=[-1])
    temperature: Optional[float] = Field(default=None, examples=[0.4])
    model: Optional[str] = None
    followup_hint: Optional[str] = None
    doc_ids: List[str] = Field(default_factory=list)


class UserProfilePatchIn(BaseModel):
    expertise_level: Optional[str] = None
    farm_type: Optional[str] = None
    region: Optional[str] = None
    preferred_language: Optional[str] = None
    communication_style: Optional[str] = None
    crops_list: Optional[List[str]] = None
    common_topics: Optional[List[str]] = None
    query_languages: Optional[List[str]] = None
    total_queries: Optional[int] = None


class UserFactCreateIn(BaseModel):
    fact_category: str
    fact_text: str
    confidence_score: float = Field(default=1.0, examples=[0.9])
    source_session_uuid: Optional[str] = None


class UserProfileBuildIn(BaseModel):
    session_uuid: str
    user_message: str
    assistant_message: str = ""

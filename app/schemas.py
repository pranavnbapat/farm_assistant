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

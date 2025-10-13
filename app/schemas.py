from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class AskIn(BaseModel):
    question: str
    page: Optional[int] = None
    k: Optional[int] = None
    model: Optional[str] = None
    include_fulltext: Optional[bool] = None
    sort_by: Optional[str] = None
    dev: Optional[bool] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None

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

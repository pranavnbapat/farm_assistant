# app/services/web_search_service.py
"""
Trusted-allowlist web-search fallback for the `normal` retrieval turn.

The backend performs the search and content extraction; the LLM never browses
directly. Output mirrors `context_service.build_context_and_sources` exactly
(`[SID]`-headed context strings + `SourceItem`s, kept length-aligned) so the
router, citation numbering, and the `sources`/`grounding` SSE events are unchanged.

Search runs through an ordered provider chain (`WEB_SEARCH_PROVIDERS`), tried left
to right: a provider is skipped when its API key is missing, and the chain advances
to the next on any error (quota/rate-limit/network) or when it returns no
allowlisted results. Tavily returns pre-cleaned page text, so its results skip the
fetch+extract step; Brave/DuckDuckGo return links+snippets that we fetch and extract
with `trafilatura`.

Optional deps (`ddgs`, `trafilatura`) are imported lazily so instances that never
enable WEB_FALLBACK_ENABLED don't need them installed.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.schemas import SourceItem
from app.services.context_service import rank_paragraphs, split_paragraphs

logger = logging.getLogger("farm-assistant.web")
S = get_settings()

# Per-source caps mirror the KO path (context_service.PER_PARENT_CHAR_CAP=3500).
_PER_SOURCE_CHAR_CAP = 2000
_USER_AGENT = (
    "Mozilla/5.0 (compatible; EU-FarmBook-Assistant/1.0; +https://eufarmbook.eu)"
)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def _registrable_host(url: str) -> str:
    """Lowercased network host without a leading 'www.'."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _host_allowed(url: str, allowlist: List[str]) -> bool:
    """True when the URL host is, or is a subdomain of, an allowlisted domain."""
    host = _registrable_host(url)
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in allowlist)


# --- Providers -------------------------------------------------------------
# Each provider returns a list of dicts: {title, url, snippet, text}. `text` is
# pre-extracted full page content when the provider supplies it (Tavily); empty
# otherwise, in which case the caller fetches + extracts the page.


class WebSearchProvider:
    name = "base"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        raise NotImplementedError


class TavilyProvider(WebSearchProvider):
    """Tavily Search API. Enforces the allowlist server-side via include_domains and
    returns cleaned page content (raw_content), so its results skip our fetch step."""

    name = "tavily"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        payload: Dict[str, object] = {
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_raw_content": True,
        }
        if allowlist:
            payload["include_domains"] = allowlist
        headers = {
            "Authorization": f"Bearer {S.TAVILY_API_KEY}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(S.WEB_FETCH_TIMEOUT + 6.0)  # raw-content extraction is slower
        async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
            res = await client.post(_TAVILY_ENDPOINT, json=payload, headers=headers)
            res.raise_for_status()
            data = res.json()

        out: List[Dict[str, str]] = []
        for r in data.get("results", []) or []:
            url = (r.get("url") or "").strip()
            if not url:
                continue
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": url,
                "snippet": (r.get("content") or "").strip(),
                "text": (r.get("raw_content") or "").strip(),
            })
        return out


class BraveProvider(WebSearchProvider):
    """Brave Search API. No server-side domain filter, so the allowlist is applied
    via `site:` query scoping plus the central host post-filter. Links + snippets
    only — the caller fetches + extracts each page."""

    name = "brave"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        site_filter = " OR ".join(f"site:{d}" for d in allowlist) if allowlist else ""
        scoped_query = f"{query} ({site_filter})" if site_filter else query
        params = {"q": scoped_query, "count": max(max_results * 3, max_results + 4)}
        headers = {"X-Subscription-Token": S.BRAVE_API_KEY or "", "Accept": "application/json"}
        timeout = httpx.Timeout(S.WEB_FETCH_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
            res = await client.get(_BRAVE_ENDPOINT, params=params, headers=headers)
            res.raise_for_status()
            data = res.json()

        out: List[Dict[str, str]] = []
        for r in ((data.get("web") or {}).get("results") or []):
            url = (r.get("url") or "").strip()
            if not url:
                continue
            out.append({
                "title": (r.get("title") or "").strip(),
                "url": url,
                "snippet": (r.get("description") or "").strip(),
                "text": "",
            })
        return out


class DuckDuckGoProvider(WebSearchProvider):
    """Keyless DuckDuckGo search via the `ddgs` package. Links + snippets only."""

    name = "duckduckgo"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        try:
            from ddgs import DDGS  # lazy: only needed when fallback is enabled
        except ImportError as e:  # pragma: no cover - environment-dependent
            logger.warning("ddgs not installed; DuckDuckGo provider unavailable: %s", e)
            return []

        site_filter = " OR ".join(f"site:{d}" for d in allowlist) if allowlist else ""
        scoped_query = f"{query} ({site_filter})" if site_filter else query
        fetch_n = max(max_results * 4, max_results + 4)

        def _run() -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            with DDGS() as ddgs:
                for r in ddgs.text(scoped_query, max_results=fetch_n):
                    url = r.get("href") or r.get("url") or ""
                    if not url:
                        continue
                    out.append({
                        "title": (r.get("title") or "").strip(),
                        "url": url,
                        "snippet": (r.get("body") or r.get("snippet") or "").strip(),
                        "text": "",
                    })
            return out

        return await asyncio.to_thread(_run)


class StaanProvider(WebSearchProvider):
    """EUSP 'Staan' European search index (Qwant + Ecosia joint venture) — the
    EU-sovereign option, opened publicly in 2026 for AI workflows.

    The auth + skip wiring is complete: the provider stays dormant until STAAN_API_KEY
    is set (see `_make_provider`). The request/response mapping below is a PLACEHOLDER —
    the public Staan search API shape was not verified at implementation time, and a
    web result claiming `api.staan.com` referred to an unrelated product. Confirm the
    real endpoint, params, and JSON fields against the official EUSP Staan API docs,
    then fill in `search()` returning {title, url, snippet, text} dicts. Until then a
    keyed call raises and the chain transparently falls back to the next provider.
    """

    name = "staan"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        raise NotImplementedError(
            "Staan provider endpoint not yet wired. Set STAAN_API_KEY and implement the "
            "request/response mapping per the official EUSP Staan API docs. The chain "
            "falls back to the next provider until this is done."
        )


class WikipediaProvider(WebSearchProvider):
    """Keyless institutional/foundational source via the MediaWiki API. Returns the
    article intro as clean plain text (no fetch+extract needed). Multilingual, but we
    query English Wikipedia to match the English-only retrieval pipeline."""

    name = "wikipedia"
    _ENDPOINT = "https://en.wikipedia.org/w/api.php"

    async def search(self, query: str, *, max_results: int, allowlist: List[str]) -> List[Dict[str, str]]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": max_results,
            "prop": "extracts|info",
            "exintro": 1,
            "explaintext": 1,
            "inprop": "url",
            "redirects": 1,
        }
        headers = {"User-Agent": _USER_AGENT}
        timeout = httpx.Timeout(S.WEB_FETCH_TIMEOUT)
        async with httpx.AsyncClient(timeout=timeout, verify=S.VERIFY_SSL) as client:
            res = await client.get(self._ENDPOINT, params=params, headers=headers)
            res.raise_for_status()
            data = res.json()

        pages = ((data.get("query") or {}).get("pages") or {})
        # MediaWiki returns an "index" per page reflecting search rank; sort by it.
        ordered = sorted(pages.values(), key=lambda p: p.get("index", 1_000_000))
        out: List[Dict[str, str]] = []
        for p in ordered:
            page_url = (p.get("fullurl") or "").strip()
            extract = (p.get("extract") or "").strip()
            if not page_url or not extract:
                continue
            out.append({
                "title": (p.get("title") or "").strip(),
                "url": page_url,
                "snippet": extract[:300],
                "text": extract,
            })
        return out


def _make_provider(name: str) -> Optional[WebSearchProvider]:
    """Instantiate a provider by name, or None when it isn't configured (missing key)."""
    name = (name or "").strip().lower()
    if name == "staan":
        return StaanProvider() if S.STAAN_API_KEY else None
    if name == "tavily":
        return TavilyProvider() if S.TAVILY_API_KEY else None
    if name == "brave":
        return BraveProvider() if S.BRAVE_API_KEY else None
    if name == "duckduckgo":
        return DuckDuckGoProvider()
    if name == "wikipedia":
        return WikipediaProvider()
    logger.warning("Unknown web search provider '%s' in WEB_SEARCH_PROVIDERS; skipping", name)
    return None


async def _chain_search(
    query: str, *, max_results: int, allowlist: List[str]
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Try each configured provider in order; return the first non-empty, allowlisted
    result set and the provider name that served it."""
    for name in S.web_search_providers_list():
        provider = _make_provider(name)
        if provider is None:
            logger.debug("Web provider '%s' not configured; skipping", name)
            continue
        try:
            raw = await provider.search(query, max_results=max_results, allowlist=allowlist)
        except Exception as e:
            logger.warning("Web provider '%s' failed (%s); falling back to next", name, e)
            continue
        results = [r for r in raw if _host_allowed(r.get("url", ""), allowlist)]
        if results:
            logger.info("Web search served by provider='%s' (%d allowlisted result(s))", name, len(results))
            return results, name
        logger.info("Web provider '%s' returned no allowlisted results; trying next", name)
    return [], None


# --- Content fetch / extraction (for providers without raw text) -----------


def _extract_main_text(html: str, url: str) -> str:
    """Best-effort main-content extraction. Returns "" on failure."""
    try:
        import trafilatura  # lazy: only needed when fallback is enabled
    except ImportError as e:  # pragma: no cover - environment-dependent
        logger.warning("trafilatura not installed; cannot extract web content: %s", e)
        return ""
    try:
        text = trafilatura.extract(
            html, url=url, include_comments=False, include_tables=False, favor_recall=True
        )
    except Exception as e:
        logger.debug("trafilatura extraction failed for %s: %s", url, e)
        return ""
    return (text or "").strip()


async def _fetch(client: httpx.AsyncClient, url: str) -> Optional[str]:
    try:
        res = await client.get(url, follow_redirects=True)
        res.raise_for_status()
        ctype = res.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            return None
        return res.text
    except Exception as e:
        logger.debug("Web fetch failed for %s: %s", url, e)
        return None


def _build_source(result: Dict[str, str], extracted: str, sid: str) -> Tuple[str, SourceItem]:
    """Build one `[SID]`-headed context string + matching SourceItem."""
    title = result.get("title") or _registrable_host(result["url"])
    domain = _registrable_host(result["url"])
    snippet = result.get("snippet") or ""

    paras = split_paragraphs(extracted)
    ranked = rank_paragraphs(paras, question=title)
    chosen: List[str] = []
    for _, p in ranked:
        if len(p) < 120:
            continue
        chosen.append(p[:800])
        if sum(len(x) for x in chosen) > _PER_SOURCE_CHAR_CAP - 400:
            break

    body = "\n- ".join(chosen) if chosen else (snippet or extracted[:800])

    header_parts = [f"[{sid}]"]
    if title:
        header_parts.append(f"Title: {title}")
    header_parts.append(f"Source: {domain} (external trusted reference)")
    header = "\n".join(header_parts)
    chunk = (f"{header}\nContent:\n- {body}").strip()[:_PER_SOURCE_CHAR_CAP]

    source = SourceItem(
        id=result["url"],
        url=result["url"],
        display_url=result["url"],
        title=title,
        score=None,
        description=(snippet[:300] or None),
        project=domain or None,
        license=None,
        sid=sid,
    )
    return chunk, source


async def web_search_and_build_contexts(
    query: str,
    *,
    max_results: int,
    max_chars: int,
    sid_offset: int = 0,
) -> Tuple[List[str], List[SourceItem]]:
    """
    Search the trusted allowlist for `query` via the provider chain, extract clean
    passages, and return (contexts, sources) in the same shape the KO path produces.
    `sid_offset` is the number of sources already present so SID numbering continues
    uniquely. Returns ([], []) on any total failure — the caller treats that as
    "no web grounding".
    """
    if max_chars <= 0:
        return [], []

    allowlist = S.web_trusted_domains_list()

    try:
        results, provider_name = await _chain_search(query, max_results=max_results, allowlist=allowlist)
    except Exception as e:
        logger.warning("Web search chain error: %s", e)
        return [], []

    if not results:
        logger.info("Web fallback: no results from any configured provider")
        return [], []

    # Dedupe by registrable domain, keep first (highest-ranked) per domain.
    seen_domains = set()
    deduped: List[Dict[str, str]] = []
    for r in results:
        dom = _registrable_host(r["url"])
        if dom in seen_domains:
            continue
        seen_domains.add(dom)
        deduped.append(r)
        if len(deduped) >= max_results:
            break

    # Fetch + extract only the results the provider didn't already supply text for.
    to_fetch = [r for r in deduped if not r.get("text")]
    fetched: Dict[str, str] = {}
    if to_fetch:
        timeout = httpx.Timeout(S.WEB_FETCH_TIMEOUT)
        headers = {"User-Agent": _USER_AGENT}
        async with httpx.AsyncClient(timeout=timeout, headers=headers, verify=S.VERIFY_SSL) as client:
            htmls = await asyncio.gather(*[_fetch(client, r["url"]) for r in to_fetch])
        for r, html in zip(to_fetch, htmls):
            fetched[r["url"]] = _extract_main_text(html, r["url"]) if html else ""

    contexts: List[str] = []
    sources: List[SourceItem] = []
    total = 0
    for r in deduped:
        extracted = r.get("text") or fetched.get(r["url"], "")
        # Keep snippet-only sources too: a clean snippet is still cited grounding.
        if not extracted and not r.get("snippet"):
            continue
        sid = f"S{sid_offset + len(sources) + 1}"
        chunk, source = _build_source(r, extracted, sid)
        if not chunk:
            continue
        if total + len(chunk) > max_chars:
            break
        contexts.append(chunk)
        sources.append(source)
        total += len(chunk)

    logger.info(
        "Web fallback (provider=%s): %d source(s) over %d char(s) from %d candidate(s)",
        provider_name, len(contexts), total, len(deduped),
    )
    return contexts, sources
